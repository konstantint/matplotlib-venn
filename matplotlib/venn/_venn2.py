'''
Venn diagram plotting routines.
Two-circle venn plotter.

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''
import numpy as np
import warnings

from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.colors import ColorConverter
from matplotlib.pyplot import gca

from _math import *

from _venn3 import make_venn3_region_patch, prepare_venn3_axes
make_venn2_region_patch = make_venn3_region_patch
prepare_venn2_axes = prepare_venn3_axes



def compute_venn2_areas(diagram_areas, normalize_to=1.0):
    '''
    The list of venn areas is given as 3 values, corresponding to venn diagram areas in the following order:
     (Ab, aB, AB)  (i.e. last element corresponds to the size of intersection A&B&C).
    The return value is a list of areas (A, B, AB), such that the total area is normalized 
    to normalize_to. If total area was 0, returns
    (1.0, 1.0, 0.0)/2.0
    
    Assumes all input values are nonnegative (to be more precise, all areas are passed through and abs() function)
    >>> compute_venn2_areas((1, 1, 0))
    (0.5, 0.5, 0.0)
    >>> compute_venn2_areas((0, 0, 0))
    (0.5, 0.5, 0.0)
    >>> compute_venn2_areas((1, 1, 1), normalize_to=3)
    (2.0, 2.0, 1.0)
    >>> compute_venn2_areas((1, 2, 3), normalize_to=6)
    (4.0, 5.0, 3.0)
    '''
    # Normalize input values to sum to 1
    areas = np.array(np.abs(diagram_areas), float)
    total_area = np.sum(areas)
    if np.abs(total_area) < tol:
        return (0.5, 0.5, 0.0)
    else:
        areas = areas/total_area*normalize_to
        return (areas[0] + areas[2], areas[1] + areas[2], areas[2])        

def solve_venn2_circles(venn_areas):
    '''
    Given the list of "venn areas" (as output from compute_venn2_areas, i.e. [A, B, AB]),
    finds the positions and radii of the two circles.
    The return value is a tuple (coords, radii), where coords is a 2x2 array of coordinates and
    radii is a 2x1 array of circle radii.

    Assumes the input values to be nonnegative and not all zero. 
    In particular, the first two values must be positive.
    
    >>> c, r = solve_venn2_circles((1, 1, 0))
    >>> np.round(r, 3)
    array([ 0.564,  0.564])
    >>> c, r = solve_venn2_circles(compute_venn2_areas((1, 2, 3)))
    >>> np.round(r, 3)
    array([ 0.461,  0.515])
    '''
    (A_a, A_b, A_ab) = map(float, venn_areas)
    r_a, r_b = np.sqrt(A_a/np.pi), np.sqrt(A_b/np.pi)
    radii = np.array([r_a, r_b])
    if A_ab > tol:
        # Nonzero intersection
        coords = np.zeros((2,2))
        coords[1][0] = find_distance_by_area(radii[0], radii[1], A_ab)
    else:
        # Zero intersection
        coords = np.zeros((2,2))
        coords[1][0] = radii[0] + radii[1] + np.mean(radii)*1.1
    coords = normalize_by_center_of_mass(coords, radii)
    return (coords, radii)

def compute_venn2_regions(centers, radii):
    '''
    See compute_venn3_regions for explanations.
    >>> centers, radii = solve_venn2_circles((1, 1, 0.5))
    >>> regions = compute_venn2_regions(centers, radii)
    '''
    intersection = circle_circle_intersection(centers[0], radii[0], centers[1], radii[1])
    if intersection is None:
        # Two circular regions
        regions = [("CIRCLE", (centers[a], radii[a], True), centers[a]) for a in [0, 1]] + [None]
    else:
        # Three curved regions
        regions = []
        for (a, b) in [(0, 1), (1, 0)]:
            # Make region a&not b:  [(AB, A-), (BA, B+)]
            points = np.array([intersection[a], intersection[b]])
            arcs = [(centers[a], radii[a], False), (centers[b], radii[b], True)]
            if centers[a][0] < centers[b][0]:
                # We are to the left
                label_pos_x = (centers[a][0] - radii[a] + centers[b][0] - radii[b])/2.0
            else:
                # We are to the right
                label_pos_x = (centers[a][0] + radii[a] + centers[b][0] + radii[b])/2.0
            label_pos = np.array([label_pos_x, centers[a][1]])
            regions.append((points, arcs, label_pos))
        
        # Make region a&b: [(AB, A+), (BA, B+)]
        (a, b) = (0, 1)
        points = np.array([intersection[a], intersection[b]])
        arcs = [(centers[a], radii[a], True), (centers[b], radii[b], True)]
        label_pos_x = (centers[a][0] + radii[a] + centers[b][0] - radii[b])/2.0
        label_pos = np.array([label_pos_x, centers[a][1]])
        regions.append((points, arcs, label_pos))
    return regions

def compute_venn2_colors(set_colors):
    '''
    Given two base colors, computes combinations of colors corresponding to all regions of the venn diagram.
    returns a list of 3 elements, providing colors for regions (10, 01, 11).
    
    >>> compute_venn2_colors(('r', 'g'))
    (array([ 1.,  0.,  0.]), array([ 0. ,  0.5,  0. ]), array([ 0.7 ,  0.35,  0.  ]))
    '''
    ccv = ColorConverter()
    base_colors = [np.array(ccv.to_rgb(c)) for c in set_colors]
    return (base_colors[0], base_colors[1], 0.7*(base_colors[0] + base_colors[1]))

def venn2_circles(subsets, normalize_to=1.0, alpha=1.0, color='black', linestyle='solid', linewidth=2.0, **kwargs):
    '''
    Plots only the two circles for the corresponding Venn diagram. 
    Useful for debugging or enhancing the basic venn diagram.
    parameters sets and normalize_to are the same as in venn2()
    kwargs are passed as-is to matplotlib.patches.Circle.
    returns a list of three Circle patches.
    
    >>> c = venn2_circles((1, 2, 3))
    '''
    if isinstance(subsets, dict):
        subsets = [s.get(t, 0) for t in ['10', '01', '11']]    
    areas = compute_venn2_areas(subsets, normalize_to)
    centers, radii = solve_venn2_circles(areas)
    ax = gca()
    prepare_venn2_axes(ax, centers, radii)
    result = []
    for (c, r) in zip(centers, radii):
        circle = Circle(c, r, alpha=alpha, edgecolor=color, facecolor='none', linestyle=linestyle, linewidth=linewidth, **kwargs)
        ax.add_patch(circle)
        result.append(circle)
    return result


class Venn2:
    '''
    A container for a set of patches and patch labels and set labels, which make up the rendered venn diagram.
    '''
    id2idx = {'10':0,'01':1,'11':2,'A':0, 'B':1} 
    def __init__(self, patches, subset_labels, set_labels):
        self.patches = patches
        self.subset_labels = subset_labels
        self.set_labels = set_labels
    def get_patch_by_id(self, id):
        '''Returns a patch by a "region id". A region id is a string '10', '01' or '11'.'''
        return self.patches[self.id2idx[id]]
    def get_label_by_id(self, id):
        '''
        Returns a subset label by a "region id". A region id is a string '10', '01' or '11'.
        Alternatively, if the string 'A' or 'B' is given, the label of the 
        corresponding set is returned (or None).'''
        if len(id) == 1:
            return self.set_labels[self.id2idx[id]] if self.set_labels is not None else None
        else:
            return self.subset_labels[self.id2idx[id]]

def venn2(subsets, set_labels = ('A', 'B'), set_colors=('r', 'g'), alpha=0.4, normalize_to=1.0):
    '''Plots a 2-set area-weighted Venn diagram.
    The subsets parameter is either a dict or a list.
     - If it is a dict, it must map regions to their sizes.
       The regions are identified via two-letter binary codes ('10', '01', and '11'), hence a valid set could look like:
       {'01': 10, '01': 20, '11': 40}. Unmentioned codes are considered to map to 0.
     - If it is a list, it must have 3 elements, denoting the sizes of the regions in the following order:
       (10, 10, 11)
    
    Set labels parameter is a list of two strings - set labels. Set it to None to disable set labels.
    The set_colors parameter should be a list of two elements, specifying the "base colors" of the two circles.
    The color of circle intersection will be computed based on those.
    
    The normalize_to parameter specifies the total (on-axes) area of the circles to be drawn. Sometimes tuning it (together 
    with the overall fiture size) may be useful to fit the text labels better.
    The return value is a Venn2 object, that keeps references to the Text and Patch objects used on the plot.
    
    >>> from matplotlib.venn import *
    >>> v = venn2(subsets=(1, 1, 1), set_labels = ('A', 'B'))
    >>> c = venn2_circles(subsets=(1, 1, 1), linestyle='dashed')
    >>> v.get_patch_by_id('10').set_alpha(1.0)
    >>> v.get_patch_by_id('10').set_color('white')
    >>> v.get_label_by_id('10').set_text('Unknown')
    >>> v.get_label_by_id('A').set_text('Set A')
    '''
    if isinstance(subsets, dict):
        subsets = [s.get(t, 0) for t in ['10', '01', '11']]    
    areas = compute_venn2_areas(subsets, normalize_to)
    centers, radii = solve_venn2_circles(areas)
    if (areas[0] < tol or areas[1] < tol):
        raise Exception("Both circles in the diagram must have positive areas.")
    centers, radii = solve_venn2_circles(areas)
    regions = compute_venn2_regions(centers, radii)
    colors = compute_venn2_colors(set_colors)
    
    ax = gca()
    prepare_venn2_axes(ax, centers, radii)
    # Create and add patches and text
    patches = [make_venn2_region_patch(r) for r in regions]
    for (p, c) in zip(patches, colors):
        if p is not None:
            p.set_facecolor(c)
            p.set_edgecolor('none')
            p.set_alpha(alpha)
            ax.add_patch(p)
    texts = [ax.text(r[2][0], r[2][1], str(s), va='center', ha='center') if r is not None else None  for (r, s) in zip(regions, subsets)]
    
    # Position labels
    if set_labels is not None:
        padding = np.mean([r * 0.1 for r in radii])
        label_positions = [centers[0] + np.array([0.0, - radii[0] - padding]),
                           centers[1] + np.array([0.0, - radii[1] - padding])]
        labels = [ax.text(pos[0], pos[1], txt, size='large', ha='right', va='top') for (pos, txt) in zip(label_positions, set_labels)]
        labels[1].set_ha('left')
    else:
        labels = None
    return Venn2(patches, texts, labels)
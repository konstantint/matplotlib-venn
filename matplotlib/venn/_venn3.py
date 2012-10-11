'''
Venn diagram plotting routines.
Three-circle venn plotter.

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under BSD.
'''
import numpy as np
import warnings
from _math import *

tol = 1e-10
    
def solve_venn3_circles(areas, normalize_to=1.0):
    '''
    The list of venn areas is given as 8 values, corresponding to venn diagram areas in the following order:
     (abc, Abc, aBc, ABc, abC, AbC, aBC, ABC)  
    (i.e. last element corresponds to the size of intersection A&B&C).
    The return value is a list (r_a, r_b, r_c, d_ab, d_ac, d_bc), denoting the radii of the three circles
    and the distances between their centers.
    Assumes all input values are nonnegative.
    Returns circles, such that their total area is normalized to <normalize_to>.
    Yes, the first value in the provided list is not used at all in this method.
    Yes, the overall match is only approximate (to be precise, what is matched are the areas of the circles and the 
    three pairwise intersections).
    
    >>> solve_venn3_circles((0, 1, 1, 0, 1, 0, 0, 0))
    (0.3257..., 0.3257..., 0.3257..., 0.6514..., 0.6514..., 0.6514...)
    >>> solve_venn3_circles((0, 1, 2, 40, 30, 4, 40, 4))
    (0.359..., 0.475..., 0.452..., 0.198..., 0.435..., 0.345...)
    '''
    # Normalize input values to sum to 1
    areas = np.array(areas[1:], float)
    total_area = np.sum(areas)
    if np.abs(total_area) < tol:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    areas = areas/total_area*normalize_to
    
    # Compute areas of the three circles
    A_a = areas[0] + areas[2] + areas[4] + areas[6]
    A_b = areas[1] + areas[2] + areas[5] + areas[6]
    A_c = areas[3] + areas[4] + areas[5] + areas[6]
    r_a, r_b, r_c = np.sqrt(A_a/np.pi), np.sqrt(A_b/np.pi), np.sqrt(A_c/np.pi)
    
    # Compute areas of the three intersections (ab, ac, bc)
    A_ab, A_ac, A_bc = areas[2] + areas[6], areas[4] + areas[6], areas[5] + areas[6]
    d_ab = find_distance_by_area(r_a, r_b, A_ab)
    d_ac = find_distance_by_area(r_a, r_c, A_ac)
    d_bc = find_distance_by_area(r_b, r_c, A_bc)
    
    # Ad-hoc fix to ensure that resulting circles can be at all positioned
    if d_ab > d_bc + d_ac:
        d_ab = 0.8*(d_ac + d_bc)
        warnings.warn("Bad circle positioning")
    if d_bc > d_ab + d_ac:
        d_bc = 0.8*(d_ab + d_ac)
        warnings.warn("Bad circle positioning")
    if d_ac > d_ab + d_bc:
        d_ac = 0.8*(d_ab + d_bc)
        warnings.warn("Bad circle positioning")
    return (r_a, r_b, r_c, d_ab, d_ac, d_bc)

def position_venn3_circles(r_a, r_b, r_c, d_ab, d_ac, d_bc):
    '''
    Given radii and distances between the circles (the output from solve_venn3_circles),
    finds the coordinates of the centers for the three circles. Returns a 3x2 array with circle center coordinates in rows.
    Circles are positioned so that the center of mass is at (0, 0), the centers of A and B are on a horizontal line, and C is just below.
    
    >>> position_venn3_circles(1, 1, 1, 0, 0, 0)
    array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0., -0.]])
    >>> position_venn3_circles(1, 1, 1, 2, 2, 2)
    array([[-1.        ,  0.577...],
           [ 1.        ,  0.577...],
           [ 0.        , -1.154...]])
    '''
    coords = np.array([[0, 0], [d_ab, 0], [0, 0]], float)
    C_x = (d_ac**2 - d_bc**2 + d_ab**2)/2.0/d_ab if np.abs(d_ab) > tol else 0.0
    C_y = -np.sqrt(d_ac**2 - C_x**2)
    coords[2,:] = C_x, C_y
    
    # Now find the center of mass.
    r_a2, r_b2, r_c2 = r_a**2, r_b**2, r_c**2
    if np.abs(r_a2 + r_b2 + r_c2) < tol:
        cmass = array([0.0, 0.0])
    else:
        cmass = (r_a2 * coords[0] + r_b2 * coords[1] + r_c2 * coords[2])/(r_a2 + r_b2 + r_c2)
    for i in range(3):
        coords[i] = coords[i] - cmass
    return coords

def compute_venn3_regions(centers, radii):
    '''
    Given the 3x2 matrix with circle center coordinates and a 3-element list (or array) with circle radii,
    returns the 7 regions, comprising the venn diagram.
    Each region is given as [array([pt_1, pt_2, pt_3]), (arc_1, arc_2, arc_3), label_pos] where each pt_i gives the coordinates of a point,
    and each arc_i is in turn a triple (circle_center, circle_radius, direction), and label_pos is the recommended center point for 
    positioning region label.
    The region is the poly-curve constructed by moving from pt_1 to pt_2 along arc_1, then to pt_3 along arc_2 and back to pt_1 along arc_3.
    Arc direction==True denotes positive (CCW) direction.
    
    Regions are returned in order (None, Abc, aBc, ABc, abC, AbC, aBC, ABC) (i.e. first element of the result list is None)
    
    >>> circ = solve_venn3_circles((0, 1, 1, 1, 1, 1, 1, 1))
    >>> centers = position_venn3_circles(*circ)
    >>> regions = compute_venn3_regions(centers, circ[0:3])
    '''
    # First compute all pairwise circle intersections
    intersections = [circle_circle_intersection(centers[i], radii[i], centers[j], radii[j]) for (i, j) in [(0, 1), (1, 2), (2, 0)]]
    regions = []
    # Regions [Abc, aBc, abC]
    for i in range(3):
        (a, b, c) = (i, (i+1)%3, (i+2)%3)
        
        if np.linalg.norm(intersections[b][0] - centers[a]) < radii[a]:
            # In the "normal" situation we use the scheme [(BA, B+), (BC, C+), (AC, A-)]
            points = np.array([intersections[a][1], intersections[b][0], intersections[c][1]])
            arcs = [(centers[b], radii[b], True), (centers[c], radii[c], True), (centers[a], radii[a], False)]
            
            # Ad-hoc label positioning
            pt_a = intersections[b][0]
            pt_b = intersections[b][1]
            pt_c = circle_line_intersection(centers[a], radii[a], pt_a, pt_b)
            if pt_c is None:
                label_pos = circle_circle_intersection(centers[b], radii[b] + 0.1*radii[a], centers[c], radii[c] + 0.1*radii[c])[0]
            else:
                label_pos = 0.5*(pt_c[1] + pt_a)
        else:
            # This is the "bad" situation (basically one disc covers two touching disks)
            # We use the scheme [(BA, B+), (AB, A-)] if (AC is inside B) and
            #                   [(CA, C+), (AC, A-)] otherwise
            if np.linalg.norm(intersections[c][0] - centers[b]) < radii[b]:
                points = np.array([intersections[a][1], intersections[a][0]])
                arcs = [(centers[b], radii[b], True), (centers[a], radii[a], False)]
            else:
                points = np.array([intersections[c][0], intersections[c][1]])
                arcs = [(centers[c], radii[c], True), (centers[a], radii[a], False)]
            label_pos = centers[a]
        regions.append((points, arcs, label_pos))

    (a, b, c) = (0, 1, 2)
    has_middle_region = np.linalg.norm(intersections[b][0] - centers[a]) < radii[a]
        
    # Regions [aBC, AbC, ABc]
    for i in range(3):
        (a, b, c) = (i, (i+1)%3, (i+2)%3)
        
        if has_middle_region:
            # This is the "normal" situation (i.e. all three circles have a common area)
            # We then use the scheme [(CB, C+), (CA, A-), (AB, B+)]
            points = np.array([intersections[b][1], intersections[c][0], intersections[a][0]])
            arcs = [(centers[c], radii[c], True), (centers[a], radii[a], False), (centers[b], radii[b], True)]
            # Ad-hoc label positioning
            pt_a = intersections[b][1]
            dir_to_a = pt_a - centers[a]
            dir_to_a = dir_to_a / np.linalg.norm(dir_to_a)
            pt_b = centers[a] + dir_to_a*radii[a]
            label_pos  = 0.5*(pt_a + pt_b)
        else:
            # This is the "bad" situation, where there is no common area
            # Then the corresponding area is made by scheme [(CB, C+), (BC, B+), None]
            points = np.array([intersections[b][1], intersections[b][0]])
            arcs = [(centers[c], radii[c], True), (centers[b], radii[b], True)]
            label_pos  = 0.5*(intersections[b][1] + intersections[b][0])
            
        regions.append((points, arcs, label_pos))
    
    # Central region made by scheme [(BC, B+), (AB, A+), (CA, C+)]
    (a, b, c) = (0, 1, 2)
    points = np.array([intersections[b][0], intersections[a][0], intersections[c][0]])
    label_pos = np.mean(points, 0) # Middle of the central region
    arcs = [(centers[b], radii[b], True), (centers[a], radii[a], True), (centers[c], radii[c], True)]
    if has_middle_region:
        regions.append((points, arcs, label_pos))
    else:
        regions.append(([], [], label_pos))
    
    #      (None, Abc,        aBc,        ABc,        abC,        AbC,        aBC,        ABC) 
    return (None, regions[0], regions[1], regions[5], regions[2], regions[4], regions[3], regions[6])

from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle
from matplotlib.text import Text
from matplotlib.pyplot import gca
from matplotlib.colors import ColorConverter

def make_venn3_region_patch(region):
    '''
    Given a venn3 region (as returned from compute_venn3_regions) produces a Patch object,
    depicting the region as a curve.
    
    >>> circ = solve_venn3_circles((0, 1, 1, 1, 1, 1, 1, 1))
    >>> centers = position_venn3_circles(*circ)
    >>> regions = compute_venn3_regions(centers, circ[0:3])
    >>> patches = [make_venn3_region_patch(r) for r in regions]
    '''
    if region is None or len(region[0]) == 0:
        return None
    pts, arcs, label_pos = region
    path = [pts[0]]
    for i in range(len(pts)):
        j = (i+1)%len(pts)
        (center, radius, direction) = arcs[i]
        fromangle = vector_angle_in_degrees(pts[i] - center)
        toangle = vector_angle_in_degrees(pts[j] - center)
        if direction:
            vertices = Path.arc(fromangle, toangle).vertices
        else:
            vertices = Path.arc(toangle, fromangle).vertices
            vertices = vertices[np.arange(len(vertices)-1, -1, -1)]
        vertices = vertices * radius + center
        path = path + list(vertices[1:])
    codes = [1] + [4]*(len(path)-1)
    return PathPatch(Path(path, codes))

def compute_venn3_colors(set_colors):
    '''
    Given three base colors, computes combinations of colors corresponding to all regions of the venn diagram.
    returns a list of 8 elements, providing colors for regions (000, 100, 010, 110, 001, 101, 011, 111).
    '''
    ccv = ColorConverter()
    base_colors = [np.array(ccv.to_rgb(c)) for c in set_colors]
    return ((1.0, 1.0, 1.0), base_colors[0], base_colors[1], 0.7*(base_colors[0] + base_colors[1]), base_colors[2],
            0.7*(base_colors[0] + base_colors[2]), 0.7*(base_colors[1] + base_colors[2]), 0.4*(base_colors[0] + base_colors[1] + base_colors[2]))
    
def prepare_venn3_axes(ax, centers, radii):
    '''
    Sets properties of the axis object to suit venn plotting. I.e. hides ticks, makes proper xlim/ylim.
    '''
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    min_x = min([centers[i][0] - radii[i] for i in range(3)])
    max_x = max([centers[i][0] + radii[i] for i in range(3)])
    min_y = min([centers[i][1] - radii[i] for i in range(3)])
    max_y = max([centers[i][1] + radii[i] for i in range(3)])
    ax.set_xlim([min_x - 0.1, max_x + 0.1])
    ax.set_ylim([min_y - 0.1, max_y + 0.1])
    ax.set_axis_off()

def venn3_circles(sets, normalize_to=1.0, alpha=1.0, color='black', linestyle='solid', linewidth=2.0, **kwargs):
    '''
    Plots only the three circles for the corresponding Venn diagram. Useful for debugging or enhancing the basic venn diagram.
    normalize_to is the same as in venn3()
    kwargs are passed as-is to matplotlib.patches.Circle.
    returns a list of three Circle patches.
    '''
    circ = solve_venn3_circles(sets, normalize_to)
    centers = position_venn3_circles(*circ)
    ax = gca()
    prepare_venn3_axes(ax, centers, circ[0:3])
    result = []
    for (c, r) in zip(centers, circ[0:3]):
        circle = Circle(c, r, alpha=alpha, edgecolor=color, facecolor='none', ls=linestyle, lw=linewidth, **kwargs)
        ax.add_patch(circle)
        result.append(circle)
    return result

class Venn3:
    '''
    A container for a set of patches and patch labels and set labels, which make up the rendered venn diagram.
    '''
    id2idx = {'100':0,'010':1,'110':2,'001':3,'101':4,'011':5,'111':6} 
    def __init__(self, patches, texts, labels):
        self.patches = patches
        self.texts = texts
        self.labels = labels
    def get_patch_by_id(self, id):
        '''Returns a patch by a "region id". A region id is a string like 001, 011, 010, etc.'''
        return self.patches[self.id2idx[id]]
    def get_text_by_id(self, id):
        '''Returns a text by a "region id". A region id is a string like 001, 011, 010, etc.'''
        return self.texts[self.id2idx[id]]
        
def venn3(sets, set_labels = ('A', 'B', 'C'), set_colors=('r', 'g', 'b'), alpha=0.4, normalize_to=1.0):
    '''Plots a 3-set Venn diagram.
    The sets parameter is either a dict or a list.
     - If it is a dict, it must map regions to their sizes.
       The regions are identified via three-letter binary codes ('000', '010', etc), hence a valid set could look like:
       {'001': 10, '010': 20, '110', ...}
     - If it is a list, it must have 8 elements, denoting the sizes of the regions in the following order:
       (000, 100, 010, 110, 001, 101, 011, 111). Note that the first element is not used.
    
    Set labels parameter is a list of three strings - set labels. Set it to None to disable set labels.
    The set_colors parameter should be a list of three elements, specifying the "base colors" of the three circles.
    The colors of circle intersections will be computed based on those.
    
    The normalize_to parameter specifies the total (on-screen) area of the circles to be drawn. Make it larger if your text does not fit.
    The return value is a Venn3 object, that keeps references to the Text and Patch objects used on the plot.
    
        >> from matplotlib.venn import *
        >> v = venn3(sets=(0, 1, 1, 1, 1, 1, 1, 1), set_labels = ('A', 'B', 'C'))
        >> venn3_circles(sets=(0, 1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
        >> v.get_patch_by_id('100').set_alpha(1.0)
        >> v.get_patch_by_id('100').set_color('white')
        >> v.get_text_by_id('100').set_text('Unknown')
    '''
    # Prepare parameters
    if isinstance(sets, dict):
        sets = [s[t] for t in ['000', '100', '010', '110', '001', '101', '011', '111']]
        
    # Solve Venn diagram
    ax = gca()
    circ = solve_venn3_circles(sets, normalize_to)
    radii = circ[0:3]
    centers = position_venn3_circles(*circ)
    regions = compute_venn3_regions(centers, radii)
    colors = compute_venn3_colors(set_colors)[1:]
    
    # Create and add patches and text
    prepare_venn3_axes(ax, centers, circ[0:3])
    patches = [make_venn3_region_patch(r) for r in regions[1:]]
    for (p, c) in zip(patches, colors):
        if p is not None:
            p.set_facecolor(c)
            p.set_edgecolor('none')
            p.set_alpha(alpha)
            ax.add_patch(p)
    texts = [ax.text(r[2][0], r[2][1], str(s), va='center', ha='center') for (r, s) in zip(regions[1:], sets[1:])]

    if set_labels is not None:
        label_positions = [centers[0] + np.array([-radii[0]/2, radii[0]]),
                           centers[1] + np.array([radii[1]/2, radii[1]]),
                           centers[2] + np.array([0.0, -radii[2]*1.1])]
        labels = [ax.text(pos[0], pos[1], txt, size='large') for (pos, txt) in zip(label_positions, set_labels)]
        labels[0].set_horizontalalignment('right')
        labels[1].set_horizontalalignment('left')
        labels[2].set_verticalalignment('top')
        labels[2].set_horizontalalignment('center')
    else:
        labels = None
    return Venn3(patches, texts, labels)
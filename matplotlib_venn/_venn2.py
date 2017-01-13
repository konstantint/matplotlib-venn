'''
Venn diagram plotting routines.
Two-circle venn plotter.

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''
# Make sure we don't try to do GUI stuff when running tests
import sys, os
if 'py.test' in os.path.basename(sys.argv[0]): # (XXX: Ugly hack)
    import matplotlib
    matplotlib.use('Agg')

import numpy as np
import warnings
from collections import Counter

from matplotlib.patches import Circle
from matplotlib.colors import ColorConverter
from matplotlib.pyplot import gca

from matplotlib_venn._math import *
from matplotlib_venn._common import *
from matplotlib_venn._region import VennCircleRegion


def compute_venn2_areas(diagram_areas, normalize_to=1.0):
    '''
    The list of venn areas is given as 3 values, corresponding to venn diagram areas in the following order:
     (Ab, aB, AB)  (i.e. last element corresponds to the size of intersection A&B&C).
    The return value is a list of areas (A, B, AB), such that the total area is normalized
    to normalize_to. If total area was 0, returns (1e-06, 1e-06, 0.0)

    Assumes all input values are nonnegative (to be more precise, all areas are passed through and abs() function)
    >>> compute_venn2_areas((1, 1, 0))
    (0.5, 0.5, 0.0)
    >>> compute_venn2_areas((0, 0, 0))
    (1e-06, 1e-06, 0.0)
    >>> compute_venn2_areas((1, 1, 1), normalize_to=3)
    (2.0, 2.0, 1.0)
    >>> compute_venn2_areas((1, 2, 3), normalize_to=6)
    (4.0, 5.0, 3.0)
    '''
    # Normalize input values to sum to 1
    areas = np.array(np.abs(diagram_areas), float)
    total_area = np.sum(areas)
    if np.abs(total_area) < tol:
        warnings.warn("Both circles have zero area")
        return (1e-06, 1e-06, 0.0)
    else:
        areas = areas / total_area * normalize_to
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
    (A_a, A_b, A_ab) = list(map(float, venn_areas))
    r_a, r_b = np.sqrt(A_a / np.pi), np.sqrt(A_b / np.pi)
    radii = np.array([r_a, r_b])
    if A_ab > tol:
        # Nonzero intersection
        coords = np.zeros((2, 2))
        coords[1][0] = find_distance_by_area(radii[0], radii[1], A_ab)
    else:
        # Zero intersection
        coords = np.zeros((2, 2))
        coords[1][0] = radii[0] + radii[1] + max(np.mean(radii) * 1.1, 0.2)   # The max here is needed for the case r_a = r_b = 0
    coords = normalize_by_center_of_mass(coords, radii)
    return (coords, radii)


def compute_venn2_regions(centers, radii):
    '''
    Returns a triple of VennRegion objects, describing the three regions of the diagram, corresponding to sets
    (Ab, aB, AB)

    >>> centers, radii = solve_venn2_circles((1, 1, 0.5))
    >>> regions = compute_venn2_regions(centers, radii)
    '''
    A = VennCircleRegion(centers[0], radii[0])
    B = VennCircleRegion(centers[1], radii[1])
    Ab, AB = A.subtract_and_intersect_circle(B.center, B.radius)
    aB, _ = B.subtract_and_intersect_circle(A.center, A.radius)
    return (Ab, aB, AB)


def compute_venn2_colors(set_colors):
    '''
    Given two base colors, computes combinations of colors corresponding to all regions of the venn diagram.
    returns a list of 3 elements, providing colors for regions (10, 01, 11).

    >>> compute_venn2_colors(('r', 'g'))
    (array([ 1.,  0.,  0.]), array([ 0. ,  0.5,  0. ]), array([ 0.7 ,  0.35,  0.  ]))
    '''
    ccv = ColorConverter()
    base_colors = [np.array(ccv.to_rgb(c)) for c in set_colors]
    return (base_colors[0], base_colors[1], mix_colors(base_colors[0], base_colors[1]))


def compute_venn2_subsets(a, b):
    '''
    Given two set or Counter objects, computes the sizes of (a & ~b, b & ~a, a & b).
    Returns the result as a tuple.

    >>> compute_venn2_subsets(set([1,2,3,4]), set([2,3,4,5,6]))
    (1, 2, 3)
    >>> compute_venn2_subsets(Counter([1,2,3,4]), Counter([2,3,4,5,6]))
    (1, 2, 3)
    >>> compute_venn2_subsets(Counter([]), Counter([]))
    (0, 0, 0)
    >>> compute_venn2_subsets(set([]), set([]))
    (0, 0, 0)
    >>> compute_venn2_subsets(set([1]), set([]))
    (1, 0, 0)
    >>> compute_venn2_subsets(set([1]), set([1]))
    (0, 0, 1)
    >>> compute_venn2_subsets(Counter([1]), Counter([1]))
    (0, 0, 1)
    >>> compute_venn2_subsets(set([1,2]), set([1]))
    (1, 0, 1)
    >>> compute_venn2_subsets(Counter([1,1,2,2,2]), Counter([1,2,3,3]))
    (3, 2, 2)
    >>> compute_venn2_subsets(Counter([1,1,2]), Counter([1,2,2]))
    (1, 1, 2)
    >>> compute_venn2_subsets(Counter([1,1]), set([]))
    Traceback (most recent call last):
    ...
    ValueError: Both arguments must be of the same type
    '''
    if not (type(a) == type(b)):
        raise ValueError("Both arguments must be of the same type")
    set_size = len if type(a) != Counter else lambda x: sum(x.values())   # We cannot use len to compute the cardinality of a Counter
    return (set_size(a - b), set_size(b - a), set_size(a & b))


def venn2_circles(subsets, normalize_to=1.0, alpha=1.0, color='black', linestyle='solid', linewidth=2.0, ax=None, **kwargs):
    '''
    Plots only the two circles for the corresponding Venn diagram.
    Useful for debugging or enhancing the basic venn diagram.
    parameters ``subsets``, ``normalize_to`` and ``ax`` are the same as in venn2()
    ``kwargs`` are passed as-is to matplotlib.patches.Circle.
    returns a list of three Circle patches.

    >>> c = venn2_circles((1, 2, 3))
    >>> c = venn2_circles({'10': 1, '01': 2, '11': 3}) # Same effect
    >>> c = venn2_circles([set([1,2,3,4]), set([2,3,4,5,6])]) # Also same effect
    '''
    if isinstance(subsets, dict):
        subsets = [subsets.get(t, 0) for t in ['10', '01', '11']]
    elif len(subsets) == 2:
        subsets = compute_venn2_subsets(*subsets)
    areas = compute_venn2_areas(subsets, normalize_to)
    centers, radii = solve_venn2_circles(areas)

    if ax is None:
        ax = gca()
    prepare_venn_axes(ax, centers, radii)
    result = []
    for (c, r) in zip(centers, radii):
        circle = Circle(c, r, alpha=alpha, edgecolor=color, facecolor='none', linestyle=linestyle, linewidth=linewidth, **kwargs)
        ax.add_patch(circle)
        result.append(circle)
    return result


def venn2(subsets, set_labels=('A', 'B'), set_colors=('r', 'g'), alpha=0.4, normalize_to=1.0, ax=None, subset_label_formatter=None):
    '''Plots a 2-set area-weighted Venn diagram.
    The subsets parameter can be one of the following:
     - A list (or a tuple) containing two set objects.
     - A dict, providing sizes of three diagram regions.
       The regions are identified via two-letter binary codes ('10', '01', and '11'), hence a valid set could look like:
       {'10': 10, '01': 20, '11': 40}. Unmentioned codes are considered to map to 0.
     - A list (or a tuple) with three numbers, denoting the sizes of the regions in the following order:
       (10, 01, 11)

    ``set_labels`` parameter is a list of two strings - set labels. Set it to None to disable set labels.
    The ``set_colors`` parameter should be a list of two elements, specifying the "base colors" of the two circles.
    The color of circle intersection will be computed based on those.

    The ``normalize_to`` parameter specifies the total (on-axes) area of the circles to be drawn. Sometimes tuning it (together
    with the overall fiture size) may be useful to fit the text labels better.
    The return value is a ``VennDiagram`` object, that keeps references to the ``Text`` and ``Patch`` objects used on the plot
    and lets you know the centers and radii of the circles, if you need it.

    The ``ax`` parameter specifies the axes on which the plot will be drawn (None means current axes).

    The ``subset_label_formatter`` parameter is a function that can be passed to format the labels
    that describe the size of each subset.

    >>> from matplotlib_venn import *
    >>> v = venn2(subsets={'10': 1, '01': 1, '11': 1}, set_labels = ('A', 'B'))
    >>> c = venn2_circles(subsets=(1, 1, 1), linestyle='dashed')
    >>> v.get_patch_by_id('10').set_alpha(1.0)
    >>> v.get_patch_by_id('10').set_color('white')
    >>> v.get_label_by_id('10').set_text('Unknown')
    >>> v.get_label_by_id('A').set_text('Set A')

    You can provide sets themselves rather than subset sizes:
    >>> v = venn2(subsets=[set([1,2]), set([2,3,4,5])], set_labels = ('A', 'B'))
    >>> c = venn2_circles(subsets=[set([1,2]), set([2,3,4,5])], linestyle='dashed')
    >>> print("%0.2f" % (v.get_circle_radius(1)/v.get_circle_radius(0)))
    1.41
    '''
    if isinstance(subsets, dict):
        subsets = [subsets.get(t, 0) for t in ['10', '01', '11']]
    elif len(subsets) == 2:
        subsets = compute_venn2_subsets(*subsets)

    if subset_label_formatter is None:
        subset_label_formatter = str

    areas = compute_venn2_areas(subsets, normalize_to)
    centers, radii = solve_venn2_circles(areas)
    regions = compute_venn2_regions(centers, radii)
    colors = compute_venn2_colors(set_colors)

    if ax is None:
        ax = gca()
    prepare_venn_axes(ax, centers, radii)

    # Create and add patches and subset labels
    patches = [r.make_patch() for r in regions]
    for (p, c) in zip(patches, colors):
        if p is not None:
            p.set_facecolor(c)
            p.set_edgecolor('none')
            p.set_alpha(alpha)
            ax.add_patch(p)
    label_positions = [r.label_position() for r in regions]
    subset_labels = [ax.text(lbl[0], lbl[1], subset_label_formatter(s), va='center', ha='center') if lbl is not None else None for (lbl, s) in zip(label_positions, subsets)]

    # Position set labels
    if set_labels is not None:
        padding = np.mean([r * 0.1 for r in radii])
        label_positions = [centers[0] + np.array([0.0, - radii[0] - padding]),
                           centers[1] + np.array([0.0, - radii[1] - padding])]
        labels = [ax.text(pos[0], pos[1], txt, size='large', ha='right', va='top') for (pos, txt) in zip(label_positions, set_labels)]
        labels[1].set_ha('left')
    else:
        labels = None
    return VennDiagram(patches, subset_labels, labels, centers, radii)

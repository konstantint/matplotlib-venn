'''
Venn diagram plotting routines.
Three-circle venn plotter.

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''
import numpy as np
import warnings
from collections import Counter

from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.colors import ColorConverter
from matplotlib.pyplot import gca

from matplotlib_venn._math import *
from matplotlib_venn._common import *
from matplotlib_venn._region import VennCircleRegion, VennEmptyRegion


def compute_venn3_areas(diagram_areas, normalize_to=1.0, _minimal_area=1e-6):
    '''
    The list of venn areas is given as 7 values, corresponding to venn diagram areas in the following order:
     (Abc, aBc, ABc, abC, AbC, aBC, ABC)
    (i.e. last element corresponds to the size of intersection A&B&C).
    The return value is a list of areas (A_a, A_b, A_c, A_ab, A_bc, A_ac, A_abc),
    such that the total area of all circles is normalized to normalize_to.
    If the area of any circle is smaller than _minimal_area, makes it equal to _minimal_area.

    Assumes all input values are nonnegative (to be more precise, all areas are passed through and abs() function)
    >>> compute_venn3_areas((1, 1, 0, 1, 0, 0, 0))
    (0.33..., 0.33..., 0.33..., 0.0, 0.0, 0.0, 0.0)
    >>> compute_venn3_areas((0, 0, 0, 0, 0, 0, 0))
    (1e-06, 1e-06, 1e-06, 0.0, 0.0, 0.0, 0.0)
    >>> compute_venn3_areas((1, 1, 1, 1, 1, 1, 1), normalize_to=7)
    (4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0)
    >>> compute_venn3_areas((1, 2, 3, 4, 5, 6, 7), normalize_to=56/2)
    (16.0, 18.0, 22.0, 10.0, 13.0, 12.0, 7.0)
    '''
    # Normalize input values to sum to 1
    areas = np.array(np.abs(diagram_areas), float)
    total_area = np.sum(areas)
    if np.abs(total_area) < _minimal_area:
        warnings.warn("All circles have zero area")
        return (1e-06, 1e-06, 1e-06, 0.0, 0.0, 0.0, 0.0)
    else:
        areas = areas / total_area * normalize_to
        A_a = areas[0] + areas[2] + areas[4] + areas[6]
        if A_a < _minimal_area:
            warnings.warn("Circle A has zero area")
            A_a = _minimal_area
        A_b = areas[1] + areas[2] + areas[5] + areas[6]
        if A_b < _minimal_area:
            warnings.warn("Circle B has zero area")
            A_b = _minimal_area
        A_c = areas[3] + areas[4] + areas[5] + areas[6]
        if A_c < _minimal_area:
            warnings.warn("Circle C has zero area")
            A_c = _minimal_area

        # Areas of the three intersections (ab, ac, bc)
        A_ab, A_ac, A_bc = areas[2] + areas[6], areas[4] + areas[6], areas[5] + areas[6]

        return (A_a, A_b, A_c, A_ab, A_bc, A_ac, areas[6])


def solve_venn3_circles(venn_areas):
    '''
    Given the list of "venn areas" (as output from compute_venn3_areas, i.e. [A, B, C, AB, BC, AC, ABC]),
    finds the positions and radii of the three circles.
    The return value is a tuple (coords, radii), where coords is a 3x2 array of coordinates and
    radii is a 3x1 array of circle radii.

    Assumes the input values to be nonnegative and not all zero.
    In particular, the first three values must all be positive.

    The overall match is only approximate (to be precise, what is matched are the areas of the circles and the
    three pairwise intersections).

    >>> c, r = solve_venn3_circles((1, 1, 1, 0, 0, 0, 0))
    >>> np.round(r, 3).tolist()
    [0.564, 0.564, 0.564]
    >>> c, r = solve_venn3_circles(compute_venn3_areas((1, 2, 40, 30, 4, 40, 4)))
    >>> np.round(r, 3).tolist()
    [0.359, 0.476, 0.453]
    '''
    (A_a, A_b, A_c, A_ab, A_bc, A_ac, A_abc) = list(map(float, venn_areas))
    r_a, r_b, r_c = np.sqrt(A_a / np.pi), np.sqrt(A_b / np.pi), np.sqrt(A_c / np.pi)
    intersection_areas = [A_ab, A_bc, A_ac]
    radii = np.array([r_a, r_b, r_c])

    # Hypothetical distances between circle centers that assure
    # that their pairwise intersection areas match the requirements.
    dists = [find_distance_by_area(radii[i], radii[j], intersection_areas[i]) for (i, j) in [(0, 1), (1, 2), (2, 0)]]

    # How many intersections have nonzero area?
    num_nonzero = sum(np.array([A_ab, A_bc, A_ac]) > tol)

    # Handle four separate cases:
    #    1. All pairwise areas nonzero
    #    2. Two pairwise areas nonzero
    #    3. One pairwise area nonzero
    #    4. All pairwise areas zero.

    if num_nonzero == 3:
        # The "generic" case, simply use dists to position circles at the vertices of a triangle.
        # Before we need to ensure that resulting circles can be at all positioned on a triangle,
        # use an ad-hoc fix.
        for i in range(3):
            i, j, k = (i, (i + 1) % 3, (i + 2) % 3)
            if dists[i] > dists[j] + dists[k]:
                a, b = (j, k) if dists[j] < dists[k] else (k, j)
                dists[i] = dists[b] + dists[a]*0.8
                warnings.warn("Bad circle positioning")
        coords = position_venn3_circles_generic(radii, dists)
    elif num_nonzero == 2:
        # One pair of circles is not intersecting.
        # In this case we can position all three circles in a line
        # The two circles that have no intersection will be on either sides.
        for i in range(3):
            if intersection_areas[i] < tol:
                (left, right, middle) = (i, (i + 1) % 3, (i + 2) % 3)
                coords = np.zeros((3, 2))
                coords[middle][0] = dists[middle]
                coords[right][0] = dists[middle] + dists[right]
                # We want to avoid the situation where left & right still intersect
                if coords[left][0] + radii[left] > coords[right][0] - radii[right]:
                    mid = (coords[left][0] + radii[left] + coords[right][0] - radii[right]) / 2.0
                    coords[left][0] = mid - radii[left] - 1e-5
                    coords[right][0] = mid + radii[right] + 1e-5
                break
    elif num_nonzero == 1:
        # Only one pair of circles is intersecting, and one circle is independent.
        # Position all on a line first two intersecting, then the free one.
        for i in range(3):
            if intersection_areas[i] > tol:
                (left, right, side) = (i, (i + 1) % 3, (i + 2) % 3)
                coords = np.zeros((3, 2))
                coords[right][0] = dists[left]
                coords[side][0] = dists[left] + radii[right] + radii[side] * 1.1  # Pad by 10%
                break
    else:
        # All circles are non-touching. Put them all in a sequence
        coords = np.zeros((3, 2))
        coords[1][0] = radii[0] + radii[1] * 1.1
        coords[2][0] = radii[0] + radii[1] * 1.1 + radii[1] + radii[2] * 1.1

    coords = normalize_by_center_of_mass(coords, radii)
    return (coords, radii)


def position_venn3_circles_generic(radii, dists):
    '''
    Given radii = (r_a, r_b, r_c) and distances between the circles = (d_ab, d_bc, d_ac),
    finds the coordinates of the centers for the three circles so that they form a proper triangle.
    The current positioning method puts the center of A and B on a horizontal line y==0,
    and C just below.

    Returns a 3x2 array with circle center coordinates in rows.

    >>> position_venn3_circles_generic((1, 1, 1), (0, 0, 0))
    array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0., -0.]])
    >>> position_venn3_circles_generic((1, 1, 1), (2, 2, 2))
    array([[ 0.        ,  0.        ],
           [ 2.        ,  0.        ],
           [ 1.        , -1.73205081]])
    '''
    (d_ab, d_bc, d_ac) = dists
    (r_a, r_b, r_c) = radii
    coords = np.array([[0, 0], [d_ab, 0], [0, 0]], float)
    C_x = (d_ac**2 - d_bc**2 + d_ab**2) / 2.0 / d_ab if np.abs(d_ab) > tol else 0.0
    C_y = -np.sqrt(d_ac**2 - C_x**2)
    coords[2, :] = C_x, C_y
    return coords


def compute_venn3_regions(centers, radii):
    '''
    Given the 3x2 matrix with circle center coordinates, and a 3-element list (or array) with circle radii [as returned from solve_venn3_circles],
    returns the 7 regions, comprising the venn diagram, as VennRegion objects.

    Regions are returned in order (Abc, aBc, ABc, abC, AbC, aBC, ABC)

    >>> centers, radii = solve_venn3_circles((1, 1, 1, 1, 1, 1, 1))
    >>> regions = compute_venn3_regions(centers, radii)
    '''
    A = VennCircleRegion(centers[0], radii[0])
    B = VennCircleRegion(centers[1], radii[1])
    C = VennCircleRegion(centers[2], radii[2])
    Ab, AB = A.subtract_and_intersect_circle(B.center, B.radius)
    ABc, ABC = AB.subtract_and_intersect_circle(C.center, C.radius)
    Abc, AbC = Ab.subtract_and_intersect_circle(C.center, C.radius)
    aB, _ = B.subtract_and_intersect_circle(A.center, A.radius)
    aBc, aBC = aB.subtract_and_intersect_circle(C.center, C.radius)
    aC, _ = C.subtract_and_intersect_circle(A.center, A.radius)
    abC, _ = aC.subtract_and_intersect_circle(B.center, B.radius)
    return [Abc, aBc, ABc, abC, AbC, aBC, ABC]


def compute_venn3_colors(set_colors):
    '''
    Given three base colors, computes combinations of colors corresponding to all regions of the venn diagram.
    returns a list of 7 elements, providing colors for regions (100, 010, 110, 001, 101, 011, 111).

    >>> str(compute_venn3_colors(['r', 'g', 'b'])).replace(' ', '')
    '(array([1.,0.,0.]),...,array([0.4,0.2,0.4]))'
    '''
    ccv = ColorConverter()
    base_colors = [np.array(ccv.to_rgb(c)) for c in set_colors]
    return (base_colors[0], base_colors[1], mix_colors(base_colors[0], base_colors[1]), base_colors[2],
            mix_colors(base_colors[0], base_colors[2]), mix_colors(base_colors[1], base_colors[2]), mix_colors(base_colors[0], base_colors[1], base_colors[2]))


def compute_venn3_subsets(a, b, c):
    '''
    Given three set or Counter objects, computes the sizes of (a & ~b & ~c, ~a & b & ~c, a & b & ~c, ....),
    as needed by the subsets parameter of venn3 and venn3_circles.
    Returns the result as a tuple.

    >>> compute_venn3_subsets(set([1,2,3]), set([2,3,4]), set([3,4,5,6]))
    (1, 0, 1, 2, 0, 1, 1)
    >>> compute_venn3_subsets(Counter([1,2,3]), Counter([2,3,4]), Counter([3,4,5,6]))
    (1, 0, 1, 2, 0, 1, 1)
    >>> compute_venn3_subsets(Counter([1,1,1]), Counter([1,1,1]), Counter([1,1,1,1]))
    (0, 0, 0, 1, 0, 0, 3)
    >>> compute_venn3_subsets(Counter([1,1,2,2,3,3]), Counter([2,2,3,3,4,4]), Counter([3,3,4,4,5,5,6,6]))
    (2, 0, 2, 4, 0, 2, 2)
    >>> compute_venn3_subsets(Counter([1,2,3]), Counter([2,2,3,3,4,4]), Counter([3,3,4,4,4,5,5,6]))
    (1, 1, 1, 4, 0, 3, 1)
    >>> compute_venn3_subsets(set([]), set([]), set([]))
    (0, 0, 0, 0, 0, 0, 0)
    >>> compute_venn3_subsets(set([1]), set([]), set([]))
    (1, 0, 0, 0, 0, 0, 0)
    >>> compute_venn3_subsets(set([]), set([1]), set([]))
    (0, 1, 0, 0, 0, 0, 0)
    >>> compute_venn3_subsets(set([]), set([]), set([1]))
    (0, 0, 0, 1, 0, 0, 0)
    >>> compute_venn3_subsets(Counter([]), Counter([]), Counter([1]))
    (0, 0, 0, 1, 0, 0, 0)
    >>> compute_venn3_subsets(set([1]), set([1]), set([1]))
    (0, 0, 0, 0, 0, 0, 1)
    >>> compute_venn3_subsets(set([1,3,5,7]), set([2,3,6,7]), set([4,5,6,7]))
    (1, 1, 1, 1, 1, 1, 1)
    >>> compute_venn3_subsets(Counter([1,3,5,7]), Counter([2,3,6,7]), Counter([4,5,6,7]))
    (1, 1, 1, 1, 1, 1, 1)
    >>> compute_venn3_subsets(Counter([1,3,5,7]), set([2,3,6,7]), set([4,5,6,7]))
    Traceback (most recent call last):
    ...
    ValueError: All arguments must be of the same type
    '''
    if not (type(a) == type(b) == type(c)):
        raise ValueError("All arguments must be of the same type")
    set_size = len if type(a) != Counter else lambda x: sum(x.values())   # We cannot use len to compute the cardinality of a Counter
    return (set_size(a - (b | c)),  # TODO: This is certainly not the most efficient way to compute.
        set_size(b - (a | c)),
        set_size((a & b) - c),
        set_size(c - (a | b)),
        set_size((a & c) - b),
        set_size((b & c) - a),
        set_size(a & b & c))


def venn3_circles(subsets, normalize_to=1.0, alpha=1.0, color='black', linestyle='solid', linewidth=2.0, ax=None, **kwargs):
    '''
    Plots only the three circles for the corresponding Venn diagram.
    Useful for debugging or enhancing the basic venn diagram.
    parameters ``subsets``, ``normalize_to`` and ``ax`` are the same as in venn3()
    kwargs are passed as-is to matplotlib.patches.Circle.
    returns a list of three Circle patches.

        >>> plot = venn3_circles({'001': 10, '100': 20, '010': 21, '110': 13, '011': 14})
        >>> plot = venn3_circles([set(['A','B','C']), set(['A','D','E','F']), set(['D','G','H'])])
    '''
    # Prepare parameters
    if isinstance(subsets, dict):
        subsets = [subsets.get(t, 0) for t in ['100', '010', '110', '001', '101', '011', '111']]
    elif len(subsets) == 3:
        subsets = compute_venn3_subsets(*subsets)

    areas = compute_venn3_areas(subsets, normalize_to)
    centers, radii = solve_venn3_circles(areas)

    if ax is None:
        ax = gca()
    prepare_venn_axes(ax, centers, radii)
    result = []
    for (c, r) in zip(centers, radii):
        circle = Circle(c, r, alpha=alpha, edgecolor=color, facecolor='none', linestyle=linestyle, linewidth=linewidth, **kwargs)
        ax.add_patch(circle)
        result.append(circle)
    return result


def venn3(subsets, set_labels=('A', 'B', 'C'), set_colors=('r', 'g', 'b'), alpha=0.4, normalize_to=1.0, ax=None, subset_label_formatter=None):
    '''Plots a 3-set area-weighted Venn diagram.
    The subsets parameter can be one of the following:
     - A list (or a tuple), containing three set objects.
     - A dict, providing sizes of seven diagram regions.
       The regions are identified via three-letter binary codes ('100', '010', etc), hence a valid set could look like:
       {'001': 10, '010': 20, '110':30, ...}. Unmentioned codes are considered to map to 0.
     - A list (or a tuple) with 7 numbers, denoting the sizes of the regions in the following order:
       (100, 010, 110, 001, 101, 011, 111).

    ``set_labels`` parameter is a list of three strings - set labels. Set it to None to disable set labels.
    The ``set_colors`` parameter should be a list of three elements, specifying the "base colors" of the three circles.
    The colors of circle intersections will be computed based on those.

    The ``normalize_to`` parameter specifies the total (on-axes) area of the circles to be drawn. Sometimes tuning it (together
    with the overall fiture size) may be useful to fit the text labels better.
    The return value is a ``VennDiagram`` object, that keeps references to the ``Text`` and ``Patch`` objects used on the plot
    and lets you know the centers and radii of the circles, if you need it.

    The ``ax`` parameter specifies the axes on which the plot will be drawn (None means current axes).

    The ``subset_label_formatter`` parameter is a function that can be passed to format the labels
    that describe the size of each subset.

    Note: if some of the circles happen to have zero area, you will probably not get a nice picture.

    >>> import matplotlib # (The first two lines prevent the doctest from falling when TCL not installed. Not really necessary in most cases)
    >>> matplotlib.use('Agg')
    >>> from matplotlib_venn import *
    >>> v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels = ('A', 'B', 'C'))
    >>> c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
    >>> v.get_patch_by_id('100').set_alpha(1.0)
    >>> v.get_patch_by_id('100').set_color('white')
    >>> v.get_label_by_id('100').set_text('Unknown')
    >>> v.get_label_by_id('C').set_text('Set C')

    You can provide sets themselves rather than subset sizes:
    >>> v = venn3(subsets=[set([1,2]), set([2,3,4,5]), set([4,5,6,7,8,9,10,11])])
    >>> print("%0.2f %0.2f %0.2f" % (v.get_circle_radius(0), v.get_circle_radius(1)/v.get_circle_radius(0), v.get_circle_radius(2)/v.get_circle_radius(0)))
    0.24 1.41 2.00
    >>> c = venn3_circles(subsets=[set([1,2]), set([2,3,4,5]), set([4,5,6,7,8,9,10,11])])
    '''
    # Prepare parameters
    if isinstance(subsets, dict):
        subsets = [subsets.get(t, 0) for t in ['100', '010', '110', '001', '101', '011', '111']]
    elif len(subsets) == 3:
        subsets = compute_venn3_subsets(*subsets)

    if subset_label_formatter is None:
        subset_label_formatter = str

    areas = compute_venn3_areas(subsets, normalize_to)
    centers, radii = solve_venn3_circles(areas)
    regions = compute_venn3_regions(centers, radii)
    colors = compute_venn3_colors(set_colors)

    # Remove regions that are too small from the diagram
    MIN_REGION_SIZE = 1e-4
    for i in range(len(regions)):
        if regions[i].size() < MIN_REGION_SIZE and subsets[i] == 0:
            regions[i] = VennEmptyRegion()

    # There is a rare case (Issue #12) when the middle region is visually empty
    # (the positioning of the circles does not let them intersect), yet the corresponding value is not 0.
    # we address it separately here by positioning the label of that empty region in a custom way
    if isinstance(regions[6], VennEmptyRegion) and subsets[6] > 0:
        intersections = [circle_circle_intersection(centers[i], radii[i], centers[j], radii[j]) for (i, j) in [(0, 1), (1, 2), (2, 0)]]
        middle_pos = np.mean([i[0] for i in intersections], 0)
        regions[6] = VennEmptyRegion(middle_pos)

    if ax is None:
        ax = gca()
    prepare_venn_axes(ax, centers, radii)

    # Create and add patches and text
    patches = [r.make_patch() for r in regions]
    for (p, c) in zip(patches, colors):
        if p is not None:
            p.set_facecolor(c)
            p.set_edgecolor('none')
            p.set_alpha(alpha)
            ax.add_patch(p)
    label_positions = [r.label_position() for r in regions]
    subset_labels = [ax.text(lbl[0], lbl[1], subset_label_formatter(s), va='center', ha='center') if lbl is not None else None for (lbl, s) in zip(label_positions, subsets)]

    # Position labels
    if set_labels is not None:
        # There are two situations, when set C is not on the same line with sets A and B, and when the three are on the same line.
        if abs(centers[2][1] - centers[0][1]) > tol:
            # Three circles NOT on the same line
            label_positions = [centers[0] + np.array([-radii[0] / 2, radii[0]]),
                               centers[1] + np.array([radii[1] / 2, radii[1]]),
                               centers[2] + np.array([0.0, -radii[2] * 1.1])]
            labels = [ax.text(pos[0], pos[1], txt, size='large') for (pos, txt) in zip(label_positions, set_labels)]
            labels[0].set_horizontalalignment('right')
            labels[1].set_horizontalalignment('left')
            labels[2].set_verticalalignment('top')
            labels[2].set_horizontalalignment('center')
        else:
            padding = np.mean([r * 0.1 for r in radii])
            # Three circles on the same line
            label_positions = [centers[0] + np.array([0.0, - radii[0] - padding]),
                               centers[1] + np.array([0.0, - radii[1] - padding]),
                               centers[2] + np.array([0.0, - radii[2] - padding])]
            labels = [ax.text(pos[0], pos[1], txt, size='large', ha='center', va='top') for (pos, txt) in zip(label_positions, set_labels)]
    else:
        labels = None
    return VennDiagram(patches, subset_labels, labels, centers, radii)

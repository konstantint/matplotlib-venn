"""
The pairwise intersection-based layout algorithm implementation.
This is the default, original layout method.

Makes sure the full circle areas and the areas of their pairwise intersections exactly match the subset areas.
The area of the triple intersection is not necessarily correct.

For situations where the triple intersection is too small in comparison to other areas it often results in bad layout.

Copyright 2012-2024, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

from typing import Optional, Tuple
import warnings
import numpy as np

from matplotlib_venn._math import (
    NUMERIC_TOLERANCE,
    Point2D,
    find_distance_by_area,
    normalize_by_center_of_mass,
)
from matplotlib_venn.layout.api import (
    LabelLayout,
    VennLayout,
    VennLayoutAlgorithm,
    SubsetSizes,
)

# The format is the same but the semantics is different.
VennAreas = SubsetSizes


class LayoutAlgorithm(VennLayoutAlgorithm):
    def __init__(
        self,
        normalize_to: float = 1.0,
        fixed_subset_sizes: Optional[SubsetSizes] = None,
    ):
        """Initialize the layout algorithm.

        Args:
            normalize_to: Specifies the total (on-axes) area of the circles to be drawn. Sometimes tuning it (together
                          with the overall figure size) can be useful to fit the text labels better.
            fixed_subset_sizes: If specified, the layout will always use these subset sizes, ignoring anything provided
                          to the actual __call__. E.g. passing (1,1,1,1,1,1,1) here will result in a non-area-weighted layout algorithm.
        """
        self._normalize_to = normalize_to
        self._fixed_subset_sizes = fixed_subset_sizes

    def __call__(
        self,
        subsets: SubsetSizes,
        set_labels: Optional[
            Tuple[str, str, str]
        ] = None,  # Not used in the layout algorithm.
    ) -> VennLayout:
        if self._fixed_subset_sizes is not None:
            subsets = self._fixed_subset_sizes
        areas = _compute_areas(subsets, self._normalize_to)
        return _compute_layout(areas)


def _compute_areas(
    subset_sizes: SubsetSizes, normalize_to: float = 1.0, _minimal_area: float = 1e-6
) -> VennAreas:
    """
    Compute areas of circles and their pairwise and triple intersections.
    Assumes all input values are nonnegative (to be more precise, all areas are passed through the abs() function)

    Args:
        subset_sizes: The relative sizes of the 7 diagram region in the following order:
            (Abc, aBc, ABc, abC, AbC, aBC, ABC)
            (i.e. last element corresponds to the size of intersection A&B&C).
        normalize_to: Normalize the values so that the total area sums to this value.
        _minimal_area: If the area of any circle is smaller than _minimal_area, makes it equal to _minimal_area.
    Returns:
        A list of areas (A_a, A_b, A_c, A_ab, A_bc, A_ac, A_abc),
        such that the total area of all circles is normalized to normalize_to (except corrections for _minimal_area)

    >>> _compute_areas((1, 1, 0, 1, 0, 0, 0))
    (0.33..., 0.33..., 0.33..., 0.0, 0.0, 0.0, 0.0)
    >>> _compute_areas((0, 0, 0, 0, 0, 0, 0))
    (1e-06, 1e-06, 1e-06, 0.0, 0.0, 0.0, 0.0)
    >>> _compute_areas((1, 1, 1, 1, 1, 1, 1), normalize_to=7)
    (4.0, 4.0, 4.0, 2.0, 2.0, 2.0, 1.0)
    >>> _compute_areas((1, 2, 3, 4, 5, 6, 7), normalize_to=56/2)
    (16.0, 18.0, 22.0, 10.0, 13.0, 12.0, 7.0)
    """
    # Normalize input values to sum to 1
    areas = np.array(np.abs(subset_sizes), float)
    total_area = np.sum(areas)
    if abs(total_area) < _minimal_area:
        warnings.warn("All circles have zero area.")
        return (1e-06, 1e-06, 1e-06, 0.0, 0.0, 0.0, 0.0)
    else:
        areas = areas / total_area * normalize_to
        A_a = areas[0] + areas[2] + areas[4] + areas[6]
        if A_a < _minimal_area:
            warnings.warn("Circle A has zero area.")
            A_a = _minimal_area
        A_b = areas[1] + areas[2] + areas[5] + areas[6]
        if A_b < _minimal_area:
            warnings.warn("Circle B has zero area.")
            A_b = _minimal_area
        A_c = areas[3] + areas[4] + areas[5] + areas[6]
        if A_c < _minimal_area:
            warnings.warn("Circle C has zero area.")
            A_c = _minimal_area

        # Areas of the three intersections (ab, ac, bc)
        A_ab, A_ac, A_bc = areas[2] + areas[6], areas[4] + areas[6], areas[5] + areas[6]

        return tuple(map(float, (A_a, A_b, A_c, A_ab, A_bc, A_ac, areas[6])))


def _compute_layout(venn_areas: VennAreas) -> VennLayout:
    """
    Given the list of "venn areas" (as output from _compute_areas, i.e. (A, B, C, AB, BC, AC, ABC)),
    finds the positions and radii of the three circles.
    Assumes the input values to be nonnegative and not all zero.
    In particular, the first three values must all be positive.

    The return value is a VennLayout struct with just the coords and radii fields.

    The overall match is only approximate (to be precise, what is matched are the areas of the circles and the
    three pairwise intersections).

    >>> layout = _compute_layout((1, 1, 1, 0, 0, 0, 0))
    >>> np.round(layout.radii, 3).tolist()
    [0.564, 0.564, 0.564]
    >>> layout = _compute_layout(_compute_areas((1, 2, 40, 30, 4, 40, 4)))
    >>> np.round(layout.radii, 3).tolist()
    [0.359, 0.476, 0.453]
    """
    (A_a, A_b, A_c, A_ab, A_bc, A_ac, A_abc) = list(map(float, venn_areas))
    r_a, r_b, r_c = np.sqrt(A_a / np.pi), np.sqrt(A_b / np.pi), np.sqrt(A_c / np.pi)
    intersection_areas = [A_ab, A_bc, A_ac]
    radii = np.array([r_a, r_b, r_c])

    # Hypothetical distances between circle centers that assure
    # that their pairwise intersection areas match the requirements.
    dists = [
        find_distance_by_area(radii[i], radii[j], intersection_areas[i])
        for (i, j) in [(0, 1), (1, 2), (2, 0)]
    ]

    # How many intersections have nonzero area?
    num_nonzero = sum(np.array([A_ab, A_bc, A_ac]) > NUMERIC_TOLERANCE)

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
                dists[i] = dists[b] + dists[a] * 0.8
                warnings.warn("Bad circle positioning.")
        coords = _compute_triangle_layout_coords(radii, dists)
    elif num_nonzero == 2:
        # One pair of circles is not intersecting.
        # In this case we can position all three circles in a line
        # The two circles that have no intersection will be on either sides.
        for i in range(3):
            if intersection_areas[i] < NUMERIC_TOLERANCE:
                (left, right, middle) = (i, (i + 1) % 3, (i + 2) % 3)
                coords = np.zeros((3, 2))
                coords[middle][0] = dists[middle]
                coords[right][0] = dists[middle] + dists[right]
                # We want to avoid the situation where left & right still intersect
                if coords[left][0] + radii[left] > coords[right][0] - radii[right]:
                    mid = (
                        coords[left][0] + radii[left] + coords[right][0] - radii[right]
                    ) / 2.0
                    coords[left][0] = mid - radii[left] - 1e-5
                    coords[right][0] = mid + radii[right] + 1e-5
                break
    elif num_nonzero == 1:
        # Only one pair of circles is intersecting, and one circle is independent.
        # Position all on a line first two intersecting, then the free one.
        for i in range(3):
            if intersection_areas[i] > NUMERIC_TOLERANCE:
                (left, right, side) = (i, (i + 1) % 3, (i + 2) % 3)
                coords = np.zeros((3, 2))
                coords[right][0] = dists[left]
                coords[side][0] = (
                    dists[left] + radii[right] + radii[side] * 1.1
                )  # Pad by 10%
                break
    else:
        # All circles are non-touching. Put them all in a sequence
        coords = np.zeros((3, 2))
        coords[1][0] = radii[0] + radii[1] * 1.1
        coords[2][0] = radii[0] + radii[1] * 1.1 + radii[1] + radii[2] * 1.1

    coords = normalize_by_center_of_mass(coords, radii)
    result = VennLayout(
        centers=(
            Point2D(coords[0][0], coords[0][1]),
            Point2D(coords[1][0], coords[1][1]),
            Point2D(coords[2][0], coords[2][1]),
        ),
        radii=(radii[0], radii[1], radii[2]),
    )
    _compute_set_labels_positions(result)
    return result


def _compute_triangle_layout_coords(
    radii: Tuple[float, float, float], dists: Tuple[float, float, float]
) -> np.ndarray:
    """
    Finds three centers for circles which form a proper triangle with given side lengths.
    The method puts the center of A and B on a horizontal line y==0, and C just below.

    Args:
        radii: The radii of the three circles (r_a, r_b, r_c).
        dists: The pairwise distances between the circle centers (d_ab, d_bc, d_ac),

    Returns:
        Coordinates of the circles to be laid out.

    >>> _compute_triangle_layout_coords((1, 1, 1), (0, 0, 0))
    array([[ 0.,  0.],
           [ 0.,  0.],
           [ 0., -0.]])
    >>> _compute_triangle_layout_coords((1, 1, 1), (2, 2, 2))
    array([[ 0.        ,  0.        ],
           [ 2.        ,  0.        ],
           [ 1.        , -1.73205081]])
    """
    (d_ab, d_bc, d_ac) = dists
    (r_a, r_b, r_c) = radii
    coords = np.array([[0, 0], [d_ab, 0], [0, 0]], float)
    C_x = (
        (d_ac**2 - d_bc**2 + d_ab**2) / 2.0 / d_ab
        if np.abs(d_ab) > NUMERIC_TOLERANCE
        else 0.0
    )
    C_y = -np.sqrt(d_ac**2 - C_x**2)
    coords[2, :] = C_x, C_y
    return coords


def _compute_set_labels_positions(layout: VennLayout):
    """Updates the set_labels_positions field of the given layout object."""
    if abs(layout.centers[2].y - layout.centers[0].y) > NUMERIC_TOLERANCE:
        # Three circles NOT on the same line
        layout.set_labels_layout = (
            LabelLayout(
                position=layout.centers[0]
                + Point2D(-layout.radii[0] / 2, layout.radii[0]),
                kwargs={"ha": "right"},
            ),
            LabelLayout(
                position=layout.centers[1]
                + Point2D(layout.radii[1] / 2, layout.radii[1]),
                kwargs={"ha": "left"},
            ),
            LabelLayout(
                position=layout.centers[2] + Point2D(0.0, -layout.radii[2] * 1.1),
                kwargs={"ha": "center", "va": "top"},
            ),
        )
    else:
        # Three circles on the same line
        padding = np.mean([r * 0.1 for r in layout.radii])
        layout.set_labels_layout = (
            LabelLayout(
                position=layout.centers[0] + Point2D(0.0, -layout.radii[0] - padding),
                kwargs={"ha": "center", "va": "top"},
            ),
            LabelLayout(
                position=layout.centers[1] + Point2D(0.0, -layout.radii[1] - padding),
                kwargs={"ha": "center", "va": "top"},
            ),
            LabelLayout(
                position=layout.centers[2] + Point2D(0.0, -layout.radii[2] - padding),
                kwargs={"ha": "center", "va": "top"},
            ),
        )

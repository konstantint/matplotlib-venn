"""
The exact area-weighted layout algorithm implementation.
This is the default, original layout method.

Copyright 2012-2024, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

from typing import Optional, Sequence
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
                          to the actual __call__. E.g. passing (1,1,1) here will result in a non-area-weighted layout algorithm.
        """
        self._normalize_to = normalize_to
        self._fixed_subset_sizes = fixed_subset_sizes

    def __call__(
        self,
        subsets: SubsetSizes,
        set_labels: Optional[Sequence[str]] = None,  # Not used in the layout algorithm
    ) -> VennLayout:
        if self._fixed_subset_sizes is not None:
            subsets = self._fixed_subset_sizes
        areas = _compute_areas(subsets, self._normalize_to)
        return _compute_layout(areas)


def _compute_areas(
    subset_sizes: SubsetSizes, normalize_to: float = 1.0, _minimal_area: float = 1e-6
) -> VennAreas:
    """
    Convert the sizes of individual regions (Ab, aB, AB) into areas (A, B, AB), used to lay out the diagram,
    normalizing the areas to sum to a given number.

    If total area was 0, returns (1e-06, 1e-06, 0.0)

    Assumes all input values are nonnegative (to be more precise, all areas are passed through and abs() function)
    >>> _compute_areas((1, 1, 0))
    (0.5, 0.5, 0.0)
    >>> _compute_areas((0, 0, 0))
    (1e-06, 1e-06, 0.0)
    >>> _compute_areas((1, 1, 1), normalize_to=3)
    (2.0, 2.0, 1.0)
    >>> _compute_areas((1, 2, 3), normalize_to=6)
    (4.0, 5.0, 3.0)
    """
    # Normalize input values to sum to 1
    areas = np.array(np.abs(subset_sizes), float)
    total_area = np.sum(areas)
    if abs(total_area) < NUMERIC_TOLERANCE:
        warnings.warn("Both circles have zero area")
        return (1e-06, 1e-06, 0.0)
    else:
        areas = areas / total_area * normalize_to
        return (float(areas[0] + areas[2]), float(areas[1] + areas[2]), float(areas[2]))


def _compute_layout(venn_areas: VennAreas) -> VennLayout:
    """
    Given the list of "venn areas" (as output from compute_venn2_areas, i.e. [A, B, AB]),
    finds the positions and radii of the two circles.

    Assumes the input values to be nonnegative and not all zero.
    In particular, the first two values must be positive.

    >>> layout = _compute_layout((1, 1, 0))
    >>> np.round(layout.radii, 3).tolist()
    [0.564, 0.564]
    >>> layout = _compute_layout(_compute_areas((1, 2, 3)))
    >>> np.round(layout.radii, 3).tolist()
    [0.461, 0.515]
    """
    (A_a, A_b, A_ab) = list(map(float, venn_areas))
    r_a, r_b = np.sqrt(A_a / np.pi), np.sqrt(A_b / np.pi)
    radii = np.array([r_a, r_b])
    if A_ab > NUMERIC_TOLERANCE:
        # Nonzero intersection
        coords = np.zeros((2, 2))
        coords[1][0] = find_distance_by_area(radii[0], radii[1], A_ab)
    else:
        # Zero intersection
        coords = np.zeros((2, 2))
        coords[1][0] = (
            radii[0] + radii[1] + max(np.mean(radii) * 1.1, 0.2)
        )  # The max here is needed for the case r_a = r_b = 0
    coords = normalize_by_center_of_mass(coords, radii)
    layout = VennLayout(
        (Point2D(*coords[0]), Point2D(*coords[1])), (radii[0], radii[1])
    )
    _compute_set_labels_positions(layout)
    return layout


def _compute_set_labels_positions(layout: VennLayout):
    """Updates the set_labels_positions field of the given layout object."""
    padding = np.mean([r * 0.1 for r in layout.radii])
    layout.set_labels_layout = (
        LabelLayout(
            position=layout.centers[0] + Point2D(0.0, -layout.radii[0] - padding),
            kwargs={"ha": "right", "va": "top"},
        ),
        LabelLayout(
            position=layout.centers[1] + Point2D(0.0, -layout.radii[1] - padding),
            kwargs={"ha": "left", "va": "top"},
        ),
    )

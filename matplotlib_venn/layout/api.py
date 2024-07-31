"""
Specification of the "layout algorithm" interface.

A layout algorithm is a method for mapping from subset sizes to circle centers, radii and label locations.

Copyright 2024, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

from typing import Any, Dict, Sequence, Optional
from abc import ABC, abstractmethod
from matplotlib_venn._math import Point2D

SubsetSizes = Sequence[float]  # .. of length 3 for venn2 and length 7 for venn3

# Failures that may be reported from the layout algorithm.
class LayoutException(Exception):
    pass

class LabelLayout:
    """Text label position in the diagram.

    Given via coordinates and a set of keyword arguments (e.g. "ha" or "va").
    """

    def __init__(self, position: Point2D, kwargs: Dict[str, Any]):
        self.position = position
        self.kwargs = kwargs


class VennLayout:
    """The circle layout specification for a Venn diagram."""

    # Centers of the (2 / 3) circles (in the Axes coordinates).
    # centers: Sequence[Point2D]
    # # Radii of the circles.
    # radii: Sequence[float]
    # # Layout information of set labels. If labels are missing, then None.
    # set_labels_layout: Optional[Sequence[LabelLayout]] = None

    def __init__(
        self,
        centers: Sequence[Point2D],
        radii: Sequence[float],
        set_labels_layout: Optional[Sequence[LabelLayout]] = None,
    ):
        self.centers = centers
        self.radii = radii
        self.set_labels_layout = set_labels_layout


class VennLayoutAlgorithm(ABC):
    """Interface for a Venn layout algorithm."""

    @abstractmethod
    def __call__(
        self,
        subsets: SubsetSizes,
        set_labels: Optional[Sequence[str]] = None,
    ) -> VennLayout:
        """Lay out the Venn circles, returning the diagram layout specification as VennLayout.
        Args:
            subsets: A tuple with 3 (for venn2) or 7 (for venn3) numbers, denoting the sizes of the
            Venn diagram regions in the following order:
                for venn2: (10, 01, 11)
                for venn3: (100, 010, 110, 001, 101, 011, 111).
            set_labels: Optional tuple of set labels. If None, resulting layout provides no label information.
        """
        pass

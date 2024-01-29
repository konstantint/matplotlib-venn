"""
Venn diagram plotting routines.
Functionality, common to venn2 and venn3.

Copyright 2012-2024, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

from typing import Optional, Sequence
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.text import Text

from matplotlib_venn._math import Point2D


class VennDiagram:
    """
    A container for a set of patches and patch labels and set labels, which make up the rendered venn diagram.
    This object is returned by a venn2 or venn3 function call.
    """

    id2idx = {
        "10": 0,
        "01": 1,
        "11": 2,
        "100": 0,
        "010": 1,
        "110": 2,
        "001": 3,
        "101": 4,
        "011": 5,
        "111": 6,
        "A": 0,
        "B": 1,
        "C": 2,
    }

    def __init__(
        self,
        patches: Sequence[Patch],
        subset_labels: Sequence[Text],
        set_labels: Sequence[Text],
        centers: Sequence[Point2D],
        radii: Sequence[float],
    ):
        self.patches = patches
        self.subset_labels = subset_labels
        self.set_labels = set_labels
        self.centers = centers
        self.radii = radii

    def get_patch_by_id(self, id: str) -> Patch:
        """Returns a patch by a "region id".
        A region id is a string '10', '01' or '11' for 2-circle diagram or a
        string like '001', '010', etc, for 3-circle diagram."""
        return self.patches[self.id2idx[id]]

    def get_label_by_id(self, id: str) -> Text:
        """
        Returns a subset label by a "region id".
        A region id is a string '10', '01' or '11' for 2-circle diagram or a
        string like '001', '010', etc, for 3-circle diagram.
        Alternatively, if the string 'A', 'B'  (or 'C' for 3-circle diagram) is given, the label of the
        corresponding set is returned (or None)."""
        if len(id) == 1:
            return (
                self.set_labels[self.id2idx[id]]
                if self.set_labels is not None
                else None
            )
        else:
            return self.subset_labels[self.id2idx[id]]

    def get_circle_center(self, id: int) -> Point2D:
        """
        Returns the coordinates of the center of a circle as a numpy array (x,y)
        id must be 0, 1 or 2 (corresponding to the first, second, or third circle).
        This is a getter-only (i.e. changing this value does not affect the diagram)
        """
        return self.centers[id]

    def get_circle_radius(self, id: int) -> float:
        """
        Returns the radius of circle id (where id is 0, 1 or 2).
        This is a getter-only (i.e. changing this value does not affect the diagram)
        """
        return self.radii[id]

    def hide_zeroes(self) -> None:
        """
        Sometimes it makes sense to hide the labels for subsets whose size is zero.
        This utility method does this.
        """
        for v in self.subset_labels:
            if v is not None and v.get_text() == "0":
                v.set_visible(False)


def mix_colors(
    col1: np.ndarray, col2: np.ndarray, col3: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Mixes two or three colors to compute a "mixed" color (for purposes of computing
    colors of the intersection regions based on the colors of the sets.
    Note that we do not simply compute averages of given colors as those seem
    too dark for some default configurations. Thus, we lighten the combination up a bit.

    Inputs are (up to) three RGB triples of floats 0.0-1.0 given as numpy arrays.

    >>> mix_colors(np.array([1.0, 0., 0.]), np.array([1.0, 0., 0.])).tolist()
    [1.0, 0.0, 0.0]
    >>> np.round(mix_colors(np.array([1.0, 1., 0.]), np.array([1.0, 0.9, 0.]), np.array([1.0, 0.8, 0.1])), 3).tolist()
    [1.0, 1.0, 0.04]
    """
    if col3 is None:
        mix_color = 0.7 * (col1 + col2)
    else:
        mix_color = 0.4 * (col1 + col2 + col3)
    mix_color = np.min([mix_color, [1.0, 1.0, 1.0]], 0)
    return mix_color


def prepare_venn_axes(
    ax: Axes, centers: Sequence[Point2D], radii: Sequence[float]
) -> None:
    """
    Sets properties of the axis object to suit venn plotting. I.e. hides ticks, makes proper xlim/ylim.
    """
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    min_x = min(c.x - r for (c, r) in zip(centers, radii))
    max_x = max(c.x + r for (c, r) in zip(centers, radii))
    min_y = min(c.y - r for (c, r) in zip(centers, radii))
    max_y = max(c.y + r for (c, r) in zip(centers, radii))
    ax.set_xlim([min_x - 0.1, max_x + 0.1])
    ax.set_ylim([min_y - 0.1, max_y + 0.1])
    ax.set_axis_off()

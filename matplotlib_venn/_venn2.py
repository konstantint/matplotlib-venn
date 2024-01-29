"""
Venn diagram plotting routines.
Two-circle venn plotter.

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

# Make sure we don't try to do GUI stuff when running tests
import sys, os

if "py.test" in os.path.basename(sys.argv[0]):  # (XXX: Ugly hack)
    import matplotlib

    matplotlib.use("Agg")

from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import warnings
from collections import Counter

from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.colors import ColorConverter
from matplotlib.pyplot import gca

from matplotlib_venn._math import Point2D
from matplotlib_venn._common import VennDiagram, prepare_venn_axes, mix_colors
from matplotlib_venn._region import VennRegion, VennCircleRegion
from matplotlib_venn.layout.api import VennLayout, VennLayoutAlgorithm
from matplotlib_venn.layout.venn2 import DefaultLayoutAlgorithm

Venn2SubsetSizes = Tuple[float, float, float]


def venn2_circles(
    subsets: Union[Tuple[set, set], Dict[str, float], Venn2SubsetSizes],
    normalize_to: Optional[float] = None,
    alpha: float = 1.0,
    color: Any = "black",
    linestyle: str = "solid",
    linewidth: float = 2.0,
    ax: Axes = None,
    layout_algorithm: Optional[VennLayoutAlgorithm] = None,
    **kwargs
):
    """
    Plots only the two circles for the corresponding Venn diagram.
    Useful for debugging or enhancing the basic venn diagram.

    Args:
        subsets: Same as in `venn2`.
        normalize_to: Same as in `venn2`.
        alpha: The alpha parameter of the circle patches.
        color: The edgecolor of the circle patches (as understood by matplotlib).
        linestyle: The linestyle of the circle patches.
        linewidth: The line width of the circle patches.
        ax: Axis to draw upon, defaults to gca().
        layout_algorithm: The layout algorithm to be used. Defaults to matplotlib_venn.layout.venn2.DefaultLayoutAlgorithm(normalize_to).
        **kwargs: passed as-is to matplotlib.patches.Circle.

    Returns:
        a list of two Circle patches plotted.

    >>> c = venn2_circles((1, 2, 3))
    >>> c = venn2_circles({'10': 1, '01': 2, '11': 3}) # Same effect
    >>> c = venn2_circles([set([1,2,3,4]), set([2,3,4,5,6])]) # Also same effect
    """
    if isinstance(subsets, dict):
        subsets = [subsets.get(t, 0) for t in ["10", "01", "11"]]
    elif len(subsets) == 2:
        subsets = _compute_subset_sizes(*subsets)

    if normalize_to is not None:
        if layout_algorithm is None:
            warnings.warn(
                "normalize_to is deprecated. Please use layout_algorithm=matplotlib_venn.layout.venn2.DefaultLayoutAlgorithm(normalize_to) instead."
            )
        else:
            raise ValueError(
                "normalize_to is deprecated and may not be specified together with a custom layout algorithm."
            )
    if layout_algorithm is None:
        layout_algorithm = DefaultLayoutAlgorithm(normalize_to=normalize_to or 1.0)

    layout = layout_algorithm(subsets)
    if ax is None:
        ax = gca()
    prepare_venn_axes(ax, layout.centers, layout.radii)
    result = []
    for c, r in zip(layout.centers, layout.radii):
        circle = Circle(
            c.asarray(),
            r,
            alpha=alpha,
            edgecolor=color,
            facecolor="none",
            linestyle=linestyle,
            linewidth=linewidth,
            **kwargs
        )
        ax.add_patch(circle)
        result.append(circle)
    return tuple(result)


def venn2(
    subsets: Union[Tuple[set, set], Dict[str, float], Venn2SubsetSizes],
    set_labels: Optional[Tuple[str, str]] = ("A", "B"),
    set_colors: Tuple[Any, Any] = ("r", "g"),
    alpha: float = 0.4,
    normalize_to: Optional[float] = None,
    ax: Optional[Axes] = None,
    subset_label_formatter: Optional[Callable[[float], str]] = None,
    layout_algorithm: Optional[VennLayoutAlgorithm] = None,
):
    """Plots a 2-set area-weighted Venn diagram.

    Args:
        subsets: one of the following:
            - A tuple of two set objects.
            - A dict, providing relative sizes of the three diagram regions.
              The regions are identified via two-letter binary codes ('10', '01', '11'), hence a valid artgument could look like:
              {'01': 10, '11': 20}. Unmentioned codes are considered to map to 0.
            - A tuple with 3 numbers, denoting the sizes of the regions in the following order:
              (10, 01, 11).
        set_labels: An optional tuple of two strings - set labels. Set it to None to disable set labels.
        set_colors: A tuple of two color specifications, specifying the base colors of the two circles.
            The colors of circle intersection will be computed based on those.
        normalize_to: Deprecated. Use normalize_to argument of matplotlib_venn.layout.venn2.DefaultLayoutAlgorithm instead.
        ax: The axes to plot upon. Defaults to gca().
        subset_label_formatter: A function that converts numeric subset sizes to strings to be shown on the subset patches in the diagram.
            Defaults to "str".
        layout_algorithm: The layout algorithm to determine the scale and position of the three circles. Defaults to
            matplotlib_venn.layout.venn2.DefaultLayoutAlgorithm().

    Returns:
        a `VennDiagram` object that keeps references to the layout information, ``Text`` and ``Patch`` objects used on the plot.

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
    """
    if isinstance(subsets, dict):
        subsets = [subsets.get(t, 0) for t in ["10", "01", "11"]]
    elif len(subsets) == 2:
        subsets = _compute_subset_sizes(*subsets)
    if normalize_to is not None:
        if layout_algorithm is None:
            warnings.warn(
                "normalize_to is deprecated. Please use layout_algorithm=matplotlib_venn.layout.venn2.DefaultLayoutAlgorithm(normalize_to) instead."
            )
        else:
            raise ValueError(
                "normalize_to is deprecated and may not be specified together with a custom layout algorithm."
            )
    if layout_algorithm is None:
        layout_algorithm = DefaultLayoutAlgorithm(normalize_to=normalize_to or 1.0)

    layout = layout_algorithm(subsets, set_labels)
    return _render_layout(
        layout, subsets, set_labels, set_colors, alpha, ax, subset_label_formatter
    )


def _render_layout(
    layout: VennLayout,
    subsets: Venn2SubsetSizes,
    set_labels: Optional[Tuple[str, str]] = ("A", "B"),
    set_colors: Tuple[Any, Any] = ("r", "g"),
    alpha: float = 0.4,
    ax: Optional[Axes] = None,
    subset_label_formatter: Optional[Callable[[float], str]] = None,
) -> VennDiagram:
    """Renders the layout."""
    if subset_label_formatter is None:
        subset_label_formatter = str
    if ax is None:
        ax = gca()
    prepare_venn_axes(ax, layout.centers, layout.radii)
    colors = _compute_colors(*set_colors)
    regions = _compute_regions(layout.centers, layout.radii)
    patches = [r.make_patch() for r in regions]
    for p, c in zip(patches, colors):
        if p is not None:
            p.set_facecolor(c)
            p.set_edgecolor("none")
            p.set_alpha(alpha)
            ax.add_patch(p)
    label_positions = [r.label_position() for r in regions]
    subset_labels = [
        (
            ax.text(lbl[0], lbl[1], subset_label_formatter(s), va="center", ha="center")
            if lbl is not None
            else None
        )
        for (lbl, s) in zip(label_positions, subsets)
    ]
    if set_labels is not None:
        labels = [
            ax.text(lbl.position.x, lbl.position.y, txt, size="large", **lbl.kwargs)
            for (lbl, txt) in zip(layout.set_labels_layout, set_labels)
        ]
    else:
        labels = None
    return VennDiagram(patches, subset_labels, labels, layout.centers, layout.radii)


def _compute_regions(
    centers: Tuple[Point2D, Point2D], radii: Tuple[float, float]
) -> Tuple[VennRegion, VennRegion, VennRegion]:
    """
    Returns a triple of VennRegion objects, describing the three regions of the diagram, corresponding to sets
    (Ab, aB, AB)

    >>> layout = DefaultLayoutAlgorithm()((1, 1, 0.5))
    >>> regions = _compute_regions(layout.centers, layout.radii)
    """
    A = VennCircleRegion(centers[0].asarray(), radii[0])
    B = VennCircleRegion(centers[1].asarray(), radii[1])
    Ab, AB = A.subtract_and_intersect_circle(B.center, B.radius)
    aB, _ = B.subtract_and_intersect_circle(A.center, A.radius)
    return (Ab, aB, AB)


def _compute_colors(
    color_a: Any, color_b: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given two base colors, computes combinations of colors corresponding to all regions of the venn diagram.
    returns a list of 3 elements, providing colors for regions (10, 01, 11).

    >>> str(_compute_colors('r', 'g')).replace(' ', '')
    '(array([1.,0.,0.]),array([0.,0.5,0.]),array([0.7,0.35,0.]))'
    """
    ccv = ColorConverter()
    base_colors = [np.array(ccv.to_rgb(c)) for c in [color_a, color_b]]
    return (base_colors[0], base_colors[1], mix_colors(base_colors[0], base_colors[1]))


def _compute_subset_sizes(
    a: Union[set, Counter], b: Union[set, Counter]
) -> Tuple[float, float, float]:
    """
    Given two set or Counter objects, computes the sizes of (a & ~b, b & ~a, a & b).
    Returns the result as a tuple.

    >>> _compute_subset_sizes(set([1,2,3,4]), set([2,3,4,5,6]))
    (1, 2, 3)
    >>> _compute_subset_sizes(Counter([1,2,3,4]), Counter([2,3,4,5,6]))
    (1, 2, 3)
    >>> _compute_subset_sizes(Counter([]), Counter([]))
    (0, 0, 0)
    >>> _compute_subset_sizes(set([]), set([]))
    (0, 0, 0)
    >>> _compute_subset_sizes(set([1]), set([]))
    (1, 0, 0)
    >>> _compute_subset_sizes(set([1]), set([1]))
    (0, 0, 1)
    >>> _compute_subset_sizes(Counter([1]), Counter([1]))
    (0, 0, 1)
    >>> _compute_subset_sizes(set([1,2]), set([1]))
    (1, 0, 1)
    >>> _compute_subset_sizes(Counter([1,1,2,2,2]), Counter([1,2,3,3]))
    (3, 2, 2)
    >>> _compute_subset_sizes(Counter([1,1,2]), Counter([1,2,2]))
    (1, 1, 2)
    >>> _compute_subset_sizes(Counter([1,1]), set([]))
    Traceback (most recent call last):
    ...
    ValueError: Both arguments must be of the same type
    """
    if not (type(a) == type(b)):
        raise ValueError("Both arguments must be of the same type")
    set_size = (
        len if type(a) != Counter else lambda x: sum(x.values())
    )  # We cannot use len to compute the cardinality of a Counter
    return (set_size(a - b), set_size(b - a), set_size(a & b))

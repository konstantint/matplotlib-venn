"""
Venn diagram plotting routines.
Three-circle venn plotter.

Copyright 2012-2024, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

from typing import Any, Callable, Dict, Optional, Tuple, Union
import numpy as np
import warnings
from collections import Counter

from matplotlib.axes import Axes
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.colors import ColorConverter
from matplotlib.pyplot import gca

from matplotlib_venn._math import circle_circle_intersection, NUMERIC_TOLERANCE, Point2D
from matplotlib_venn._common import VennDiagram, prepare_venn_axes, mix_colors
from matplotlib_venn._region import VennRegion, VennCircleRegion, VennEmptyRegion
from matplotlib_venn.layout.api import VennLayout, VennLayoutAlgorithm
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm

Venn3SubsetSizes = Tuple[float, float, float, float, float, float, float]


def venn3_circles(
    subsets: Union[Tuple[set, set, set], Dict[str, float], Venn3SubsetSizes],
    normalize_to: Optional[float] = None,
    alpha: float = 1.0,
    color: Any = "black",
    linestyle: str = "solid",
    linewidth: str = 2.0,
    ax: Optional[Axes] = None,
    layout_algorithm: Optional[VennLayoutAlgorithm] = None,
    **kwargs
) -> Tuple[Circle, Circle, Circle]:
    """
    Plots only the three circles for the corresponding Venn diagram.
    Useful for debugging or enhancing the basic venn diagram.

    Args:
        subsets: Same as in `venn3`.
        normalize_to: Same as in `venn3`.
        alpha: The alpha parameter of the circle patches.
        color: The edgecolor of the circle patches (as understood by matplotlib).
        linestyle: The linestyle of the circle patches.
        linewidth: The line width of the circle patches.
        ax: Axis to draw upon, defaults to gca().
        layout_algorithm: The layout algorithm to be used. Defaults to matplotlib_venn.layout.venn3.DefaultLayoutAlgorithm(normalize_to).
        **kwargs: passed as-is to matplotlib.patches.Circle.

    Returns:
        a list of three Circle patches plotted.

    >>> plot = venn3_circles({'001': 10, '100': 20, '010': 21, '110': 13, '011': 14})
    >>> plot = venn3_circles([set(['A','B','C']), set(['A','D','E','F']), set(['D','G','H'])])
    """
    # Prepare parameters
    if isinstance(subsets, dict):
        subsets = [
            subsets.get(t, 0) for t in ["100", "010", "110", "001", "101", "011", "111"]
        ]
    elif len(subsets) == 3:
        subsets = _compute_subset_sizes(*subsets)

    if normalize_to is not None:
        if layout_algorithm is None:
            warnings.warn(
                "normalize_to is deprecated. Please use layout_algorithm=matplotlib_venn.layout.venn3.DefaultLayoutAlgorithm(normalize_to) instead."
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


def venn3(
    subsets: Union[Tuple[set, set, set], Dict[str, float], Venn3SubsetSizes],
    set_labels: Optional[Tuple[str, str, str]] = ("A", "B", "C"),
    set_colors: Tuple[Any, Any, Any] = ("r", "g", "b"),
    alpha: float = 0.4,
    normalize_to: Optional[float] = None,
    ax: Optional[Axes] = None,
    subset_label_formatter: Optional[Callable[[float], str]] = None,
    layout_algorithm: Optional[VennLayoutAlgorithm] = None,
) -> VennDiagram:
    """Plots a 3-set area-weighted Venn diagram.

    Note: if some of the circles happen to have zero area, you will probably not get a nice picture.

    Args:
        subsets: one of the following:
            - A tuple of three set objects.
            - A dict, providing relative sizes of the seven diagram regions.
              The regions are identified via three-letter binary codes ('100', '010', etc), hence a valid artgument could look like:
              {'001': 10, '010': 20, '110':30, ...}. Unmentioned codes are considered to map to 0.
            - A tuple with 7 numbers, denoting the sizes of the regions in the following order:
              (100, 010, 110, 001, 101, 011, 111).
        set_labels: An optional tuple of three strings - set labels. Set it to None to disable set labels.
        set_colors: A tuple of three color specifications, specifying the base colors of the three circles.
            The colors of circle intersections will be computed based on those.
        normalize_to: Deprecated. Use normalize_to argument of matplotlib_venn.layout.venn3.DefaultLayoutAlgorithm instead.
        ax: The axes to plot upon. Defaults to gca().
        subset_label_formatter: A function that converts numeric subset sizes to strings to be shown on the subset patches in the diagram.
            Defaults to "str".
        layout_algorithm: The layout algorithm to determine the scale and position of the three circles. Defaults to
            matplotlib_venn.layout.venn3.DefaultLayoutAlgorithm().

    Returns:
        a `VennDiagram` object that keeps references to the layout information, ``Text`` and ``Patch`` objects used on the plot.

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
    """
    # Prepare parameters
    if isinstance(subsets, dict):
        subsets = [
            subsets.get(t, 0) for t in ["100", "010", "110", "001", "101", "011", "111"]
        ]
    elif len(subsets) == 3:
        subsets = _compute_subset_sizes(*subsets)

    if normalize_to is not None:
        if layout_algorithm is None:
            warnings.warn(
                "normalize_to is deprecated. Please use layout_algorithm=matplotlib_venn.layout.venn3.DefaultLayoutAlgorithm(normalize_to) instead."
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
    subsets: Venn3SubsetSizes,
    set_labels: Optional[Tuple[str, str, str]] = ("A", "B", "C"),
    set_colors: Tuple[Any, Any, Any] = ("r", "g", "b"),
    alpha: float = 0.4,
    ax: Optional[Axes] = None,
    subset_label_formatter: Optional[Callable[[float], str]] = None,
) -> VennDiagram:
    """Given a VennLayout and the relevant rendering information, generates the diagram."""
    if subset_label_formatter is None:
        subset_label_formatter = str
    if ax is None:
        ax = gca()
    prepare_venn_axes(ax, layout.centers, layout.radii)
    colors = _compute_colors(*set_colors)
    regions = list(_compute_regions(layout.centers, layout.radii))

    # Remove regions that are too small from the diagram
    MIN_REGION_SIZE = 1e-4
    for i in range(len(regions)):
        if regions[i].size() < MIN_REGION_SIZE and subsets[i] == 0:
            regions[i] = VennEmptyRegion()

    # There is a rare case (Issue #12) when the middle region is visually empty
    # (the positioning of the circles does not let them intersect), yet the corresponding value is not 0.
    # we address it separately here by positioning the label of that empty region in a custom way
    if isinstance(regions[6], VennEmptyRegion) and subsets[6] > 0:
        intersections = [
            circle_circle_intersection(
                layout.centers[i].asarray(),
                layout.radii[i] + 0.001,
                layout.centers[j].asarray(),
                layout.radii[j] + 0.001,
            )
            for (i, j) in [(0, 1), (1, 2), (2, 0)]
        ]
        middle_pos = np.mean([i[0] for i in intersections], 0)
        regions[6] = VennEmptyRegion(middle_pos)

    # Create and add patches and text
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

    # Position set labels
    if set_labels is not None:
        labels = [
            ax.text(lbl.position.x, lbl.position.y, txt, size="large", **lbl.kwargs)
            for (lbl, txt) in zip(layout.set_labels_layout, set_labels)
        ]
    else:
        labels = None
    return VennDiagram(patches, subset_labels, labels, layout.centers, layout.radii)


def _compute_regions(
    centers: Tuple[Point2D, Point2D, Point2D], radii: Tuple[float, float, float]
) -> Tuple[
    VennRegion, VennRegion, VennRegion, VennRegion, VennRegion, VennRegion, VennRegion
]:
    """
    Given the three centers and radii of circles, returns the 7 regions, comprising the venn diagram, as VennRegion objects.

    Regions are returned in order (Abc, aBc, ABc, abC, AbC, aBC, ABC)

    >>> layout = DefaultLayoutAlgorithm()((1, 1, 1, 1, 1, 1, 1))
    >>> regions = _compute_regions(layout.centers, layout.radii)
    """
    A = VennCircleRegion(centers[0].asarray(), radii[0])
    B = VennCircleRegion(centers[1].asarray(), radii[1])
    C = VennCircleRegion(centers[2].asarray(), radii[2])
    Ab, AB = A.subtract_and_intersect_circle(B.center, B.radius)
    ABc, ABC = AB.subtract_and_intersect_circle(C.center, C.radius)
    Abc, AbC = Ab.subtract_and_intersect_circle(C.center, C.radius)
    aB, _ = B.subtract_and_intersect_circle(A.center, A.radius)
    aBc, aBC = aB.subtract_and_intersect_circle(C.center, C.radius)
    aC, _ = C.subtract_and_intersect_circle(A.center, A.radius)
    abC, _ = aC.subtract_and_intersect_circle(B.center, B.radius)
    return (Abc, aBc, ABc, abC, AbC, aBC, ABC)


def _compute_colors(
    color_a: Any, color_b: Any, color_c: Any
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Given three base colors, computes combinations of colors corresponding to all regions of the venn diagram.
    returns a list of 7 elements, providing colors for regions (100, 010, 110, 001, 101, 011, 111).

    >>> str(_compute_colors('r', 'g', 'b')).replace(' ', '')
    '(array([1.,0.,0.]),...,array([0.4,0.2,0.4]))'
    """
    ccv = ColorConverter()
    base_colors = [np.array(ccv.to_rgb(c)) for c in [color_a, color_b, color_c]]
    return (
        base_colors[0],
        base_colors[1],
        mix_colors(base_colors[0], base_colors[1]),
        base_colors[2],
        mix_colors(base_colors[0], base_colors[2]),
        mix_colors(base_colors[1], base_colors[2]),
        mix_colors(base_colors[0], base_colors[1], base_colors[2]),
    )


def _compute_subset_sizes(
    a: Union[set, Counter], b: Union[set, Counter], c: Union[set, Counter]
) -> Venn3SubsetSizes:
    """
    Given three set or Counter objects, computes the sizes of (a & ~b & ~c, ~a & b & ~c, a & b & ~c, ....),
    as needed by the subsets parameter of venn3 and venn3_circles.
    Returns the result as a tuple.

    >>> _compute_subset_sizes(set([1,2,3]), set([2,3,4]), set([3,4,5,6]))
    (1, 0, 1, 2, 0, 1, 1)
    >>> _compute_subset_sizes(Counter([1,2,3]), Counter([2,3,4]), Counter([3,4,5,6]))
    (1, 0, 1, 2, 0, 1, 1)
    >>> _compute_subset_sizes(Counter([1,1,1]), Counter([1,1,1]), Counter([1,1,1,1]))
    (0, 0, 0, 1, 0, 0, 3)
    >>> _compute_subset_sizes(Counter([1,1,2,2,3,3]), Counter([2,2,3,3,4,4]), Counter([3,3,4,4,5,5,6,6]))
    (2, 0, 2, 4, 0, 2, 2)
    >>> _compute_subset_sizes(Counter([1,2,3]), Counter([2,2,3,3,4,4]), Counter([3,3,4,4,4,5,5,6]))
    (1, 1, 1, 4, 0, 3, 1)
    >>> _compute_subset_sizes(set([]), set([]), set([]))
    (0, 0, 0, 0, 0, 0, 0)
    >>> _compute_subset_sizes(set([1]), set([]), set([]))
    (1, 0, 0, 0, 0, 0, 0)
    >>> _compute_subset_sizes(set([]), set([1]), set([]))
    (0, 1, 0, 0, 0, 0, 0)
    >>> _compute_subset_sizes(set([]), set([]), set([1]))
    (0, 0, 0, 1, 0, 0, 0)
    >>> _compute_subset_sizes(Counter([]), Counter([]), Counter([1]))
    (0, 0, 0, 1, 0, 0, 0)
    >>> _compute_subset_sizes(set([1]), set([1]), set([1]))
    (0, 0, 0, 0, 0, 0, 1)
    >>> _compute_subset_sizes(set([1,3,5,7]), set([2,3,6,7]), set([4,5,6,7]))
    (1, 1, 1, 1, 1, 1, 1)
    >>> _compute_subset_sizes(Counter([1,3,5,7]), Counter([2,3,6,7]), Counter([4,5,6,7]))
    (1, 1, 1, 1, 1, 1, 1)
    >>> _compute_subset_sizes(Counter([1,3,5,7]), set([2,3,6,7]), set([4,5,6,7]))
    Traceback (most recent call last):
    ...
    ValueError: All arguments must be of the same type
    """
    if not (type(a) == type(b) == type(c)):
        raise ValueError("All arguments must be of the same type")
    set_size = (
        len if type(a) != Counter else lambda x: sum(x.values())
    )  # We cannot use len to compute the cardinality of a Counter
    return (
        set_size(
            a - (b | c)
        ),  # TODO: This is certainly not the most efficient way to compute.
        set_size(b - (a | c)),
        set_size((a & b) - c),
        set_size(c - (a | b)),
        set_size((a & c) - b),
        set_size((b & c) - a),
        set_size(a & b & c),
    )

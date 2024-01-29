"""
Venn diagram plotting routines.
Utility routines

Copyright 2012-2024, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

import warnings
from matplotlib_venn._venn2 import venn2
from matplotlib_venn._venn3 import venn3
from matplotlib_venn.layout.venn2 import DefaultLayoutAlgorithm as Venn2Layout
from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm as Venn3Layout


def venn2_unweighted(
    subsets,
    set_labels=("A", "B"),
    set_colors=("r", "g"),
    alpha=0.4,
    normalize_to=1.0,
    subset_areas=(1, 1, 1),
    ax=None,
    subset_label_formatter=None,
):
    """
    This function is deprecated and will be removed in a future version.
    Use venn2(..., layout_algorithm=matplotlib_venn.layout.venn2.DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1))) instead.
    """
    warnings.warn(
        "venn2_unweighted is deprecated. Use venn2 with the appropriate layout_algorithm instead."
    )
    return venn2(
        subsets,
        set_labels,
        set_colors,
        alpha,
        ax,
        subset_label_formatter=subset_label_formatter,
        layout_algorithm=Venn2Layout(
            normalize_to=normalize_to, fixed_subset_sizes=subset_areas
        ),
    )


def venn3_unweighted(
    subsets,
    set_labels=("A", "B", "C"),
    set_colors=("r", "g", "b"),
    alpha=0.4,
    normalize_to=1.0,
    subset_areas=(1, 1, 1, 1, 1, 1, 1),
    ax=None,
    subset_label_formatter=None,
):
    """
    This function is deprecated and will be removed in a future version.
    Use venn3(..., layout_algorithm=matplotlib_venn.layout.venn3.DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1,1,1,1,1))) instead.
    """
    warnings.warn(
        "venn3_unweighted is deprecated. Use venn3 with the appropriate layout_algorithm instead."
    )
    return venn3(
        subsets,
        set_labels,
        set_colors,
        alpha,
        ax,
        subset_label_formatter=subset_label_formatter,
        layout_algorithm=Venn3Layout(
            normalize_to=normalize_to, fixed_subset_sizes=subset_areas
        ),
    )

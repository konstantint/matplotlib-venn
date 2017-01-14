'''
Venn diagram plotting routines.
Utility routines

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''
from matplotlib_venn._venn2 import venn2, compute_venn2_subsets
from matplotlib_venn._venn3 import venn3, compute_venn3_subsets


def venn2_unweighted(subsets, set_labels=('A', 'B'), set_colors=('r', 'g'), alpha=0.4, normalize_to=1.0, subset_areas=(1, 1, 1), ax=None, subset_label_formatter=None):
    '''
    The version of venn2 without area-weighting.
    It is implemented as a wrapper around venn2. Namely, venn2 is invoked as usual, but with all subset areas
    set to 1. The subset labels are then replaced in the resulting diagram with the provided subset sizes.
    
    The parameters are all the same as that of venn2.
    In addition there is a subset_areas parameter, which specifies the actual subset areas.
    (it is (1, 1, 1) by default. You are free to change it, within reason).
    '''
    v = venn2(subset_areas, set_labels, set_colors, alpha, normalize_to, ax)
    # Now rename the labels
    if subset_label_formatter is None:
        subset_label_formatter = str    
    subset_ids = ['10', '01', '11']
    if isinstance(subsets, dict):
        subsets = [subsets.get(t, 0) for t in subset_ids]
    elif len(subsets) == 2:
        subsets = compute_venn2_subsets(*subsets)
    for n, id in enumerate(subset_ids):
        lbl = v.get_label_by_id(id)
        if lbl is not None:
            lbl.set_text(subset_label_formatter(subsets[n]))
    return v


def venn3_unweighted(subsets, set_labels=('A', 'B', 'C'), set_colors=('r', 'g', 'b'), alpha=0.4, normalize_to=1.0, subset_areas=(1, 1, 1, 1, 1, 1, 1), ax=None, subset_label_formatter=None):
    '''
    The version of venn3 without area-weighting.
    It is implemented as a wrapper around venn3. Namely, venn3 is invoked as usual, but with all subset areas
    set to 1. The subset labels are then replaced in the resulting diagram with the provided subset sizes.
    
    The parameters are all the same as that of venn2.
    In addition there is a subset_areas parameter, which specifies the actual subset areas.
    (it is (1, 1, 1, 1, 1, 1, 1) by default. You are free to change it, within reason).
    '''
    v = venn3(subset_areas, set_labels, set_colors, alpha, normalize_to, ax)
    # Now rename the labels
    if subset_label_formatter is None:
        subset_label_formatter = str    
    subset_ids = ['100', '010', '110', '001', '101', '011', '111']
    if isinstance(subsets, dict):
        subsets = [subsets.get(t, 0) for t in subset_ids]
    elif len(subsets) == 3:
        subsets = compute_venn3_subsets(*subsets)
    for n, id in enumerate(subset_ids):
        lbl = v.get_label_by_id(id)
        if lbl is not None:
            lbl.set_text(subset_label_formatter(subsets[n]))
    return v
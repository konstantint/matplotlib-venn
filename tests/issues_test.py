'''
Venn diagram plotting routines.
Test module (meant to be used via py.test).

Copyright 2015, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''

def test_issue_17():
    import matplotlib_venn as mv
    import numpy as np
    venn_areas = mv._venn3.compute_venn3_areas((135, 409, 17398, 122, 201, 135, 122), normalize_to=1.0, _minimal_area=1e-6)
    b = mv._venn3.solve_venn3_circles(venn_areas)
    assert not np.any(np.isnan(b[0]))


def test_pr_28():
    import matplotlib_venn as mv
    v = mv.venn3((1, 2, 3, 4, 5, 6, 7), subset_label_formatter = None)
    assert v.get_label_by_id('010').get_text() == '2'
    v = mv.venn3((1, 2, 3, 4, 5, 6, 7), subset_label_formatter = lambda x: 'Value: %+0.3f' % (x / 100.0))
    assert v.get_label_by_id('010').get_text() == 'Value: +0.020'
    v = mv.venn2((1, 2, 3), subset_label_formatter = None)
    assert v.get_label_by_id('01').get_text() == '2'
    v = mv.venn2((1, 2, 3), subset_label_formatter = lambda x: 'Value: %+0.3f' % (x / 100.0))
    assert v.get_label_by_id('01').get_text() == 'Value: +0.020'
    
    v = mv.venn3_unweighted((1, 2, 3, 4, 5, 6, 7), subset_label_formatter = lambda x: 'Value: %+0.3f' % (x / 100.0))
    assert v.get_label_by_id('010').get_text() == 'Value: +0.020'
    v = mv.venn2_unweighted((1, 2, 3), subset_label_formatter = lambda x: 'Value: %+0.3f' % (x / 100.0))
    assert v.get_label_by_id('01').get_text() == 'Value: +0.020'

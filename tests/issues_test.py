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

    
'''
Venn diagram plotting routines.
Test module (meant to be used via py.test).

Tests of the classes and methods in _regions.py

Copyright 2014, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''
import pytest
import os
import numpy as np
from tests.utils import exec_ipynb
from matplotlib_venn._region import VennCircleRegion, VennArcgonRegion, VennRegionException
from matplotlib_venn._math import tol


def test_circle_region():
    with pytest.raises(VennRegionException):
        vcr = VennCircleRegion((0, 0), -1)
        
    vcr = VennCircleRegion((0, 0), 10)
    assert abs(vcr.size() - np.pi*100) <= tol
    
    # Interact with non-intersecting circle
    sr, ir = vcr.subtract_and_intersect_circle((11, 1), 1)
    assert sr == vcr
    assert ir.is_empty()
    
    # Interact with self
    sr, ir = vcr.subtract_and_intersect_circle((0, 0), 10)
    assert sr.is_empty()
    assert ir == vcr
    
    # Interact with a circle that makes a hole
    with pytest.raises(VennRegionException):
        sr, ir = vcr.subtract_and_intersect_circle((0, 8.9), 1)
    
    # Interact with a circle that touches the side from the inside
    for (a, r) in [(0, 1), (90, 2), (180, 3), (290, 0.01), (42, 9.99), (-0.1, 9.999), (180.1, 0.001)]:
        cx = np.cos(a * np.pi / 180.0) * (10 - r)
        cy = np.sin(a * np.pi / 180.0) * (10 - r)
        #print "Next test case", a, r, cx, cy, r
        TEST_TOLERANCE = tol if r > 0.001 and r < 9.999 else 1e-4 # For tricky circles the numeric errors for arc lengths are just too big here
        
        sr, ir = vcr.subtract_and_intersect_circle((cx, cy), r)
        sr.verify()
        ir.verify()
        assert len(sr.arcs) == 2 and len(ir.arcs) == 2
        for a in sr.arcs:
            assert abs(a.length_degrees() - 360) < TEST_TOLERANCE 
        assert abs(ir.arcs[0].length_degrees() - 0) < TEST_TOLERANCE
        assert abs(ir.arcs[1].length_degrees() - 360) < TEST_TOLERANCE
        assert abs(sr.size() + np.pi*r**2 - vcr.size()) < tol
        assert abs(ir.size() - np.pi*r**2) < tol
    
    # Interact with a circle that touches the side from the outside
    for (a, r) in [(0, 1), (90, 2), (180, 3), (290, 0.01), (42, 9.99), (-0.1, 9.999), (180.1, 0.001)]:
        cx = np.cos(a * np.pi / 180.0) * (10 + r)
        cy = np.sin(a * np.pi / 180.0) * (10 + r)
        
        sr, ir = vcr.subtract_and_intersect_circle((cx, cy), r)
        # Depending on numeric roundoff we may get either an self and VennEmptyRegion or two arc regions. In any case the sizes should match
        assert abs(sr.size() + ir.size() - vcr.size()) < tol
        if (sr == vcr):
            assert ir.is_empty()
        else:
            sr.verify()
            ir.verify()
            assert len(sr.arcs) == 2 and len(ir.arcs) == 2
            assert abs(sr.arcs[0].length_degrees() - 0) < tol
            assert abs(sr.arcs[1].length_degrees() - 360) < tol
            assert abs(ir.arcs[0].length_degrees() - 0) < tol
            assert abs(ir.arcs[1].length_degrees() - 0) < tol
            
    # Interact with some cases of intersecting circles
    for (a, r) in [(0, 1), (90, 2), (180, 3), (290, 0.01), (42, 9.99), (-0.1, 9.999), (180.1, 0.001)]:
        cx = np.cos(a * np.pi / 180.0) * 10
        cy = np.sin(a * np.pi / 180.0) * 10
        
        sr, ir = vcr.subtract_and_intersect_circle((cx, cy), r)
        sr.verify()
        ir.verify()
        assert len(sr.arcs) == 2 and len(ir.arcs) == 2
        assert abs(sr.size() + ir.size() - vcr.size()) < tol
        assert sr.size() > 0
        assert ir.size() > 0
        
        # Do intersection the other way
        vcr2 = VennCircleRegion([cx, cy], r)
        sr2, ir2 = vcr2.subtract_and_intersect_circle(vcr.center, vcr.radius)
        sr2.verify()
        ir2.verify()
        assert len(sr2.arcs) == 2 and len(ir2.arcs) == 2
        assert abs(sr2.size() + ir2.size() - vcr2.size()) < tol
        assert sr2.size() > 0
        assert ir2.size() > 0
        for i in range(2):
            assert ir.arcs[i].approximately_equal(ir.arcs[i]) 

def test_region_visual():
    pass #exec_ipynb(os.path.join(os.path.dirname(__file__), "region_visual.ipynb"))


def test_region_label_visual():
    exec_ipynb(os.path.join(os.path.dirname(__file__), "region_label_visual.ipynb"))

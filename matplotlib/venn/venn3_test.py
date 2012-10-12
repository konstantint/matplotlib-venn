'''
Venn diagram plotting routines.
Test module (meant to be used via py.test).

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''
from numpy import array, pi, sqrt, arcsin
from _venn3 import *
from _math import *

def test_circle_intersection():
    f = lambda x: (sqrt(1-x**2)*x + arcsin(x))*0.5        # Integral [sqrt(1 - x^2) dx]
    area_x = lambda R: 4*R**2*(f(1) - f(0.5))             # Area of intersection of two circles of radius R at distance R
    
    tests = [(0.0, 0.0, 0.0, 0.0), (0.0, 2.0, 1.0, 0.0), (2.0, 0.0, 1.0, 0.0), 
             (1.0, 1.0, 0, pi), (2.0, 2.0, 0, pi*4), (1, 1, 2, 0.0), (2.5, 3.5, 6.0, 0.0),
             (1, 1, 1, area_x(1)), (0.5, 0.5, 0.5, area_x(0.5)), (1.9, 1.9, 1.9, area_x(1.9))]
    for (r, R, d, a) in tests:
        assert abs(circle_intersection_area(r, R, d) - a) < tol

def test_find_distances_by_area():
    tests = [(0.0, 0.0, 0.0, 0.0), (1.2, 1.3, 0.0, 2.5), (1.0, 1.0, pi, 0.0), (sqrt(1.0/pi), sqrt(1.0/pi), 1.0, 0.0)]
    for (r, R, a, d) in tests:
        assert abs(find_distance_by_area(r, R, a) - d) < tol
    
    tests = [(1, 2, 2), (1, 2, 1.1), (2, 3, 1.5), (2, 3, 1.0), (10, 20, 10), 
             (20, 10, 10), (20, 10, 11), (0.9, 0.9, 0.0)]
    for (r, R, d) in tests:
        a = circle_intersection_area(r, R, d)
        assert abs(find_distance_by_area(r, R, a) - d) < tol

def test_solve_venn3_circles():
    tests = []
    for i in range(8):
        t = [0]*8
        t[i] = 1
        tests.append(tuple(t))
        t = [1]*8
        t[i] = 0
        tests.append(tuple(t))
    tests.append(tuple(range(8)))
    
    for t in tests:
        (R_a, R_b, R_c, d_ab, d_ac, d_bc) = solve_venn3_circles(t)
        t = np.array(t, float)
        if np.sum(t[1:]) > 0:
            t = t/np.sum(t[1:])
        (abc, Abc, aBc, ABc, abC, AbC, aBC, ABC) = t
        assert abs(pi*R_a**2 - (Abc + ABc + AbC + ABC)) < tol
        assert abs(pi*R_b**2 - (aBc + ABc + aBC + ABC)) < tol
        assert abs(pi*R_c**2 - (abC + AbC + aBC + ABC)) < tol
        assert abs(circle_intersection_area(R_a, R_b, d_ab) - (ABc + ABC)) < tol
        assert abs(circle_intersection_area(R_a, R_c, d_ac) - (AbC + ABC)) < tol
        assert abs(circle_intersection_area(R_c, R_b, d_bc) - (aBC + ABC)) < tol

def test_position_venn3_circles():
    from numpy.linalg import norm
    tests = [(1, 1, 1, 2, 2, 2), (1, 2, 3, 0, 2, 2), (1, 1, 1, 0, 0, 0), (1, 2, 1, 1, 0.5, 0.6)]
    for t in tests:
        (R_a, R_b, R_c, d_ab, d_ac, d_bc) = t
        coords = position_venn3_circles(*t)
        print coords
        assert abs(norm(coords[0] - coords[1]) - d_ab) < tol
        assert abs(norm(coords[0] - coords[2]) - d_ac) < tol
        assert abs(norm(coords[2] - coords[1]) - d_bc) < tol
        assert abs(norm(R_a**2 * coords[0] + R_b**2 * coords[1] + R_c**2 * coords[2] - array([0.0, 0.0]))) < tol


def test_circle_circle_intersection():
    from numpy.linalg import norm
    tests = [([0, 0], 1, [1, 0], 1, 2), 
             ([0, 0], 1, [2, 0], 1, 1),
             ([0, 0], 1, [0.5, 0], 0.5, 1),
             ([0, 0], 1, [0, 0], 1, 0),
             ([0, 0], 1, [0, 0.1], 0.8, 0),
             ([0, 0], 1, [2.1, 0], 1, 0),
             ([10, 20], 100, [200, 200], 50, 0),
             ([10, 20], 100, [40, 50], 20, 0),
             ([-2.0, -3.1], 10, [2.0, 3.1], 10, 2),
             ([-3.0, 1.0], 10.0, [0.0, 0.0], 9.0, 2)]
    for (C_a, r_a, C_b, r_b, num_intersections) in tests:
        res = circle_circle_intersection(C_a, r_a, C_b, r_b)
        res2 = circle_circle_intersection(C_b, r_b, C_a, r_a)
        if num_intersections == 0:
            assert res is None
            assert res2 is None
        else:
            assert res is not None
            assert res2 is not None
            assert res.shape == (2,2)
            assert res2.shape == (2,2)
            C_a, C_b = array(C_a, float), array(C_b, float)
            for pt in res:
                assert abs(norm(pt - C_a) - r_a) < tol
                assert abs(norm(pt - C_b) - r_b) < tol
            if num_intersections == 1:
                assert abs(norm(res[0] - res[1])) < tol
            else:
                assert abs(norm(res[0] - res[1])) > tol
                # Verify the order of points
                v2, v1 = res[0] - C_a, res[1] - C_a
                outer_prod = v1[0]*v2[1] - v1[1]*v2[0]
                assert outer_prod < 0
            # Changing the order of circles must change the order of points
            assert abs(norm(res[0] - res2[1])) < tol
            assert abs(norm(res[1] - res2[0])) < tol
    

def test_compute_venn3_regions():
    from numpy import mean
    from numpy.linalg import norm
    coords = array([[-1, 0], [1, 0], [0, -1]], float)
    radii  = [2, 2, 2]
    regions = compute_venn3_regions(coords, radii)
    assert regions[0] is None
    
    region_signatures = [(False, False, False), (True, False, False), (False, True, False), (True, True, False), 
                         (False, False, True),  (True, False, True), (False, True, True), (True, True, True)]
                
    for i in range(1, len(regions)):
        pts, arcs, lbl = regions[i]
        assert len(pts) == 3
        assert len(arcs) == 3
        # Compute for each point which circles it lies upon
        point_circle_relation = array([[norm(c-pt) < r + tol for (c, r) in zip(coords, radii)] for pt in pts])
        # Assert that each point lies at least on one circle
        assert all([any(p) for p in point_circle_relation])
        # Find which circles contain all the points in the region
        region_signature = tuple([all(point_circle_relation[:,j]) for j in range(3)])
        assert region_signature == region_signatures[i]

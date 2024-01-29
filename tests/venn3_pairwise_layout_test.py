"""
Venn diagram plotting routines.
Test module (meant to be used via py.test).

Copyright 2012-2024, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

import numpy as np
from matplotlib_venn.layout.venn3.pairwise import (
    _compute_areas,
    _compute_layout,
)
from matplotlib_venn._math import (
    NUMERIC_TOLERANCE as tol,
    circle_intersection_area,
    circle_circle_intersection,
)


def test_compute_areas():
    tests = []
    for i in range(7):
        t = [0] * 7
        t[i] = 1
        tests.append(tuple(t))
        t = [1] * 7
        t[i] = 0
        tests.append(tuple(t))
    tests.append(tuple(range(7)))

    for t in tests:
        (A, B, C, AB, BC, AC, ABC) = _compute_areas(t, _minimal_area=0)
        t = np.array(t, float)
        t = t / np.sum(t)
        (Abc, aBc, ABc, abC, AbC, aBC, ABC) = t
        assert abs(A - (Abc + ABc + AbC + ABC)) < tol
        assert abs(B - (aBc + ABc + aBC + ABC)) < tol
        assert abs(C - (abC + AbC + aBC + ABC)) < tol
        assert abs(AB - (ABc + ABC)) < tol
        assert abs(AC - (AbC + ABC)) < tol
        assert abs(BC - (aBC + ABC)) < tol


def test_compute_layout():
    from numpy.linalg import norm

    tol = 1e-5  # Test #2 does not pass at the desired tolerance.

    tests = [
        (2, 2, 2, 1, 1, 1, 0),
        (10, 40, 90, 0, 40, 10, 0),
        (1, 1, 1, 0, 0, 0, 0),
        (1.2, 2, 1, 1, 0.5, 0.6, 0),
    ]
    for t in tests:
        (A, B, C, AB, BC, AC, ABC) = t
        layout = _compute_layout(t)
        coords, radii = layout.centers, layout.radii
        assert (
            abs(
                circle_intersection_area(
                    radii[0], radii[1], norm(coords[0].asarray() - coords[1].asarray())
                )
                - AB
            )
            < tol
        )
        assert (
            abs(
                circle_intersection_area(
                    radii[0], radii[2], norm(coords[0].asarray() - coords[2].asarray())
                )
                - AC
            )
            < tol
        )
        assert (
            abs(
                circle_intersection_area(
                    radii[1], radii[2], norm(coords[1].asarray() - coords[2].asarray())
                )
                - BC
            )
            < tol
        )
        assert (
            abs(
                norm(
                    radii[0] ** 2 * coords[0].asarray()
                    + radii[1] ** 2 * coords[1].asarray()
                    + radii[2] ** 2 * coords[2].asarray()
                    - np.array([0.0, 0.0])
                )
            )
            < tol
        )


def test_circle_circle_intersection():
    from numpy.linalg import norm

    tests = [
        ([0, 0], 1, [1, 0], 1, 2),
        ([0, 0], 1, [2, 0], 1, 1),
        ([0, 0], 1, [0.5, 0], 0.5, 1),
        ([0, 0], 1, [0, 0], 1, 0),
        ([0, 0], 1, [0, 0.1], 0.8, 0),
        ([0, 0], 1, [2.1, 0], 1, 0),
        ([10, 20], 100, [200, 200], 50, 0),
        ([10, 20], 100, [40, 50], 20, 0),
        ([-2.0, -3.1], 10, [2.0, 3.1], 10, 2),
        ([-3.0, 1.0], 10.0, [0.0, 0.0], 9.0, 2),
    ]
    for C_a, r_a, C_b, r_b, num_intersections in tests:
        res = circle_circle_intersection(C_a, r_a, C_b, r_b)
        res2 = circle_circle_intersection(C_b, r_b, C_a, r_a)
        if num_intersections == 0:
            assert res is None
            assert res2 is None
        else:
            assert res is not None
            assert res2 is not None
            assert res.shape == (2, 2)
            assert res2.shape == (2, 2)
            C_a, C_b = np.array(C_a, float), np.array(C_b, float)
            for pt in res:
                assert abs(norm(pt - C_a) - r_a) < tol
                assert abs(norm(pt - C_b) - r_b) < tol
            if num_intersections == 1:
                assert abs(norm(res[0] - res[1])) < tol
            else:
                assert abs(norm(res[0] - res[1])) > tol
                # Verify the order of points
                v2, v1 = res[0] - C_a, res[1] - C_a
                outer_prod = v1[0] * v2[1] - v1[1] * v2[0]
                assert outer_prod < 0
            # Changing the order of circles must change the order of points
            assert abs(norm(res[0] - res2[1])) < tol
            assert abs(norm(res[1] - res2[0])) < tol

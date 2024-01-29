"""
Venn diagram plotting routines.
Test module (meant to be used via py.test).

Copyright 2012-2024, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

from numpy import pi, sqrt, arcsin
from matplotlib_venn._math import (
    NUMERIC_TOLERANCE as tol,
    circle_intersection_area,
    find_distance_by_area,
)


def test_circle_intersection():
    f = lambda x: (sqrt(1 - x**2) * x + arcsin(x)) * 0.5  # Integral [sqrt(1 - x^2) dx]
    area_x = (
        lambda R: 4 * R**2 * (f(1) - f(0.5))
    )  # Area of intersection of two circles of radius R at distance R

    tests = [
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 2.0, 1.0, 0.0),
        (2.0, 0.0, 1.0, 0.0),
        (1.0, 1.0, 0, pi),
        (2.0, 2.0, 0, pi * 4),
        (1, 1, 2, 0.0),
        (2.5, 3.5, 6.0, 0.0),
        (1, 1, 1, area_x(1)),
        (0.5, 0.5, 0.5, area_x(0.5)),
        (1.9, 1.9, 1.9, area_x(1.9)),
    ]
    for r, R, d, a in tests:
        assert abs(circle_intersection_area(r, R, d) - a) < tol


def test_find_distances_by_area():
    tests = [
        (0.0, 0.0, 0.0, 0.0),
        (1.2, 1.3, 0.0, 2.5),
        (1.0, 1.0, pi, 0.0),
        (sqrt(1.0 / pi), sqrt(1.0 / pi), 1.0, 0.0),
    ]
    for r, R, a, d in tests:
        assert abs(find_distance_by_area(r, R, a, 0.0) - d) < tol

    tests = [
        (1, 2, 2),
        (1, 2, 1.1),
        (2, 3, 1.5),
        (2, 3, 1.0),
        (10, 20, 10),
        (20, 10, 10),
        (20, 10, 11),
        (0.9, 0.9, 0.0),
    ]
    for r, R, d in tests:
        a = circle_intersection_area(r, R, d)
        assert abs(find_distance_by_area(r, R, a, 0.0) - d) < tol

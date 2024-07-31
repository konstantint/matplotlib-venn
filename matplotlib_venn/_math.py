"""
Venn diagram plotting routines.
Math helper functions.

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

from typing import Optional, Sequence, Union
from scipy.optimize import brentq
import math
import numpy as np


class Point2D:
    """A simple representation of a 2D point.

    Note that the methods below use raw 2D np.arrays to represent points.
    This is due to the original desire to keep things simple and assumption that those
    are all internal.
    Since version 1.0.0 we expose layout logic for external interfacing and
    would rather use semantically meaningful structures.
    Maybe this should move over to a module without an underscore in the name.
    """

    __slots__ = ("x", "y")

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def asarray(self):
        return np.array([self.x, self.y], float)

    def __add__(self, other: "Point2D") -> "Point2D":
        return Point2D(self.x + other.x, self.y + other.y)

    def __repr__(self):
        return "Point2D({}, {})".format(self.x, self.y)


Point2DInternal = Union[Sequence[float], np.ndarray]

NUMERIC_TOLERANCE = 1e-10


def point_in_circle(
    pt: Point2DInternal, center: Point2DInternal, radius: float
) -> bool:
    """
    Returns true if a given point is located inside (or on the border) of a circle.

    >>> point_in_circle((0, 0), (0, 0), 1)
    True
    >>> point_in_circle((1, 0), (0, 0), 1)
    True
    >>> point_in_circle((1, 1), (0, 0), 1)
    False
    """
    d = np.linalg.norm(np.asarray(pt) - np.asarray(center))
    return bool(d <= radius)


def box_product(v1: Point2DInternal, v2: Point2DInternal) -> float:
    """Returns a determinant |v1 v2|. The value is equal to the signed area of a parallelogram built on v1 and v2.
    The value is positive is v2 is to the left of v1.

    >>> box_product((0.0, 1.0), (0.0, 1.0))
    0.0
    >>> box_product((1.0, 0.0), (0.0, 1.0))
    1.0
    >>> box_product((0.0, 1.0), (1.0, 0.0))
    -1.0
    """
    return v1[0] * v2[1] - v1[1] * v2[0]


def circle_intersection_area(r: float, R: float, d: float) -> float:
    """
    Formula from: http://mathworld.wolfram.com/Circle-CircleIntersection.html
    Does not make sense for negative r, R or d

    >>> circle_intersection_area(0.0, 0.0, 0.0)
    0.0
    >>> circle_intersection_area(1.0, 1.0, 0.0)
    3.1415...
    >>> circle_intersection_area(1.0, 1.0, 1.0)
    1.2283...
    """
    if abs(d) < NUMERIC_TOLERANCE:
        minR = min(r, R)
        return np.pi * minR**2
    if abs(r - 0) < NUMERIC_TOLERANCE or abs(R - 0) < NUMERIC_TOLERANCE:
        return 0.0
    d2, r2, R2 = float(d**2), float(r**2), float(R**2)
    arg = (d2 + r2 - R2) / 2 / d / r
    # Even with valid arguments, the above computation may result in things like -1.001
    arg = max(min(arg, 1.0), -1.0)
    A = r2 * math.acos(arg)
    arg = (d2 + R2 - r2) / 2 / d / R
    arg = max(min(arg, 1.0), -1.0)
    B = R2 * math.acos(arg)
    arg = (-d + r + R) * (d + r - R) * (d - r + R) * (d + r + R)
    arg = max(arg, 0)
    C = -0.5 * math.sqrt(arg)
    return A + B + C


def circle_line_intersection(
    center: Point2DInternal, r: float, a: Point2DInternal, b: Point2DInternal
) -> Optional[np.ndarray]:
    """
    Computes two intersection points between the circle centered at <center> and radius <r> and a line given by two points a and b.
    If no intersection exists, or if a==b, None is returned. If one intersection exists, it is repeated in the answer.

    >>> circle_line_intersection(np.array([0.0, 0.0]), 1, np.array([-1.0, 0.0]), np.array([1.0, 0.0]))
    array([[ 1.,  0.],
           [-1.,  0.]])
    >>> abs(np.round(circle_line_intersection(np.array([1.0, 1.0]), np.sqrt(2), np.array([-1.0, 1.0]), np.array([1.0, -1.0])), 6)).tolist()
    [[0.0, 0.0], [0.0, 0.0]]
    """
    s = b - a
    # Quadratic eqn coefs
    A = np.linalg.norm(s) ** 2
    if abs(A) < NUMERIC_TOLERANCE:
        return None
    B = 2 * np.dot(a - center, s)
    C = np.linalg.norm(a - center) ** 2 - r**2
    disc = B**2 - 4 * A * C
    if disc < 0.0:
        return None
    t1 = (-B + math.sqrt(disc)) / 2.0 / A
    t2 = (-B - math.sqrt(disc)) / 2.0 / A
    return np.array([a + t1 * s, a + t2 * s])


def find_distance_by_area(
    r: float, R: float, a: float, numeric_correction: float = 0.0001
) -> float:
    """
    Solves circle_intersection_area(r, R, d) == a for d numerically (analytical solution seems to be too ugly to pursue).
    Assumes that a < pi * min(r, R)**2, will fail otherwise.

    The numeric correction parameter is used whenever the computed distance is exactly (R - r) (i.e. one circle must be inside another).
    In this case the result returned is (R-r+correction). This helps later when we position the circles and need to ensure they intersect.

    >>> find_distance_by_area(1, 1, 0, 0.0)
    2.0
    >>> round(find_distance_by_area(1, 1, 3.1415, 0.0), 4)
    0.0
    >>> d = find_distance_by_area(2, 3, 4, 0.0)
    >>> d
    3.37...
    >>> round(circle_intersection_area(2, 3, d), 10)
    4.0
    >>> find_distance_by_area(1, 2, np.pi)
    1.0001
    """
    if r > R:
        r, R = R, r
    if abs(a) < NUMERIC_TOLERANCE:
        return float(r + R)
    if abs(min(r, R) ** 2 * np.pi - a) < NUMERIC_TOLERANCE:
        return abs(R - r + numeric_correction)
    return brentq(lambda x: circle_intersection_area(r, R, x) - a, R - r, R + r)


def circle_circle_intersection(
    C_a: Point2DInternal, r_a: float, C_b: Point2DInternal, r_b: float
) -> Optional[np.ndarray]:
    """
    Finds the coordinates of the intersection points of two circles A and B.
    Circle center coordinates C_a and C_b, should be given as tuples (or 1x2 arrays).
    Returns a 2x2 array result with result[0] being the first intersection point (to the right of the vector C_a -> C_b)
    and result[1] being the second intersection point.

    If there is a single intersection point, it is repeated in output.
    If there are no intersection points or an infinite number of those, None is returned.

    >>> circle_circle_intersection([0, 0], 1, [1, 0], 1) # Two intersection points
    array([[ 0.5      , -0.866...],
           [ 0.5      ,  0.866...]])
    >>> circle_circle_intersection([0, 0], 1, [2, 0], 1).tolist() # Single intersection point (circles touch from outside)
    [[1.0, 0.0], [1.0, 0.0]]
    >>> circle_circle_intersection([0, 0], 1, [0.5, 0], 0.5).tolist() # Single intersection point (circles touch from inside)
    [[1.0, 0.0], [1.0, 0.0]]
    >>> circle_circle_intersection([0, 0], 1, [0, 0], 1) is None # Infinite number of intersections (circles coincide)
    True
    >>> circle_circle_intersection([0, 0], 1, [0, 0.1], 0.8) is None # No intersections (one circle inside another)
    True
    >>> circle_circle_intersection([0, 0], 1, [2.1, 0], 1) is None # No intersections (one circle outside another)
    True
    """
    C_a, C_b = np.asarray(C_a, float), np.asarray(C_b, float)
    v_ab = C_b - C_a
    d_ab = np.linalg.norm(v_ab)
    if (
        np.abs(d_ab) < NUMERIC_TOLERANCE
    ):  # No intersection points or infinitely many of them (circle centers coincide)
        return None
    cos_gamma = (d_ab**2 + r_a**2 - r_b**2) / 2.0 / d_ab / r_a

    if (
        abs(cos_gamma) > 1.0 + NUMERIC_TOLERANCE / 10
    ):  # Allow for a tiny numeric tolerance here too (always better to be return something instead of None, if possible)
        return None  # No intersection point (circles do not touch)
    if cos_gamma > 1.0:
        cos_gamma = 1.0
    if cos_gamma < -1.0:
        cos_gamma = -1.0

    sin_gamma = math.sqrt(1 - cos_gamma**2)
    u = v_ab / d_ab
    v = np.array([-u[1], u[0]])
    pt1 = C_a + r_a * cos_gamma * u - r_a * sin_gamma * v
    pt2 = C_a + r_a * cos_gamma * u + r_a * sin_gamma * v
    return np.array([pt1, pt2])


def vector_angle_in_degrees(v: Point2DInternal) -> float:
    """
    Given a vector, returns its elevation angle in degrees (-180..180).

    >>> vector_angle_in_degrees([1, 0])
    0.0
    >>> vector_angle_in_degrees([1, 1])
    45.0
    >>> vector_angle_in_degrees([0, 1])
    90.0
    >>> vector_angle_in_degrees([-1, 1])
    135.0
    >>> vector_angle_in_degrees([-1, 0])
    180.0
    >>> vector_angle_in_degrees([-1, -1])
    -135.0
    >>> vector_angle_in_degrees([0, -1])
    -90.0
    >>> vector_angle_in_degrees([1, -1])
    -45.0
    """
    return math.atan2(v[1], v[0]) * 180 / np.pi


def normalize_by_center_of_mass(coords: np.ndarray, radii: np.ndarray) -> np.ndarray:
    """
    Given coordinates of circle centers and radii, as two arrays,
    returns new coordinates array, computed such that the center of mass of the
    three circles is (0, 0).

    >>> normalize_by_center_of_mass(np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 3.0]]), np.array([1.0, 1.0, 1.0]))
    array([[-1., -1.],
           [ 1., -1.],
           [ 0.,  2.]])
    >>> normalize_by_center_of_mass(np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 2.0]]), np.array([1.0, 1.0, np.sqrt(2.0)]))
    array([[-1., -1.],
           [ 1., -1.],
           [ 0.,  1.]])
    """
    # Now find the center of mass.
    radii = radii**2
    sum_r = np.sum(radii)
    if sum_r < NUMERIC_TOLERANCE:
        return coords
    else:
        return coords - np.dot(radii, coords) / sum_r

'''
Venn diagram plotting routines.
General-purpose math routines for computing with circular arcs.
Everything is encapsulated in the "Arc" class.

Copyright 2014, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''
import numpy as np
from matplotlib_venn._math import tol, circle_circle_intersection, vector_angle_in_degrees

class Arc(object):
    '''
    A representation of a directed circle arc.
    Essentially it is a namedtuple(center, radius, from_angle, to_angle, direction) with a bunch of helper methods
    for measuring arc lengths and intersections.
    
    The from_angle and to_angle of an arc must be represented in degrees.
    The direction is a boolean, with True corresponding to counterclockwise (positive) direction, and False - clockwise (negative).
    For convenience, the class defines a "sign" property, which is +1 if direction = True and -1 otherwise.
    '''
    
    def __init__(self, center, radius, from_angle, to_angle, direction):
        '''Raises a ValueError if radius is negative.
        
        >>> a = Arc((0, 0), -1, 0, 0, True)
        Traceback (most recent call last):
        ...
        ValueError: Arc's radius may not be negative
        >>> a = Arc((0, 0), 0, 0, 0, True)        
        >>> a = Arc((0, 0), 1, 0, 0, True)        
        '''
        self.center = np.asarray(center)
        self.radius = float(radius)
        if radius < 0.0:
            raise ValueError("Arc's radius may not be negative")
        self.from_angle = float(from_angle)
        self.to_angle = float(to_angle)
        self.direction = direction
        self.sign = 1 if direction else -1
        
    def length_degrees(self):
        '''Computes the length of the arc in degrees.
        The length computation corresponds to what you would expect if you would draw the arc using matplotlib taking direction into account.
        
        >>> Arc((0,0), 1, 0, 0, True).length_degrees()
        0.0
        >>> Arc((0,0), 2, 0, 0, False).length_degrees()
        0.0
        
        >>> Arc((0,0), 3, 0, 1, True).length_degrees()
        1.0
        >>> Arc((0,0), 4, 0, 1, False).length_degrees()
        359.0

        >>> Arc((0,0), 5, 0, 360, True).length_degrees()
        360.0
        >>> Arc((0,0), 6, 0, 360, False).length_degrees()
        0.0
        
        >>> Arc((0,0), 7, 0, 361, True).length_degrees()
        360.0
        >>> Arc((0,0), 8, 0, 361, False).length_degrees()
        359.0
        
        >>> Arc((0,0), 9, 10, -10, True).length_degrees()
        340.0
        >>> Arc((0,0), 10, 10, -10, False).length_degrees()
        20.0
        
        >>> Arc((0,0), 1, 10, 5, True).length_degrees()
        355.0
        >>> Arc((0,0), 1, -10, -5, False).length_degrees()
        355.0
        >>> Arc((0,0), 1, 180, -180, True).length_degrees()
        0.0
        >>> Arc((0,0), 1, 180, -180, False).length_degrees()
        360.0
        >>> Arc((0,0), 1, -180, 180, True).length_degrees()
        360.0
        >>> Arc((0,0), 1, -180, 180, False).length_degrees()
        0.0
        >>> Arc((0,0), 1, 175, -175, True).length_degrees()
        10.0
        >>> Arc((0,0), 1, 175, -175, False).length_degrees()
        350.0
        '''
        d_angle = self.sign * (self.to_angle - self.from_angle)
        if (d_angle > 360):
            return 360.0
        elif (d_angle < 0):
            return d_angle % 360.0
        else:
            return abs(d_angle)   # Yes, abs() is needed, otherwise we get the weird "-0.0" output in the doctests

    def length_radians(self):
        '''Returns the length of the arc in radians.
        
        >>> Arc((0,0), 1, 0, 0, True).length_radians()
        0.0
        >>> Arc((0,0), 2, 0, 360, True).length_radians()
        6.283...
        >>> Arc((0,0), 6, -18, 18, True).length_radians()
        0.6283...
        '''
        return self.length_degrees() * np.pi / 180.0
    
    def length(self):
        '''Returns the actual length of the arc.
        
        >>> Arc((0,0), 2, 0, 360, True).length()
        12.566...
        >>> Arc((0,0), 2, 90, 360, False).length()
        3.1415...
        >>> Arc((0,0), 0, 90, 360, True).length()
        0.0
        '''
        return self.radius * self.length_radians()
    
    def sector_area(self):
        '''Returns the area of the corresponding arc sector.
        
        >>> Arc((0,0), 2, 0, 360, True).sector_area()
        12.566...
        >>> Arc((0,0), 2, 0, 36, True).sector_area()
        1.2566...
        >>> Arc((0,0), 2, 0, 36, False).sector_area()
        11.3097...
        '''
        return self.radius**2 / 2 * self.length_radians()
    
    def segment_area(self):
        '''Returns the area of the corresponding arc segment.
        
        >>> Arc((0,0), 2, 0, 360, True).segment_area()
        12.566...
        >>> Arc((0,0), 2, 0, 180, True).segment_area()
        6.283...
        >>> Arc((0,0), 2, 0, 90, True).segment_area()
        1.14159...
        >>> Arc((0,0), 2, 0, 90, False).segment_area()
        11.42477796...
        >>> Arc((0,0), 2, 0, 0, False).segment_area()
        0.0
        >>> Arc((0, 9), 1, 89.99, 90, False).segment_area()
        3.1415...
        '''
        theta = self.length_radians()
        return self.radius**2 / 2 * (theta - np.sin(theta))

    def angle_as_point(self, angle):
        '''
        Converts a given angle in degrees to the point coordinates on the arc's circle.
        Inverse of point_to_angle.
        
        >>> Arc((1, 1), 1, 0, 0, True).angle_as_point(0).tolist()
        [2.0, 1.0]
        >>> Arc((1, 1), 1, 0, 0, True).angle_as_point(90).tolist()
        [1.0, 2.0]
        >>> np.all(np.isclose(Arc((1, 1), 1, 0, 0, True).angle_as_point(-270), [1.0, 2.0]))
        True
        '''
        angle_rad = angle * np.pi / 180.0
        return self.center + self.radius * np.array([np.cos(angle_rad), np.sin(angle_rad)])
    
    def start_point(self):
        '''
        Returns a 2x1 numpy array with the coordinates of the arc's start point.
        
        >>> Arc((0, 0), 1, 0, 0, True).start_point().tolist()
        [1.0, 0.0]
        >>> Arc((0, 0), 1, 45, 0, True).start_point().tolist()
        [0.707..., 0.707...]
        '''
        return self.angle_as_point(self.from_angle)

    def end_point(self):
        '''
        Returns a 2x1 numpy array with the coordinates of the arc's end point.
        
        >>> np.all(Arc((0, 0), 1, 0, 90, True).end_point() - np.array([0, 1]) < tol)
        True
        '''
        return self.angle_as_point(self.to_angle)
    
    def mid_point(self):
        '''
        Returns the midpoint of the arc as a 1x2 numpy array.
        '''
        midpoint_angle = self.from_angle + self.sign*self.length_degrees() / 2
        return self.angle_as_point(midpoint_angle)
    
    def approximately_equal(self, arc, tolerance=tol):
        '''
        Returns true if the parameters of this arc are within <tolerance> of the parameters of the other arc, and the direction is the same.
        Note that no angle simplification is performed (i.e. some arcs that might be equal in principle are not declared as such
        by this method)
        
        >>> Arc((0, 0), 10, 20, 30, True).approximately_equal(Arc((tol/2, tol/2), 10+tol/2, 20-tol/2, 30-tol/2, True))
        True
        >>> Arc((0, 0), 10, 20, 30, True).approximately_equal(Arc((0, 0), 10, 20, 30, False))
        False
        >>> Arc((0, 0), 10, 20, 30, True).approximately_equal(Arc((0, 0+tol), 10, 20, 30, True))
        False
        '''
        return self.direction == arc.direction \
                and np.all(abs(self.center - arc.center) < tolerance) and abs(self.radius - arc.radius) < tolerance \
                and abs(self.from_angle - arc.from_angle) < tolerance and abs(self.to_angle - arc.to_angle) < tolerance
    
    def point_as_angle(self, pt):
        '''
        Given a point located on the arc's circle, return the corresponding angle in degrees.
        No check is done that the point lies on the circle
        (this is essentially a convenience wrapper around _math.vector_angle_in_degrees)
        
        >>> a = Arc((0, 0), 1, 0, 0, True)
        >>> a.point_as_angle((1, 0))
        0.0
        >>> a.point_as_angle((1, 1))
        45.0
        >>> a.point_as_angle((0, 1))
        90.0
        >>> a.point_as_angle((-1, 1))
        135.0
        >>> a.point_as_angle((-1, 0))
        180.0
        >>> a.point_as_angle((-1, -1))
        -135.0
        >>> a.point_as_angle((0, -1))
        -90.0
        >>> a.point_as_angle((1, -1))
        -45.0
        '''
        return vector_angle_in_degrees(np.asarray(pt) - self.center)
    
    def contains_angle_degrees(self, angle):
        '''
        Returns true, if a point with the corresponding angle (given in degrees) is within the arc.
        Does no tolerance checks (i.e. if the arc is of length 0, you must provide angle == from_angle == to_angle to get a positive answer here)
        
        >>> a = Arc((0, 0), 1, 0, 0, True)
        >>> assert a.contains_angle_degrees(0)
        >>> assert a.contains_angle_degrees(360)
        >>> assert not a.contains_angle_degrees(1)
        
        >>> a = Arc((0, 0), 1, 170, -170, True)
        >>> assert not a.contains_angle_degrees(165)
        >>> assert a.contains_angle_degrees(170)
        >>> assert a.contains_angle_degrees(175)
        >>> assert a.contains_angle_degrees(180)
        >>> assert a.contains_angle_degrees(185)
        >>> assert a.contains_angle_degrees(190)
        >>> assert not a.contains_angle_degrees(195)
        
        >>> assert not a.contains_angle_degrees(-195)
        >>> assert a.contains_angle_degrees(-190)
        >>> assert a.contains_angle_degrees(-185)
        >>> assert a.contains_angle_degrees(-180)
        >>> assert a.contains_angle_degrees(-175)
        >>> assert a.contains_angle_degrees(-170)
        >>> assert not a.contains_angle_degrees(-165)
        >>> assert a.contains_angle_degrees(-170 - 360)
        >>> assert a.contains_angle_degrees(-190 - 360)
        >>> assert a.contains_angle_degrees(170 + 360)
        >>> assert not a.contains_angle_degrees(0)
        >>> assert not a.contains_angle_degrees(100)
        >>> assert not a.contains_angle_degrees(-100)
        '''
        _d = self.sign * (angle - self.from_angle) % 360.0
        return (_d <= self.length_degrees())
    
    def intersect_circle(self, center, radius):
        '''
        Given a circle, finds the intersection point(s) of the arc with the circle.
        Returns a list of 2x1 numpy arrays. The list has length 0, 1 or 2, depending on how many intesection points there are.
        If the circle touches the arc, it is reported as two intersection points (which are equal).
        Points are ordered along the arc.
        Intersection with the same circle as the arc's own (which means infinitely many points usually) is reported as no intersection at all.
        
        >>> a = Arc((0, 0), 1, -60, 60, True)
        >>> str(a.intersect_circle((1, 0), 1)).replace(' ', '')
        '[array([0.5...,-0.866...]),array([0.5...,0.866...])]'
        >>> a.intersect_circle((0.9, 0), 1)
        []
        >>> str(a.intersect_circle((1,-0.1), 1)).replace(' ', '')
        '[array([0.586...,0.810...])]'
        >>> str(a.intersect_circle((1, 0.1), 1)).replace(' ', '')
        '[array([0.586...,-0.810...])]'
        >>> a.intersect_circle((0, 0), 1)  # Infinitely many intersection points
        []
        >>> str(a.intersect_circle((2, 0), 1)).replace(' ', '')  # Touching point, hence repeated twice
        '[array([1.,0.]),array([1.,0.])]'
        
        >>> a = Arc((0, 0), 1, 60, -60, False) # Same arc, different direction
        >>> str(a.intersect_circle((1, 0), 1)).replace(' ', '')
        '[array([0.5...,0.866...]),array([0.5...,-0.866...])]'
        
        >>> a = Arc((0, 0), 1, 120, -120, True)
        >>> a.intersect_circle((-1, 0), 1)
        [array([-0.5...,  0.866...]), array([-0.5..., -0.866...])]
        >>> a.intersect_circle((-0.9, 0), 1)
        []
        >>> a.intersect_circle((-1,-0.1), 1)
        [array([-0.586...,  0.810...])]
        >>> a.intersect_circle((-1, 0.1), 1)
        [array([-0.586..., -0.810...])]
        >>> a.intersect_circle((-2, 0), 1)
        [array([-1.,  0.]), array([-1.,  0.])]
        >>> a = Arc((0, 0), 1, -120, 120, False)
        >>> a.intersect_circle((-1, 0), 1)
        [array([-0.5..., -0.866...]), array([-0.5...,  0.866...])]
        '''
        intersections = circle_circle_intersection(self.center, self.radius, center, radius)
        if intersections is None:
            return []
        
        # Check whether the points lie on the arc and order them accordingly
        _len = self.length_degrees()
        isections = [[self.sign * (self.point_as_angle(pt) - self.from_angle) % 360.0, pt] for pt in intersections]
        
        # Try to find as many candidate intersections as possible (i.e. +- tol within arc limits)
        # Unless arc's length is 360, interpret intersections just before the arc's starting point as belonging to the starting point.
        if _len < 360.0 - tol:
            for isec in isections:
                if isec[0] > 360.0 - tol:
                    isec[0] = 0.0
        
        isections = [(a, pt[0], pt[1]) for (a, pt) in isections if a < _len + tol or a > 360 - tol]
        isections.sort()
        return [np.array([b, c]) for (a, b, c) in isections]
    
    def intersect_arc(self, arc):
        '''
        Given an arc, finds the intersection point(s) of this arc with that.
        Returns a list of 2x1 numpy arrays. The list has length 0, 1 or 2, depending on how many intesection points there are.
        Points are ordered along the arc.
        Intersection with the arc along the same circle (which means infinitely many points usually) is reported as no intersection at all.
        
        >>> a = Arc((0, 0), 1, -90, 90, True)
        >>> str(a.intersect_arc(Arc((1, 0), 1, 90, 270, True))).replace(' ', '')
        '[array([0.5,-0.866...]),array([0.5,0.866...])]'
        >>> str(a.intersect_arc(Arc((1, 0), 1, 90, 180, True))).replace(' ', '')
        '[array([0.5,0.866...])]'
        >>> a.intersect_arc(Arc((1, 0), 1, 121, 239, True))
        []
        >>> str(a.intersect_arc(Arc((1, 0), 1, 120-tol, 240+tol, True))).replace(' ', '')  # Without -tol and +tol the results differ on different architectures due to rounding (see Debian #813782).
        '[array([0.5,-0.866...]),array([0.5,0.866...])]'
        '''
        intersections = self.intersect_circle(arc.center, arc.radius)
        isections = [pt for pt in intersections if arc.contains_angle_degrees(arc.point_as_angle(pt))]
        return isections
    
    def subarc(self, from_angle=None, to_angle=None):
        '''
        Creates a sub-arc from a given angle (or beginning of this arc) to a given angle (or end of this arc).
        Verifies that from_angle and to_angle are within the arc and properly ordered.
        If from_angle is None, start of this arc is used instead.
        If to_angle is None, end of this arc is used instead.
        Angles are given in degrees.
        
        >>> a = Arc((0, 0), 1, 0, 360, True)
        >>> a.subarc(None, None)
        Arc([0.000, 0.000], 1.000,   0.000, 360.000,   True,   degrees=360.000)
        >>> a.subarc(360, None)
        Arc([0.000, 0.000], 1.000,   360.000, 360.000,   True,   degrees=0.000)
        >>> a.subarc(0, None)
        Arc([0.000, 0.000], 1.000,   0.000, 360.000,   True,   degrees=360.000)
        >>> a.subarc(-10, None)
        Arc([0.000, 0.000], 1.000,   350.000, 360.000,   True,   degrees=10.000)
        >>> a.subarc(None, -10)
        Arc([0.000, 0.000], 1.000,   0.000, 350.000,   True,   degrees=350.000)
        >>> a.subarc(1, 359).subarc(2, 358).subarc()
        Arc([0.000, 0.000], 1.000,   2.000, 358.000,   True,   degrees=356.000)
        '''
        
        if from_angle is None:
            from_angle = self.from_angle
        if to_angle is None:
            to_angle = self.to_angle
        cur_length = self.length_degrees()
        d_new_from = self.sign * (from_angle - self.from_angle)
        if (d_new_from != 360.0):
            d_new_from = d_new_from % 360.0
        d_new_to = self.sign * (to_angle - self.from_angle)
        if (d_new_to != 360.0):
            d_new_to = d_new_to % 360.0
        # Gracefully handle numeric precision issues for zero-length arcs
        if abs(d_new_from - d_new_to) < tol:
            d_new_from = d_new_to
        if d_new_to < d_new_from:
            raise ValueError("Subarc to-angle must be smaller than from-angle.")
        if d_new_to > cur_length + tol:
            raise ValueError("Subarc to-angle must lie within the current arc.")
        return Arc(self.center, self.radius, self.from_angle + self.sign*d_new_from, self.from_angle + self.sign*d_new_to, self.direction)
    
    def subarc_between_points(self, p_from=None, p_to=None):
        '''
        Given two points on the arc, extract a sub-arc between those points.
        No check is made to verify the points are actually on the arc.
        It is basically a wrapper around subarc(point_as_angle(p_from), point_as_angle(p_to)).
        Either p_from or p_to may be None to denote first or last arc endpoints.
        
        >>> a = Arc((0, 0), 1, 0, 90, True)
        >>> a.subarc_between_points((1, 0), (np.cos(np.pi/4), np.sin(np.pi/4)))
        Arc([0.000, 0.000], 1.000,   0.000, 45.000,   True,   degrees=45.000)
        >>> a.subarc_between_points(None, None)
        Arc([0.000, 0.000], 1.000,   0.000, 90.000,   True,   degrees=90.000)
        >>> a.subarc_between_points((np.cos(np.pi/4), np.sin(np.pi/4)))
        Arc([0.000, 0.000], 1.000,   45.000, 90.000,   True,   degrees=45.000)
        '''
        a_from = self.point_as_angle(p_from) if p_from is not None else None
        a_to = self.point_as_angle(p_to) if p_to is not None else None
        return self.subarc(a_from, a_to)
    
    def reversed(self):
        '''
        Returns a copy of this arc, with the direction flipped.
        
        >>> Arc((0, 0), 1, 0, 360, True).reversed()
        Arc([0.000, 0.000], 1.000,   360.000, 0.000,   False,   degrees=360.000)
        >>> Arc((0, 0), 1, 175, -175, True).reversed()
        Arc([0.000, 0.000], 1.000,   -175.000, 175.000,   False,   degrees=10.000)
        >>> Arc((0, 0), 1, 0, 370, True).reversed()
        Arc([0.000, 0.000], 1.000,   370.000, 0.000,   False,   degrees=360.000)
        '''
        return Arc(self.center, self.radius, self.to_angle, self.from_angle, not self.direction)
    
    def direction_vector(self, angle):
        '''
        Returns a unit vector, pointing in the arc's movement direction at a given (absolute) angle (in degrees).
        No check is made whether angle lies within the arc's span (the results for angles outside of the arc's span )
        Returns a 2x1 numpy array.
        
        >>> a = Arc((0, 0), 1, 0, 90, True)
        >>> assert all(abs(a.direction_vector(0) - np.array([0.0, 1.0])) < tol)
        >>> assert all(abs(a.direction_vector(45) - np.array([ -0.70710678, 0.70710678])) < 1e-6)
        >>> assert all(abs(a.direction_vector(90) - np.array([-1.0, 0.0])) < tol)
        >>> assert all(abs(a.direction_vector(135) - np.array([-0.70710678, -0.70710678])) < 1e-6)
        >>> assert all(abs(a.direction_vector(-180) - np.array([0.0, -1.0])) < tol)
        >>> assert all(abs(a.direction_vector(-90) - np.array([1.0, 0.0])) < tol)
        >>> a = a.reversed()
        >>> assert all(abs(a.direction_vector(0) - np.array([0.0, -1.0])) < tol)
        >>> assert all(abs(a.direction_vector(45) - np.array([ 0.70710678, -0.70710678])) < 1e-6)
        >>> assert all(abs(a.direction_vector(90) - np.array([1.0, 0.0])) < tol)
        >>> assert all(abs(a.direction_vector(135) - np.array([0.70710678, 0.70710678])) < 1e-6)
        >>> assert all(abs(a.direction_vector(-180) - np.array([0.0, 1.0])) < tol)
        >>> assert all(abs(a.direction_vector(-90) - np.array([-1.0, 0.0])) < tol)
        '''
        a = angle + self.sign * 90
        a = a * np.pi / 180.0
        return np.array([np.cos(a), np.sin(a)])
    
    def fix_360_to_0(self):
        '''
        Sometimes we have to create an arc using from_angle and to_angle computed numerically.
        If from_angle == to_angle, it may sometimes happen that a tiny discrepancy will make from_angle > to_angle, and instead of
        getting a 0-length arc we end up with a 360-degree arc.
        Sometimes we know for sure that a 360-degree arc is not what we want, and in those cases
        the problem is easy to fix. This helper method does that. It checks whether from_angle and to_angle are numerically similar,
        and if so makes them equal.
        
        >>> a = Arc((0, 0), 1, 0, -tol/2, True)
        >>> a
        Arc([0.000, 0.000], 1.000,   0.000, -0.000,   True,   degrees=360.000)
        >>> a.fix_360_to_0()
        >>> a
        Arc([0.000, 0.000], 1.000,   -0.000, -0.000,   True,   degrees=0.000)
        '''
        if abs(self.from_angle - self.to_angle) < tol:
            self.from_angle = self.to_angle
    
    def lies_on_circle(self, center, radius):
        '''Tests whether the arc circle's center and radius match the given ones within <tol> tolerance.
        
        >>> a = Arc((0, 0), 1, 0, 0, False)
        >>> a.lies_on_circle((tol/2, tol/2), 1+tol/2)
        True
        >>> a.lies_on_circle((tol/2, tol/2), 1-tol)
        False
        '''
        return np.all(abs(np.asarray(center) - self.center) < tol) and abs(radius - self.radius) < tol
        
    def __repr__(self):
        return "Arc([%0.3f, %0.3f], %0.3f,   %0.3f, %0.3f,   %s,   degrees=%0.3f)" \
                % (self.center[0], self.center[1], self.radius, self.from_angle, self.to_angle, self.direction, self.length_degrees())

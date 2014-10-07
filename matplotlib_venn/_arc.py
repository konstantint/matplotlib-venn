'''
Venn diagram plotting routines.
General-purpose math routines for computing with circular arcs.
Everything is encapsulated in the "Arc" class.

Copyright 2014, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''
import numpy as np
from matplotlib_venn._math import tol, circle_circle_intersection

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
        self.center = np.asarray(center)
        self.radius = float(radius)
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
        '''
        d_angle = self.sign * (self.to_angle - self.from_angle)
        if (d_angle > 360):
            return 360.0
        elif (d_angle < 0):
            return d_angle % 360.0
        else:
            return abs(d_angle)   # Yes, this is needed, otherwise we get the weird "-0.0" output in the doctests

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

    def start_point(self):
        '''
        Returns a 2x1 numpy array with the coordinates of the arc's start point.
        
        >>> Arc((0, 0), 1, 0, 0, True).start_point()
        array([ 1.,  0.])
        >>> Arc((0, 0), 1, 45, 0, True).start_point()
        array([ 0.707...,  0.707...])
        '''
        x = self.center[0] + self.radius * np.cos(self.from_angle * np.pi / 180.0)
        y = self.center[1] + self.radius * np.sin(self.from_angle * np.pi / 180.0)
        return np.array([x, y])

    def end_point(self):
        '''
        Returns a 2x1 numpy array with the coordinates of the arc's end point.
        
        >>> np.all(Arc((0, 0), 1, 0, 90, True).end_point() - np.array([0, 1]) < tol)
        True
        '''
        x = self.center[0] + self.radius * np.cos(self.to_angle * np.pi / 180.0)
        y = self.center[1] + self.radius * np.sin(self.to_angle * np.pi / 180.0)
        return np.array([x, y])
    
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
    
    def __repr__(self):
        return "Arc([%0.3f, %0.3f], %0.3f,   %0.3f, %0.3f,   %s,   degrees=%0.3f)" \
                % (self.center[0], self.center[1], self.radius, self.from_angle, self.to_angle, self.direction, self.length_degrees())

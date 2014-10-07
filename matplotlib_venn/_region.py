'''
Venn diagram plotting routines.
Math for computing with venn diagram regions.

Copyright 2014, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.

The current logic of drawing the venn diagram is the following:
 - Position the circles.
 - Compute the regions of the diagram based on circles
 - Compute the position of the label within each region.
 - Create matplotlib PathPatch or Circle objects for each of the regions.
 
This module contains functionality necessary for the second and third steps of this process.
The regions of an up to 3-circle Venn diagram may be of the following kinds:
 - No region
 - A circle
 - A 2, 3 or 4-arc "poly-arc-gon".  (I.e. a polygon with up to 4 vertices, that are connected by circle arcs)
 - A set of two 3-arc-gons.

We create each of the regions by starting with a circle, and then either intersecting or subtracting the second and the third circles.
The classes below implement the region representation and the intersection/subtraction procedures.
In addition, each region type has a "label positioning" procedure assigned.
'''
import numpy as np
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib_venn._math import tol, circle_circle_intersection, vector_angle_in_degrees
from matplotlib_venn._arc import Arc

class VennRegionException(Exception):
    pass

class VennRegion(object):
    '''
    This is a superclass of a Venn diagram region, defining the interface that has to be supported by the different region types.
    '''
    def subtract_and_intersect_circle(self, center, radius):
        '''
        Given a circular region, compute two new regions:
        one obtained by subtracting the circle from this region, and another obtained by intersecting the circle with the region.
        
        In all implementations it is assumed that the circle to be subtracted is not completely within
        the current region without touching its borders, i.e. it will not form a "hole" when subtracted.
        
        Arguments:
           center (tuple):  A two-element tuple-like, representing the coordinates of the center of the circle.
           radius (float):  A nonnegative number, the radius of the circle.
           
        Returns:
           a list with two elements - the result of subtracting the circle, and the result of intersecting with the circle.
        '''
        pass    
    
    def label_position(self):
        '''Compute the position of a label for this region and return it as a 1x2 numpy array (x, y).'''
        pass

    def size(self):
        '''Return a number, representing the size of the region. It is not important that the number would be a precise
        measurement, as long as sizes of various regions can be compared to choose the largest one.'''
        pass
    
    def make_patch(self):
        '''Create a matplotlib patch object, corresponding to this region. May return None if no patch has to be created.'''
        pass


class VennEmptyRegion(VennRegion):
    '''
    An empty region. To save some memory, returns [self, self] on the subtract_and_intersect_circle operation.
    
    >>> v = VennEmptyRegion()
    >>> [a, b] = v.subtract_and_intersect_circle((1,2), 3)
    >>> assert a == v and b == v
    >>> assert list(v.label_position()) == [0, 0]
    >>> assert v.size() == 0
    >>> assert v.make_patch() is None
    >>> assert v.is_empty()
    '''
    
    def subtract_and_intersect_circle(self, center, radius):
        return [self, self]
    def size(self):
        return 0
    def label_position(self):
        return np.array([0, 0])
    def make_patch(self):
        return None
    def is_empty(self):  # We use this in tests as an equivalent of isinstance(VennEmptyRegion)
        return True


class VennCircleRegion(VennRegion):
    '''
    A circle-shaped region.
    
    >>> vcr = VennCircleRegion((0, 0), 1)
    >>> vcr.size()
    3.1415...
    >>> vcr.label_position()
    array([ 0.,  0.])
    >>> vcr.make_patch()
    <matplotlib.patches.Circle object at ...>
    >>> sr, ir = vcr.subtract_and_intersect_circle((0.5, 0), 1)
    >>> assert abs(sr.size() + ir.size() - vcr.size()) < tol
    '''
    
    def __init__(self, center, radius):
        self.center = np.asarray(center, float)
        self.radius = abs(radius)
        if (radius < -tol):
            raise VennRegionException("Circle with a negative radius is invalid")
    
    def subtract_and_intersect_circle(self, center, radius):
        '''Will throw a VennRegionException if the circle to be subtracted is completely inside and not touching the given region.'''
        
        # Check whether the target circle intersects us
        center = np.asarray(center, float)
        d = np.linalg.norm(center - self.center)
        if d > (radius + self.radius - tol):
            return [self, VennEmptyRegion()] # The circle does not intersect us
        elif d < tol:
            if radius > self.radius - tol:
                # We are completely covered by that circle or we are the same circle
                return [VennEmptyRegion(), self]
            else:
                # That other circle is inside us and smaller than us - we can't deal with it
                raise VennRegionException("Invalid configuration of circular regions (holes are not supported).")
        else:
            # We *must* intersect the other circle. If it is not the case, then it is inside us completely,
            # and we'll complain.            
            intersections = circle_circle_intersection(self.center, self.radius, center, radius)
            
            if intersections is None:
                raise VennRegionException("Invalid configuration of circular regions (holes are not supported).")
            else:
                # Otherwise the subtracted region is a 2-arc-gon
                # Before we need to convert the intersection points as angles wrt each circle.
                a_1 = vector_angle_in_degrees(intersections[0] - self.center)
                a_2 = vector_angle_in_degrees(intersections[1] - self.center)
                b_1 = vector_angle_in_degrees(intersections[0] - center)
                b_2 = vector_angle_in_degrees(intersections[1] - center)

                # We must take care of the situation where the intersection points happen to be the same
                if (abs(b_1 - b_2) < tol):
                    b_1 = b_2 - tol/2
                if (abs(a_1 - a_2) < tol):
                    a_2 = a_1 + tol/2
                
                # The subtraction is a 2-arc-gon [(AB, B-), (BA, A+)]
                s_arc1 = Arc(center, radius, b_1, b_2, False)
                s_arc2 = Arc(self.center, self.radius, a_2, a_1, True)                
                subtraction = VennArcgonRegion([s_arc1, s_arc2])
                
                # .. and the intersection is a 2-arc-gon [(AB, A+), (BA, B+)]
                i_arc1 = Arc(self.center, self.radius, a_1, a_2, True)
                i_arc2 = Arc(center, radius, b_2, b_1, True)
                intersection = VennArcgonRegion([i_arc1, i_arc2])
                return [subtraction, intersection]
    
    def size(self):
        '''
        Return the area of the circle
        
        >>> VennCircleRegion((0, 0), 1).size()
        3.1415...
        >>> VennCircleRegion((0, 0), 2).size()
        12.56637...
        '''
        return np.pi * self.radius**2;
    
    def label_position(self):
        '''
        The label should be positioned in the center of the circle
        
        >>> VennCircleRegion((0, 0), 1).label_position()
        array([ 0.,  0.])
        >>> VennCircleRegion((-1.2, 3.4), 1).label_position()
        array([-1.2,  3.4])
        '''
        return self.center
    
    def make_patch(self):
        '''
        Returns the corresponding circular patch.
        
        >>> patch = VennCircleRegion((1, 2), 3).make_patch()
        >>> patch
        <matplotlib.patches.Circle object at ...>
        >>> patch.center, patch.radius
        (array([ 1.,  2.]), 3.0)
        '''
        return Circle(self.center, self.radius)


class VennArcgonRegion(VennRegion):
    '''
    A poly-arc region.
    Note that we essentially only support 2, 3 and 4 arced regions,
    whereas intersections and subtractions only work for 2-arc regions.
    '''
    
    def __init__(self, arcs):
        '''
        Create a poly-arc region given a list of Arc objects.        
        The arcs list must be of length 2, 3 or 4.
        The arcs must form a closed polygon, i.e. the last point of each arc must be the first point of the next arc.
        The vertices of a 3 or 4-arcgon must be listed in a CCW order. Arcs must not intersect.
        
        This is not verified in the constructor, but a special verify() method can be used to check
        for validity.
        '''
        self.arcs = arcs
        
    def verify(self):
        '''
        Verify the correctness of the region arcs. Throws an VennRegionException if verification fails
        (or any other exception if it happens during verification).
        '''
        # Verify size of arcs list
        if (len(self.arcs) < 2):
            raise VennRegionException("At least two arcs needed in a poly-arc region")
        if (len(self.arcs) > 4):
            raise VennRegionException("At most 4 arcs are supported currently for poly-arc regions")
        
        # Verify connectedness of arcs
        for i in range(len(self.arcs)):
            if not np.all(self.arcs[i-1].end_point() - self.arcs[i].start_point() < tol):
                raise VennRegionException("Arcs of an poly-arc-gon must be connected via endpoints")
        
        # Verify that arcs do not cross-intersect except at endpoints
        # TODO
        
        # Verify that vertices are properly ordered
        
    
    def subtract_and_intersect_circle(self, center, radius):
        '''
        '''
        return self
    
    def label_position(self):
        return np.array([0, 0])
    
    def size(self):
        '''Return the area of the patch'''
        if len(self.arcs) == 2:
            return sum([a.sign * a.segment_area()  for a in self.arcs])
        else:
            raise VennRegionException("Size function not implemented for 3 and 4-arc regions")
    
    def make_patch(self):
        return Circle(self.center, self.radius)


class VennMultipieceRegion(VennRegion):
    '''
    A region containing several pieces.
    In principle, any number of pieces is supported,
    although no more than 2 should ever be needed in a 3-circle Venn diagram.
    '''
    
    def __init__(self, pieces):
        '''
        Create a multi-piece region from a list of VennRegion objects.
        The list may be empty or contain a single item (although those regions can be converted to a
        VennEmptyRegion or a single region of the necessary type.
        '''
        self.pieces = pieces
            
    def subtract_circle(self, center, radius):
        '''
          Performs subtraction on each piece, drop empty regions.
        '''
        # Subtract the circle from each piece
        results = [p.subtract_circle(center, radius) for p in self.pieces]
        # Drop any VennEmptyRegion instances
        results = [p for p in results if not isinstance(p, VennEmptyRegion)]
        if len(results) == 0:
            return VennEmptyRegion()
        elif len(results) == 1:
            return results[0]
        else:
            return VennMultipieceRegion(results)
        
    def intersect_circle(self, center, radius):
        '''
        Perform piece-wise intersection, drop empty regions.
        '''
        # Subtract the circle from each piece
        results = [p.intersect_circle(center, radius) for p in self.pieces]
        results = [p for p in results if not isinstance(p, VennEmptyRegion)]
        if len(results) == 0:
            return VennEmptyRegion()
        elif len(results) == 1:
            return results[0]
        else:
            return VennMultipieceRegion(results)
        
    def label_position(self):
        '''
        Find the largest region and position the label in that.
        '''
        return np.array([0, 0])
    
    def make_patch(self):
        '''Currently only works if all the pieces are Arcgons.
           In this case returns a multiple-piece path. Otherwise throws an exception.'''
        return None
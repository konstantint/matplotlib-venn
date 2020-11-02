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
 
This module contains functionality necessary for the second, third and fourth steps of this process.

Note that the regions of an up to 3-circle Venn diagram may be of the following kinds:
 - No region
 - A circle
 - A 2, 3 or 4-arc "poly-arc-gon".  (I.e. a polygon with up to 4 vertices, that are connected by circle arcs)
 - A set of two 3-arc-gons.

We create each of the regions by starting with a circle, and then either intersecting or subtracting the second and the third circles.
The classes below implement the region representation, the intersection/subtraction procedures and the conversion to matplotlib patches.
In addition, each region type has a "label positioning" procedure assigned.
'''
import warnings
import numpy as np
from matplotlib.patches import Circle, PathPatch, Path
from matplotlib.path import Path
from matplotlib_venn._math import tol, circle_circle_intersection, vector_angle_in_degrees
from matplotlib_venn._math import point_in_circle, box_product
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
        raise NotImplementedError("Method not implemented")
    
    
    def label_position(self):
        '''Compute the position of a label for this region and return it as a 1x2 numpy array (x, y).
        May return None if label is not applicable.'''
        raise NotImplementedError("Method not implemented")

    def size(self):
        '''Return a number, representing the size of the region. It is not important that the number would be a precise
        measurement, as long as sizes of various regions can be compared to choose the largest one.'''
        raise NotImplementedError("Method not implemented")
    
    def make_patch(self):
        '''Create a matplotlib patch object, corresponding to this region. May return None if no patch has to be created.'''
        raise NotImplementedError("Method not implemented")

    def verify(self):
        '''Self-verification routine for purposes of testing. Raises a VennRegionException if some inconsistencies of internal representation
        are discovered.'''
        raise NotImplementedError("Method not implemented")
    
class VennEmptyRegion(VennRegion):
    '''
    An empty region. To save some memory, returns [self, self] on the subtract_and_intersect_circle operation.
    It is possible to create an empty region with a non-None label position, by providing it in the constructor.
    
    >>> v = VennEmptyRegion()
    >>> [a, b] = v.subtract_and_intersect_circle((1,2), 3)
    >>> assert a == v and b == v
    >>> assert v.label_position() is None
    >>> assert v.size() == 0
    >>> assert v.make_patch() is None
    >>> assert v.is_empty()
    >>> v = VennEmptyRegion((0, 0))
    >>> v.label_position().tolist()
    [0.0, 0.0]
    '''
    def __init__(self, label_pos = None):
        self.label_pos = None if label_pos is None else np.asarray(label_pos, float)
    def subtract_and_intersect_circle(self, center, radius):
        return [self, self]
    def size(self):
        return 0
    def label_position(self):
        return self.label_pos
    def make_patch(self):
        return None
    def is_empty(self):  # We use this in tests as an equivalent of isinstance(VennEmptyRegion)
        return True
    def verify(self):
        pass

class VennCircleRegion(VennRegion):
    '''
    A circle-shaped region.
    
    >>> vcr = VennCircleRegion((0, 0), 1)
    >>> vcr.size()
    3.1415...
    >>> vcr.label_position().tolist()
    [0.0, 0.0]
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
            elif np.all(abs(intersections[0] - intersections[1]) < tol) and self.radius < radius:
                # There is a single intersection point (i.e. we are touching the circle),
                # the circle to be subtracted is not outside of us (this was checked before), and is larger than us.
                # This is a particular corner case that is not dealt with correctly by the general-purpose code below and must
                # be handled separately
                return [VennEmptyRegion(), self]
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
        
        >>> VennCircleRegion((0, 0), 1).label_position().tolist()
        [0.0, 0.0]
        >>> VennCircleRegion((-1.2, 3.4), 1).label_position().tolist()
        [-1.2, 3.4]
        '''
        return self.center
    
    def make_patch(self):
        '''
        Returns the corresponding circular patch.
        
        >>> patch = VennCircleRegion((1, 2), 3).make_patch()
        >>> patch
        <matplotlib.patches.Circle object at ...>
        >>> patch.center.tolist(), patch.radius
        ([1.0, 2.0], 3.0)
        '''
        return Circle(self.center, self.radius)

    def verify(self):
        pass
    

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
        
        TRIG_TOL = 100*tol  # We need to use looser tolerance level here because conversion to angles and back is prone to large errors.
        # Verify connectedness of arcs
        for i in range(len(self.arcs)):
            if not np.all(self.arcs[i-1].end_point() - self.arcs[i].start_point() < TRIG_TOL):
                raise VennRegionException("Arcs of an poly-arc-gon must be connected via endpoints")
        
        # Verify that arcs do not cross-intersect except at endpoints
        for i in range(len(self.arcs)-1):
            for j in range(i+1, len(self.arcs)):
                ips = self.arcs[i].intersect_arc(self.arcs[j])
                for ip in ips:
                    if not (np.all(abs(ip - self.arcs[i].start_point()) < TRIG_TOL) or np.all(abs(ip - self.arcs[i].end_point()) < TRIG_TOL)):
                        raise VennRegionException("Arcs of a poly-arc-gon may only intersect at endpoints")
                
                if len(ips) != 0 and (i - j) % len(self.arcs) > 1 and (j - i) % len(self.arcs) > 1:
                    # Two non-consecutive arcs intersect. This is in general not good, but
                    # may occasionally happen when all arcs inbetween have length 0.
                    pass # raise VennRegionException("Non-consecutive arcs of a poly-arc-gon may not intersect")
        
        # Verify that vertices are ordered so that at each point the direction along the polyarc changes towards the left.
        # Note that this test only makes sense for polyarcs obtained using circle intersections & subtractions.
        # A "flower-like" polyarc may have its vertices ordered counter-clockwise yet the direction would turn to the right at each of them.
        for i in range(len(self.arcs)):
            prev_arc = self.arcs[i-1]
            cur_arc = self.arcs[i]
            if box_product(prev_arc.direction_vector(prev_arc.to_angle), cur_arc.direction_vector(cur_arc.from_angle)) < -tol:
                raise VennRegionException("Arcs must be ordered so that the direction at each vertex changes counter-clockwise")
        
    def subtract_and_intersect_circle(self, center, radius):
        '''
        Circle subtraction / intersection only supported by 2-gon regions, otherwise a VennRegionException is thrown.
        In addition, such an exception will be thrown if the circle to be subtracted is completely within the region and forms a "hole".
        
        The result may be either a VennArcgonRegion or a VennMultipieceRegion (the latter happens when the circle "splits" a crescent in two).
        '''
        if len(self.arcs) != 2:
            raise VennRegionException("Circle subtraction and intersection with poly-arc regions is currently only supported for 2-arc-gons.")
        
        # In the following we consider the 2-arc-gon case.
        # Before we do anything, we check for a special case, where the circle of interest is one of the two circles forming the arcs.
        # In this case we can determine the answer quite easily.
        matching_arcs = [a for a in self.arcs if a.lies_on_circle(center, radius)]
        if len(matching_arcs) != 0:
            # If the circle matches a positive arc, the result is [empty, self], otherwise [self, empty]
            return [VennEmptyRegion(), self] if matching_arcs[0].direction else [self, VennEmptyRegion()]
            
        # Consider the intersection points of the circle with the arcs.
        # If any of the intersection points corresponds exactly to any of the arc's endpoints, we will end up with
        # a lot of messy special cases (as if the usual situation is not messy enough, eh).
        # To avoid that, we cheat by slightly increasing the circle's radius until this is not the case any more.
        center = np.asarray(center)
        illegal_intersections = [a.start_point() for a in self.arcs]
        while True:
            valid = True
            intersections = [a.intersect_circle(center, radius) for a in self.arcs]
            for ints in intersections:
                for pt in ints:
                    for illegal_pt in illegal_intersections:
                        if np.all(abs(pt - illegal_pt) < tol):
                            valid = False
            if valid:
                break
            else:
                radius += tol
                

        # There must be an even number of those points in total.
        # (If this is not the case, then we have an unfortunate case with weird numeric errors [TODO: find examples and deal with it?]).
        # There are three possibilities with the following subcases:
        #   I. No intersection points
        #       a) The polyarc is completely within the circle.
        #           result = [ empty, self ]
        #       b) The polyarc is completely outside the circle.
        #           result = [ self, empty ]
        #   II. Four intersection points, two for each arc. Points x1, x2 for arc X and y1, y2 for arc Y, ordered along the arc.
        #       a) The polyarc endpoints are both outside the circle.
        #           result_subtraction = a combination of two 3-arc polyarcs:
        #               1: {X - start to x1,
        #                   x1 to y2 along circle (negative direction)),
        #                   Y - y2 to end}
        #               2: {Y start to y1,
        #                   y1 to x2 along circle (negative direction)),
        #                   X - x2 to end}
        #       b) The polyarc endpoints are both inside the circle
        #               same as above, but the "along circle" arc directions are flipped and subtraction/intersection parts are exchanged
        #   III. Two intersection points
        #       a) One arc, X, has two intersection points i & j, another arc, Y, has no intersection points
        #           a.1) Polyarc endpoints are outside the circle
        #               result_subtraction = {X from start to i, circle i to j (direction = negative), X j to end, Y}
        #               result_intersection = {X i to j, circle j to i (direction = positive}
        #           a.2) Polyarc endpoints are inside the circle
        #               result_subtraction = {X i to j, circle j to i negative}
        #               result_intersection = {X 0 to i, circle i to j positive, X j to end, Y}
        #       b) Both arcs, X and Y, have one intersection point each. In this case one of the arc endpoints must be inside circle, another outside.
        #          call the arc that starts with the outside point X, the other arc Y.
        #           result_subtraction = {X start to intersection, intersection to intersection along circle (negative direction), Y from intersection to end}
        #           result_intersection = {X intersection to end, Y start to intersecton, intersection to intersecion along circle (positive)}
        center = np.asarray(center)
        intersections = [a.intersect_circle(center, radius) for a in self.arcs]
        
        if len(intersections[0]) == 0 and len(intersections[1]) == 0:
            # Case I
            if point_in_circle(self.arcs[0].start_point(), center, radius):
                # Case I.a)
                return [VennEmptyRegion(), self]
            else:
                # Case I.b)
                return [self, VennEmptyRegion()]
        elif len(intersections[0]) == 2 and len(intersections[1]) == 2:
            # Case II. a) or b)
            case_II_a = not point_in_circle(self.arcs[0].start_point(), center, radius)
            
            a1 = self.arcs[0].subarc_between_points(None, intersections[0][0])
            a2 = Arc(center, radius,
                     vector_angle_in_degrees(intersections[0][0] - center),
                     vector_angle_in_degrees(intersections[1][1] - center),
                     not case_II_a)
            a2.fix_360_to_0()
            a3 = self.arcs[1].subarc_between_points(intersections[1][1], None)
            piece1 = VennArcgonRegion([a1, a2, a3])
            
            b1 = self.arcs[1].subarc_between_points(None, intersections[1][0])
            b2 = Arc(center, radius,
                     vector_angle_in_degrees(intersections[1][0] - center),
                     vector_angle_in_degrees(intersections[0][1] - center),
                     not case_II_a)
            b2.fix_360_to_0()
            b3 = self.arcs[0].subarc_between_points(intersections[0][1], None)
            piece2 = VennArcgonRegion([b1, b2, b3])
            
            subtraction = VennMultipieceRegion([piece1, piece2])
            
            c1 = self.arcs[0].subarc(a1.to_angle, b3.from_angle)
            c2 = b2.reversed()
            c3 = self.arcs[1].subarc(b1.to_angle, a3.from_angle)
            c4 = a2.reversed()
            intersection = VennArcgonRegion([c1, c2, c3, c4])
            
            return [subtraction, intersection] if case_II_a else [intersection, subtraction]
        else:
            # Case III. Yuck.
            if len(intersections[0]) == 0 or len(intersections[1]) == 0:
                # Case III.a)
                x = 0 if len(intersections[0]) != 0 else 1
                y = 1 - x
                if len(intersections[x]) != 2:
                    warnings.warn("Numeric precision error during polyarc intersection, case IIIa. Expect wrong results.")
                    intersections[x] = [intersections[x][0], intersections[x][0]]  # This way we'll at least produce some result, although it will probably be wrong
                if not point_in_circle(self.arcs[0].start_point(), center, radius):
                    # Case III.a.1)
                    #   result_subtraction = {X from start to i, circle i to j (direction = negative), X j to end, Y}
                    a1 = self.arcs[x].subarc_between_points(None, intersections[x][0])
                    a2 = Arc(center, radius,
                             vector_angle_in_degrees(intersections[x][0] - center),
                             vector_angle_in_degrees(intersections[x][1] - center),
                             False)
                    a3 = self.arcs[x].subarc_between_points(intersections[x][1], None)
                    a4 = self.arcs[y]
                    subtraction = VennArcgonRegion([a1, a2, a3, a4])
                    
                    #   result_intersection = {X i to j, circle j to i (direction = positive)}
                    b1 = self.arcs[x].subarc(a1.to_angle, a3.from_angle)
                    b2 = a2.reversed()
                    intersection = VennArcgonRegion([b1, b2])
                    
                    return [subtraction, intersection]
                else:
                    # Case III.a.2)
                    #   result_subtraction = {X i to j, circle j to i negative}
                    a1 = self.arcs[x].subarc_between_points(intersections[x][0], intersections[x][1])
                    a2 = Arc(center, radius,
                             vector_angle_in_degrees(intersections[x][1] - center),
                             vector_angle_in_degrees(intersections[x][0] - center),
                             False)
                    subtraction = VennArcgonRegion([a1, a2])
                    
                    #   result_intersection = {X 0 to i, circle i to j positive, X j to end, Y}
                    b1 = self.arcs[x].subarc(None, a1.from_angle)
                    b2 = a2.reversed()
                    b3 = self.arcs[x].subarc(a1.to_angle, None)
                    b4 = self.arcs[y]
                    intersection = VennArcgonRegion([b1, b2, b3, b4])
                    
                    return [subtraction, intersection]
            else:
                # Case III.b)
                if len(intersections[0]) == 2 or len(intersections[1]) == 2:
                    warnings.warn("Numeric precision error during polyarc intersection, case IIIb. Expect wrong results.")
                
                # One of the arcs must start outside the circle, call it x
                x = 0 if not point_in_circle(self.arcs[0].start_point(), center, radius) else 1
                y = 1 - x
                
                a1 = self.arcs[x].subarc_between_points(None, intersections[x][0])
                a2 = Arc(center, radius,
                         vector_angle_in_degrees(intersections[x][0] - center),
                         vector_angle_in_degrees(intersections[y][0] - center), False)
                a3 = self.arcs[y].subarc_between_points(intersections[y][0], None)
                subtraction = VennArcgonRegion([a1, a2, a3])
                
                b1 = self.arcs[x].subarc(a1.to_angle, None)
                b2 = self.arcs[y].subarc(None, a3.from_angle)
                b3 = a2.reversed()
                intersection = VennArcgonRegion([b1, b2, b3])
                return [subtraction, intersection]
    
    def label_position(self):
        # Position the label right inbetween the midpoints of the arcs
        midpoints = [a.mid_point() for a in self.arcs]
        # For two-arc regions take the usual average
        # For more than two arcs, use arc lengths as the weights.
        if len(self.arcs) == 2:
            return np.mean(midpoints, 0)
        else:
            lengths = [a.length_degrees() for a in self.arcs]
            avg = np.sum([mp * l for (mp, l) in zip(midpoints, lengths)], 0)
            return avg / np.sum(lengths)
    
    def size(self):
        '''Return the area of the patch.
        
        The area can be computed using the standard polygon area formula + signed segment areas of each arc.
        '''
        polygon_area = 0
        for a in self.arcs:
            polygon_area += box_product(a.start_point(), a.end_point())
        polygon_area /= 2.0
        return polygon_area + sum([a.sign * a.segment_area() for a in self.arcs])
    
    def make_patch(self):
        '''
        Retuns a matplotlib PathPatch representing the current region.
        '''
        path = [self.arcs[0].start_point()]
        for a in self.arcs:
            if a.direction:
                vertices = Path.arc(a.from_angle, a.to_angle).vertices
            else:
                vertices = Path.arc(a.to_angle, a.from_angle).vertices
                vertices = vertices[np.arange(len(vertices) - 1, -1, -1)]
            vertices = vertices * a.radius + a.center
            path = path + list(vertices[1:])
        codes = [1] + [4] * (len(path) - 1)  # NB: We could also add a CLOSEPOLY code (and a random vertex) to the end
        return PathPatch(Path(path, codes))


class VennMultipieceRegion(VennRegion):
    '''
    A region containing several pieces.
    In principle, any number of pieces is supported,
    although no more than 2 should ever be needed in a 3-circle Venn diagram.
    Although subtraction/intersection are straightforward to implement we do
    not need those for matplotlib-venn, we raise exceptions in those methods.
    '''
    
    def __init__(self, pieces):
        '''
        Create a multi-piece region from a list of VennRegion objects.
        The list may be empty or contain a single item (although those regions can be converted to a
        VennEmptyRegion or a single region of the necessary type.
        '''
        self.pieces = pieces
        
    def label_position(self):
        '''
        Find the largest region and position the label in that.
        '''
        reg_sizes = [(r.size(), r) for r in self.pieces]
        reg_sizes.sort()
        return reg_sizes[-1][1].label_position()
    
    def size(self):
        return sum([p.size() for p in self.pieces])
    
    def make_patch(self):
        '''Currently only works if all the pieces are Arcgons.
           In this case returns a multiple-piece path. Otherwise throws an exception.'''
        paths = [p.make_patch().get_path() for p in self.pieces]
        vertices = np.concatenate([p.vertices for p in paths])
        codes = np.concatenate([p.codes for p in paths])
        return PathPatch(Path(vertices, codes))
    
    def verify(self):
        for p in self.pieces:
            p.verify()
        
    

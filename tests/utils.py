'''
Venn diagram plotting routines.
Utility functions used in tests.

Copyright 2014, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''
import json
import sys
from matplotlib.patches import Circle
from matplotlib.pyplot import scatter

def point_in_patch(patch, point):
    '''
    Given a patch, which is either a CirclePatch, a PathPatch or None, 
    returns true if the patch is not None and the point is inside it.
    '''
    if patch is None:
        return False
    elif isinstance(patch, Circle):
        c = patch.center
        return (c[0] - point[0])**2 + (c[1] - point[1])**2 <= patch.radius**2
    else:
        return patch.get_path().contains_point(point)


def verify_diagram(diagram, test_points):
    '''
    Given an object returned from venn2/venn3 verifies that the regions of the diagram contain the given points.
    Parameters:
       diagram: a VennDiagram object
       test_points: a dict, mapping region ids to lists of points that must be located in that region.
                    if some region is mapped to None rather than a list, the region must not be present in the diagram.
                    Region '' lists points that must not be present in any other region.
    '''
    for region in test_points.keys():
        points = test_points[region]
        if points is None:
            assert diagram.get_patch_by_id(region) is None
        else:
            if (region != ''):
                assert diagram.get_patch_by_id(region) is not None
            for pt in points:
                scatter(pt[0], pt[1])
                for test_region in test_points.keys(): # Test that the point is in its own region and no one else's
                    if (test_region != ''):
                        assert point_in_patch(diagram.get_patch_by_id(test_region), pt) == (region == test_region), \
                               "Point %s should %s in region %s" % (pt, "be" if (region == test_region) else "not be", test_region)


def exec_ipynb(filename):
    '''Executes all cells in a given ipython notebook consequentially.'''
    s = json.load(open(filename))
    for ws in s['worksheets']:
        for cell in ws['cells']:
            if cell['cell_type'] == 'code':
                code = ''.join(cell['input'])
                if sys.version_info.major == 2:
                    exec("exec code in locals()")
                else:
                    exec(code, locals())

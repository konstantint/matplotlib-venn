"""
Venn diagram plotting routines.
Utility functions used in tests.

Copyright 2014-2024, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
"""

from typing import Dict, Sequence, Union
import json
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Circle
from matplotlib.pyplot import scatter

from matplotlib_venn._common import VennDiagram
from matplotlib_venn._math import Point2DInternal


def point_in_patch(patch: Union[PathPatch, Circle], point: np.ndarray):
    """
    Given a patch, which is either a CirclePatch, a PathPatch or None,
    returns true if the patch is not None and the point is inside it.
    """
    if patch is None:
        return False
    elif isinstance(patch, Circle):
        c = patch.center
        return (c[0] - point[0]) ** 2 + (c[1] - point[1]) ** 2 <= patch.radius**2
    else:
        return patch.get_path().contains_point(point)


def verify_diagram(
    diagram: VennDiagram, test_points: Dict[str, Sequence[Point2DInternal]]
) -> None:
    """
    Given an object returned from venn2/venn3 verifies that the regions of the diagram contain the given points.
    In addition, makes sure that the diagram labels are within the corresponding regions (for all regions that are claimed to exist).
    Parameters:
       diagram: a VennDiagram object
       test_points: a dict, mapping region ids to lists of points that must be located in that region.
                    if some region is mapped to None rather than a list, the region must not be present in the diagram.
                    Region '' lists points that must not be present in any other region.
                    All keys of this dictionary not mapped to None (except key '') correspond to regions that must exist in the diagram.
                    For those regions we check that the region's label is positioned inside the region.
    """
    for region in test_points.keys():
        points = test_points[region]
        if points is None:
            assert diagram.get_patch_by_id(region) is None, (
                "Region %s must be None" % region
            )
        else:
            if region != "":
                assert diagram.get_patch_by_id(region) is not None, (
                    "Region %s must exist" % region
                )
                assert point_in_patch(
                    diagram.get_patch_by_id(region),
                    diagram.get_label_by_id(region).get_position(),
                ), (
                    "Label for region %s must be within this region" % region
                )
            for pt in points:
                scatter(pt[0], pt[1])
                for (
                    test_region
                ) in (
                    test_points.keys()
                ):  # Test that the point is in its own region and no one else's
                    if test_region != "":
                        assert point_in_patch(
                            diagram.get_patch_by_id(test_region), pt
                        ) == (
                            region == test_region
                        ), "Point %s should %s in region %s" % (
                            pt,
                            "be" if (region == test_region) else "not be",
                            test_region,
                        )


def exec_ipynb(filename: str) -> None:
    """Executes all cells in a given ipython notebook consequentially."""
    s = json.load(open(filename))
    locals_dict = locals()
    for cell in s["cells"]:
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            exec(code, locals_dict)

            # Explicitly close any figures created by this cell, which
            # would normally (in a notebook) be done by the
            # matplotlib-inline backend. This prevents a warning about "too
            # many figures opened" from Matplotlib.
            plt.close("all")

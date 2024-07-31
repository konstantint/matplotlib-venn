"""
Cost optimization-based layout algorithm for 3-way Venn diagrams. 

Unlike the rest of the code in the package, this implementation depends on the shapely library.
To include the dependency, the library should be installed as

```
pip install 'matplotlib-venn[shapely]'
```

(Shapely will quite probably become a core dependency in a future version).

Usage
-----

This layout algorithm makes most sense in the cases when the default, "pairwise" layout
does not work well enough for your data (which is usually the case for very skewed subset sizes).

In this case just try doing:

>>> from matplotlib_venn.layout.venn3 import cost_based
>>> from matplotlib_venn import venn3
>>> subset_sizes = (100,200,10000,10,20,3,1)
>>> venn3(subset_sizes, layout_algorithm=cost_based.LayoutAlgorithm())
<matplotlib_venn...VennDiagram object at ...>

You may further tune the behaviour of the algorithm by redefining the cost function.
By default the algorithm tries to optimize the sum of |log(1+target_size)-log(1+actual_size)|
over all 7 regions. If for some reason you believe |target_size - actual_size| should work better
for your case, you can achieve it as follows:

>>> alg = cost_based.LayoutAlgorithm(cost_fn=cost_based.WeightedAggregateCost(transform_fn=lambda x: x))
>>> venn3(subset_sizes, layout_algorithm=alg)
<matplotlib_venn...VennDiagram object at ...>

Alternatively, you may want the optimization to give more weight to some of the regions or even ignore some of the
larger ones.

>>> alg = cost_based.LayoutAlgorithm(cost_fn=cost_based.WeightedAggregateCost(weights=(0,0,0,1,1,1,1)))
>>> venn3(subset_sizes, layout_algorithm=alg)
<matplotlib_venn...VennDiagram object at ...>

In theory, if the cost is defined as a difference in sizes of "pairwise" regions (AB, BC, AC), the result of optimizing
it should be equivalent to what the default ('pairwise') algorithm does. To play with this idea, the module defines the
respective `pairwise_cost` function. The result is not exactly the same as that of the default algorithm, but it would
nearly always succeed, even when the default algorithm sometimes fails. E.g.:

>>> subset_sizes = (1, 0, 0, 650, 0, 76, 13)
>>> # Fails
>>> venn3(subset_sizes)  # doctest: +IGNORE_EXCEPTION_DETAIL
Traceback (most recent call last):
matplotlib_venn._region.VennRegionException: Invalid configuration of circular regions (holes are not supported).

>>> # Succeeds, producing what the default algorithm should have produced
>>> venn3(subset_sizes, layout_algorithm=cost_based.LayoutAlgorithm(cost_fn=cost_based.pairwise_cost))
<matplotlib_venn...VennDiagram object at ...>

NB: This implementation is still in "alpha" stage, the code and behaviour may change in backwards-incompatible ways.

Copyright 2024, Konstantin Tretyakov.
http://kt.era.ee/

Based on a prototype by Paul Brodersen (https://github.com/konstantint/matplotlib-venn/issues/35).

Licensed under MIT license.
"""

from typing import Callable, Optional, Sequence
import warnings
import numpy as np
from shapely.geometry import Point
from scipy.optimize import minimize, NonlinearConstraint
from scipy.spatial.distance import pdist
from matplotlib_venn._math import NUMERIC_TOLERANCE
from matplotlib_venn.layout.venn3 import pairwise
from matplotlib_venn.layout.api import (
    LayoutException,
    Point2D,
    SubsetSizes,
    VennLayout,
    VennLayoutAlgorithm,
)


def _initialize_centers(radii: np.ndarray) -> np.ndarray:
    """Initialize centers on a small circle around (0, 0).

    The centers are positioned at 90+60, 90-60, -90, matching the
    positioning logic of the pairwise algorithm.

    >>> centers = _initialize_centers(np.asarray([1, 2, 3]))
    >>> np.allclose(centers, \
                    np.array([[-0.866, 0.5], \
                              [ 0.866, 0.5], \
                              [ 0, -1]]), atol=0.001)
    True
    """
    angles = 2 * np.pi * np.array([1 / 12 + 1 / 3, 1 / 12, 1 / 12 + 2 / 3])
    return np.vstack([np.cos(angles), np.sin(angles)]).T * np.min(radii)


def _normalize_subset_sizes(
    subset_sizes: SubsetSizes, normalize_to: float = 1.0
) -> SubsetSizes:
    """Normalize provided subset sizes to areas with total area equal to <normalize_to>.
    If the total area is less than _minimal_area falls back to a normalized version of the
    'unweighted' set of areas (1,1,1,1,1,1,1).

    >>> _normalize_subset_sizes((1,0,0,0,0,0,0))
    array([1., 0., 0., 0., 0., 0., 0.])
    >>> _normalize_subset_sizes((1,1,0,0,0,0,0))
    array([0.5, 0.5, 0. , 0. , 0. , 0. , 0. ])
    >>> _normalize_subset_sizes((0,0,0,0,0,0,0))
    array([0.14..., 0.14..., 0.14..., 0.14..., 0.14..., 0.14..., 0.14...])
    """
    areas = np.array(np.abs(subset_sizes), float)
    total_area = np.sum(areas)
    if np.abs(total_area) < NUMERIC_TOLERANCE:
        warnings.warn(
            "All regions have zero area. Falling back to an unweighted diagram."
        )
        return _normalize_subset_sizes((1, 1, 1, 1, 1, 1, 1), normalize_to)
    else:
        return areas / total_area * normalize_to


def _compute_radii(areas: SubsetSizes) -> np.ndarray:
    """Compute radii of the three circles based on a given SubsetSizes vector.

    Returns an array of three radii for the three circles.

    >>> regions = np.pi*np.array([1, 2, 0, 3, 0, 0, 0])**2
    >>> _compute_radii(regions)
    array([1., 2., 3.])
    >>> deltas = np.array([-0.3, -0.3, 0.1, -0.3, 0.1, 0.1, 0.1])
    >>> _compute_radii(regions + deltas)
    array([1., 2., 3.])
    """
    (Abc, aBc, ABc, abC, AbC, aBC, ABC) = areas
    A = Abc + ABc + AbC + ABC
    B = aBc + ABc + aBC + ABC
    C = abC + AbC + aBC + ABC
    return np.sqrt(np.array([A, B, C]) / np.pi)


def _compute_subset_areas(centers: np.ndarray, radii: np.ndarray) -> SubsetSizes:
    """Given centers and radii of a venn3 diagram, return the respective subset areas.

    >>> areas = _compute_subset_areas(np.asarray([[0,0], [2,2], [4,4]]), np.asarray([1, 1, 1])) 
    >>> np.allclose(areas, [np.pi, np.pi, 0, np.pi, 0, 0, 0], atol=0.01)
    True
    >>> from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm
    >>> layout = DefaultLayoutAlgorithm()((1,1,1,1,1,1,1))
    >>> areas = _compute_subset_areas(\
                np.asarray([c.asarray() for c in layout.centers]),\
                np.asarray(layout.radii))
    >>> layout = DefaultLayoutAlgorithm()(areas)
    >>> new_areas = _compute_subset_areas(\
                np.asarray([c.asarray() for c in layout.centers]),\
                np.asarray(layout.radii))
    >>> np.allclose(areas, new_areas, atol=0.01)
    True
    """
    a, b, c = [Point(*center).buffer(radius) for center, radius in zip(centers, radii)]
    regions = [
        a.difference(b).difference(c),  # Abc
        b.difference(a).difference(c),  # aBc
        a.intersection(b).difference(c),  # ABc
        c.difference(a).difference(b),  # abC
        a.intersection(c).difference(b),  # AbC
        b.intersection(c).difference(a),  # aBC
        a.intersection(b).intersection(c),  # ABC
    ]
    return np.array([region.area for region in regions])


# A cost function is a callable that accepts the desired subset sizes
# and the actual sizes (for the current layout) and returns the cost ("loss")
# of the discrepancy.
CostFunction = Callable[[SubsetSizes, SubsetSizes], float]


class WeightedAggregateCost(CostFunction):
    """A cost function that aggregates differences over all regions.

    The function computes:
      np.dot(weights, np.abs(fn(target_size) - fn(current_size))**power).

    >>> fn = WeightedAggregateCost()
    >>> fn([1]*7, [1]*7)
    0.0
    >>> fn([1]*7, [0]*7)
    7.0
    >>> fn = WeightedAggregateCost(lambda x: x**2)
    >>> fn([1]*7, [1]*7)
    0.0
    >>> fn([2]*7, [0]*7)
    28.0
    >>> fn = WeightedAggregateCost(weights=(1,2,3))
    >>> fn([1,2,0], [0,0,3])
    14.0
    >>> fn = WeightedAggregateCost(weights=(0,0,1), power=2)
    >>> fn([1,2,0], [0,0,3])
    9.0
    """

    def __init__(
        self,
        transform_fn: Callable[[np.ndarray], np.ndarray] = lambda x: x,
        weights: Sequence[float] = (1, 1, 1, 1, 1, 1, 1),
        power: float = 1,
    ):
        self.transform_fn = transform_fn
        self.weights = np.asarray(weights)
        self.power = power

    def __call__(self, target_areas: SubsetSizes, current_areas: SubsetSizes) -> float:
        targets = self.transform_fn(np.asarray(target_areas))
        current = self.transform_fn(np.asarray(current_areas))
        return float(np.dot(self.weights, np.abs(targets - current) ** self.power))


def pairwise_cost(target_areas: SubsetSizes, actual_areas: SubsetSizes) -> float:
    """The cost, computed as the absolute difference between pairwise (A&B, B&C, A&C) areas.

    This matches the logic of the default ("pairwise") layout algorithm and thus
    produces mostly the same results (not exactly the same due to some randomness
    involved in the iterative nature of the optimization).

    It is here primarily for experimentation and "completeness' sake".

    >>> pairwise_cost([1]*7, [1]*7)
    0.0
    >>> pairwise_cost([1]*7, (2,2,1,2,1,1,1))
    0.0
    >>> pairwise_cost([1]*7, (2,2,1,2,1,1,1.5))
    1.5
    >>> pairwise_cost([1]*7, (2,2,1.5,2,1,1,1))
    0.5
    >>> pairwise_cost([1]*7, (2,2,1.5,2,1.1,1,1))
    0.6...
    """
    (tAbc, taBc, tABc, tabC, tAbC, taBC, tABC) = target_areas
    (Abc, aBc, ABc, abC, AbC, aBC, ABC) = actual_areas
    dAB = tABC + tABc - (ABC + ABc)
    dBC = tABC + taBC - (ABC + aBC)
    dAC = tABC + tAbC - (ABC + AbC)
    return float(abs(dAB) + abs(dBC) + abs(dAC))


class LayoutAlgorithm(VennLayoutAlgorithm):
    """3-way Venn layout that positions circles by numerically optimizing a given discrepancy cost.

    >>> alg = LayoutAlgorithm()
    >>> layout = alg((1,1,1,1,1,1,1), ("A", "B", "C"))
    >>> layout.centers
    [Point2D(-0.13..., 0.07...), Point2D(0.13..., 0.077...), Point2D(-1..., -0.15...)]
    >>> layout.radii
    [0.42..., 0.42..., 0.42...]
    """

    def __init__(self, cost_fn: Optional[CostFunction] = None, fallback: bool = True):
        """Initialize the cost-based layout algorithm.

        Args:
            cost_fn: A cost function to be optimized.
                     Default is WeightedAggregateCost(lambda x: np.log(1 + x)).
                     This has been determined to work well enough in practice.
            fallback: Whether to fall back to the default ("pairwise") layout
                     algorithm if optimization does not converge. True by default.
                     If there is no fallback, a LayoutException will be raised if
                     optimization fails.
        """
        self._cost_fn = cost_fn or WeightedAggregateCost(lambda x: np.log(1 + x))
        self._fallback = fallback
        # This is a convenience field that will carry the result of the most
        # recent "minimize" call.
        self.last_optimization_result = None

    def __call__(
        self,
        subsets: SubsetSizes,
        set_labels: Optional[Sequence[str]] = None,
    ) -> VennLayout:
        target_areas = _normalize_subset_sizes(subsets)
        radii = _compute_radii(target_areas)
        centers = _initialize_centers(radii)

        # We will position the circles by optimizing this cost function ...
        def _cost_function(centers_flattened: np.ndarray) -> np.ndarray:
            """Computes the cost of positioning circles at given centers."""
            current_areas = _compute_subset_areas(
                centers_flattened.reshape(-1, 2), radii
            )
            return self._cost_fn(target_areas, current_areas)

        # ... while making sure the pairwise distances between circles do not exceed sum of radii:
        # (this is the order in which pdist computes pairwise distances).
        upper_bounds = np.array(
            [radii[0] + radii[1], radii[0] + radii[2], radii[1] + radii[2]]
        )

        # ... and are not below differences between radii:
        lower_bounds = np.abs(
            np.array([radii[0] - radii[1], radii[0] - radii[2], radii[1] - radii[2]])
        )

        def _pairwise_distances(centers_flattened: np.ndarray) -> np.ndarray:
            return pdist(np.reshape(centers_flattened, (-1, 2)))

        result = minimize(
            _cost_function,
            centers.flatten(),
            method="SLSQP",
            constraints=[
                NonlinearConstraint(
                    _pairwise_distances, ub=upper_bounds, lb=lower_bounds
                )
            ],
        )
        self.last_optimization_result = result

        if not result.success:
            warnings.warn("Optimization failed: {0}".format(result.message))
            if self._fallback:
                # Fall back to _pairwise
                return pairwise.LayoutAlgorithm()(subsets, set_labels)
            else:
                raise LayoutException("Optimization failed: {0}".format(result.message))

        centers = result.x.reshape((-1, 2))
        result = VennLayout(
            centers=[Point2D(*center) for center in centers],
            radii=list(map(float, radii)),
        )
        # TODO: We reuse the pairwise algorithm implementation for set label positioning.
        # It does not always do the most correct job.
        pairwise._compute_set_labels_positions(result)
        return result

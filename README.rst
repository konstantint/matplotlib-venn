====================================================
Venn diagram plotting routines for Python/Matplotlib
====================================================

.. image::  https://travis-ci.org/konstantint/matplotlib-venn.png?branch=master
   :target: https://travis-ci.org/konstantint/matplotlib-venn

Routines for plotting area-weighted two- and three-circle venn diagrams.

Installation
------------

Install the package as usual via ``pip``::

    $ python -m pip install matplotlib-venn

Since version 1.1.0 the package includes an extra "cost based" layout algorithm for `venn3` diagrams,
that relies on the `shapely` package, which is not installed as a default dependency. If you need the
new algorithm (or just have nothing against installing `shapely` along the way), instead do::

    $ python -m pip install "matplotlib-venn[shapely]"

It is quite probable that `shapely` will become a required dependency eventually in one of the future versions.

Dependencies
------------

- ``numpy``,
- ``scipy``,
- ``matplotlib``,
- ``shapely`` (optional).

Usage
-----
The package provides four main functions: ``venn2``,
``venn2_circles``, ``venn3`` and ``venn3_circles``.

The functions ``venn2`` and ``venn2_circles`` accept as their only
required argument a 3-element tuple ``(Ab, aB, AB)`` of subset sizes,
and draw a two-circle venn diagram with respective region areas, e.g.::

    venn2(subsets = (3, 2, 1))

In this example, the region, corresponding to subset ``A and not B`` will
be three times larger in area than the region, corresponding to subset ``A and B``.

You can also provide a tuple of two ``set`` or ``Counter`` (i.e. multi-set)
objects instead (new in version 0.7), e.g.::

    venn2((set(['A', 'B', 'C', 'D']), set(['D', 'E', 'F'])))

Similarly, the functions ``venn3`` and ``venn3_circles`` take a
7-element tuple of subset sizes ``(Abc, aBc, ABc, abC, AbC, aBC,
ABC)``, and draw a three-circle area-weighted Venn
diagram: 

.. image:: https://user-images.githubusercontent.com/13646666/87874366-96924800-c9c9-11ea-8b06-ac1336506b59.png

Alternatively, a tuple of three ``set`` or ``Counter`` objects may be provided.

The functions ``venn2`` and ``venn3`` draw the diagrams as a collection of colored
patches, annotated with text labels. The functions ``venn2_circles`` and
``venn3_circles`` draw just the circles.

The functions ``venn2_circles`` and ``venn3_circles`` return the list of ``matplotlib.patch.Circle`` objects that may be tuned further
to your liking. The functions ``venn2`` and ``venn3`` return an object of class ``VennDiagram``,
which gives access to constituent patches, text elements, and (since
version 0.7) the information about the centers and radii of the
circles.

Basic Example::

    from matplotlib_venn import venn2
    venn2(subsets = (3, 2, 1))

For the three-circle case::

    from matplotlib_venn import venn3
    venn3(subsets = (1, 1, 1, 2, 1, 2, 2), set_labels = ('Set1', 'Set2', 'Set3'))

A more elaborate example::

    from matplotlib import pyplot as plt
    import numpy as np
    from matplotlib_venn import venn3, venn3_circles
    plt.figure(figsize=(4,4))
    v = venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels = ('A', 'B', 'C'))
    v.get_patch_by_id('100').set_alpha(1.0)
    v.get_patch_by_id('100').set_color('white')
    v.get_label_by_id('100').set_text('Unknown')
    v.get_label_by_id('A').set_text('Set "A"')
    c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
    c[0].set_lw(1.0)
    c[0].set_ls('dotted')
    plt.title("Sample Venn diagram")
    plt.annotate('Unknown set', xy=v.get_label_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
                 ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
    plt.show()

An example with multiple subplots::

    from matplotlib_venn import venn2, venn2_circles
    figure, axes = plt.subplots(2, 2)
    venn2(subsets={'10': 1, '01': 1, '11': 1}, set_labels = ('A', 'B'), ax=axes[0][0])
    venn2_circles((1, 2, 3), ax=axes[0][1])
    venn3(subsets=(1, 1, 1, 1, 1, 1, 1), set_labels = ('A', 'B', 'C'), ax=axes[1][0])
    venn3_circles({'001': 10, '100': 20, '010': 21, '110': 13, '011': 14}, ax=axes[1][1])
    plt.show()

Perhaps the most common use case is generating a Venn diagram given
three sets of objects::

    set1 = set(['A', 'B', 'C', 'D'])
    set2 = set(['B', 'C', 'D', 'E'])
    set3 = set(['C', 'D',' E', 'F', 'G'])

    venn3([set1, set2, set3], ('Set1', 'Set2', 'Set3'))
    plt.show()

Tuning the diagram layout
-------------------------

Note that for a three-circle venn diagram it is not in general
possible to achieve exact correspondence between the required set
sizes and region areas. The default layout algorithm aims to correctly represent:

  * Relative areas of the full individual sets (A, B, C).
  * Relative areas of pairwise intersections of sets (A&B, A&C, B&C, not to be confused with the regions
    A&B&~C, A&~B&C, ~A&B&C, on the diagram).

Sometimes the result is unsatisfactory and either the area weighting or the layout logic needs
to be tuned.

The area weighing can be adjusted by providing a `fixed_subset_sizes` argument to the `DefaultLayoutAlgorithm`::

    from matplotlib_venn.layout.venn2 import DefaultLayoutAlgorithm
    venn2((1,2,3), layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1)))

    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm
    venn3((7,6,5,4,3,2,1), layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1,1,1,1,1)))

In the above examples the diagram regions will be plotted as if `venn2((1,1,1))` and `venn3((1,1,1,1,1,1,1))` were
invoked, yet the actual numbers will be `(1,2,3)` and `(7,6,5,4,3,2,1)` respectively.

The diagram can be tuned further by switching the layout algorithm to a different implementation.
At the moment the package offers an alternative layout algorithm for `venn3` diagrams that lays the circles out by
optimizing a user-provided *cost function*. The following examples illustrate its usage::

    from matplotlib_venn.layout.venn3 import cost_based
    subset_sizes = (100,200,10000,10,20,3,1)
    venn3(subset_sizes, layout_algorithm=cost_based.LayoutAlgorithm())

    opts = cost_based.LayoutAlgorithmOptions(cost_fn=cost_based.WeightedAggregateCost(transform_fn=lambda x: x))
    venn3(subset_sizes, layout_algorithm=cost_based.LayoutAlgorithm(opts))

    opts = cost_based.LayoutAlgorithmOptions(cost_fn=cost_based.WeightedAggregateCost(weights=(0,0,0,1,1,1,1)))
    venn3(subset_sizes, layout_algorithm=cost_based.LayoutAlgorithm(opts))

The default "pairwise" algorithm is, theoretically, a special case of the cost-based method with the respective cost function::

    opts = cost_based.LayoutAlgorithmOptions(cost_fn=cost_based.pairwise_cost)
    venn3(subset_sizes, layout_algorithm=cost_based.LayoutAlgorithm(opts))

(The latter plot will be close, but not perfectly equal to the outcome of `DefaultLayoutAlgorithm()`).

Note that the import::

    from matplotlib_venn.layout.venn3 import cost_based

will fail unless you have the optional `shapely` package installed (see "Installation" above).


Questions
---------

* If you ask your questions at `StackOverflow <http://stackoverflow.com/>`_ and tag them 
  `matplotlib-venn <http://stackoverflow.com/questions/tagged/matplotlib-venn>`_, chances are high you could get
  an answer from the maintainer of this package.

See also
--------

* Report issues and submit fixes at Github:
  https://github.com/konstantint/matplotlib-venn
  
  Check out the ``DEVELOPER-README.rst`` for development-related notes.
* Some alternative means of plotting a Venn diagram (as of
  October 2012) are reviewed in the blog post:
  http://fouryears.eu/2012/10/13/venn-diagrams-in-python/
* The `matplotlib-subsets
  <https://pypi.python.org/pypi/matplotlib-subsets>`_ package
  visualizes a hierarchy of sets as a tree of rectangles.
* The `matplotlib_set_diagrams <https://pypi.org/project/matplotlib-set-diagrams>`_ package
  is a GPL-licensed alternative that offers a different layout algorithm, which supports more than
  three sets and provides a cool ability to incorporate wordclouds into your Venn (Euler) diagrams.


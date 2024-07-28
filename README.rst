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

Dependencies
------------

- ``numpy``,
- ``scipy``,
- ``matplotlib``.

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
ABC)``, and draw a three-circle area-weighted venn
diagram. Alternatively, a tuple of three ``set`` or ``Counter`` objects may be provided.

The functions ``venn2`` and ``venn3`` draw the diagrams as a collection of colored
patches, annotated with text labels. The functions ``venn2_circles`` and
``venn3_circles`` draw just the circles.

Sometimes the area weighing needs to be disabled or manually tuned to achieve a visually
better result. This can be achieved as follows::

    from matplotlib_venn.layout.venn2 import DefaultLayoutAlgorithm
    venn2((1,2,3), layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1)))

    from matplotlib_venn.layout.venn3 import DefaultLayoutAlgorithm
    venn3((7,6,5,4,3,2,1), layout_algorithm=DefaultLayoutAlgorithm(fixed_subset_sizes=(1,1,1,1,1,1,1)))

Note that for a three-circle venn diagram it is not in general
possible to achieve exact correspondence between the required set
sizes and region areas, however in most cases the picture will still
provide a useful representation.

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


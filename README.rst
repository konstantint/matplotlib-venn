Venn diagram plotting routines for Python/Matplotlib
====================================================

About
-----
This package contains routines for plotting area-weighted two- and three-circle venn diagrams.

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT.


Installation
------------
Installable as any other Python package, either through :code:`easy_install`, or through :code:`python setup.py install`, or by simply including the package to :code:`sys.path`.

Usage
-----
There are four main functions here: :code:`venn2`, :code:`venn2_circles`, :code:`venn3`, :code:`venn3_circles`.

The :code:`venn2` and :code:`venn2_circles`  accept as their only required argument a 3-element list of subset sizes:

    subsets = (Ab, aB, AB)

That is, for example, subsets[0] contains the size of the subset (A and not B), and
subsets[2] contains the size of the set (A and B), etc.

Similarly, the functions :code:`venn3` and :code:`venn3_circles` require a 7-element list:

    subsets = (Abc, aBc, ABc, abC, AbC, aBC, ABC)

The functions :code:`venn2_circles` and :code:`venn3_circles` simply draw two or three circles respectively,
such that their intersection areas correspond to the desired set intersection sizes. 
Note that for a three-circle venn diagram it is not possible to achieve exact correspondence, although in
most cases the picture will still provide a decent indication.

The functions :code:`venn2` and :code:`venn3` draw diagram as a collection of separate colored patches with text labels.

The functions :code:`venn2_circles` and :code:`venn3_circles` return the list of Circle patches that may be tuned further 
to your liking.

The functions :code:`venn2` and :code:`venn3` return an object of class :code:`Venn2` or :code:`Venn3` respectively,
which give access to constituent patches and text elements.

Basic Example::
    
    from matplotlib.venn improt venn2
    venn2(subsets = (3, 2, 1))
    
For the three-circle case::

    from matplotlib.venn import venn3
    venn3(subsets = (1, 1, 1, 2, 1, 2, 2), set_labels = ('Set1', 'Set2', 'Set3'))
    
More elaborate example::

    from matplotlib import pyplot as plt
    import numpy as np
    from matplotlib.venn import venn3, venn3_circles
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

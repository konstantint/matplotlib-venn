Venn diagram plotting routines for Python/Matplotlib
====================================================

About
-----
This package contains a rountine for plotting area-weighted three-circle venn diagrams.

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under BSD.


Installation
------------
Installable as any other Python package, either through :code:`easy_install`, or through :code:`python setup.py install`, or by simply including the package to :code:`sys.path`.

Usage
-----
There are two main functions in the package: :code:`venn3_circles` and :code:`venn3`

Both accept as their only required argument an 8-element list of set sizes,

:code:`sets = (abc, Abc, aBc, ABc, abC, AbC, aBC, ABC)`

That is, for example, :code:`sets[1]` contains the size of the set (A and not B and not C),
:code:`sets[3]` contains the size of the set (A and B and not C), etc. Note that the value in :code:`sets[0]` is not used.

The function :code:`venn3_circles` simply draws three circles such that their intersection areas would correspond
to the desired set intersection sizes. Note that it is not impossible to achieve exact correspondence, but in
most cases the picture will still provide a decent indication.

The function :code:`venn3` draws the venn diagram as a collection of 8 separate colored patches with text labels.

The function :code:`venn3_circles` returns the list of Circle patches that may be tuned further.
The function :code:`venn3` returns an object of class :code:`Venn3`, which also gives access to diagram patches and text elements.

Basic Example::
    
    from matplotlib.venn import venn3
    venn3(sets = (0, 1, 1, 1, 2, 1, 2, 2), set_labels = ('Set1', 'Set2', 'Set3'))
    
More elaborate example::

    from matplotlib import pyplot as plt
    import numpy as np
    from matplotlib.venn import venn3, venn3_circles
    plt.figure(figsize=(4,4))
    v = venn3(sets=(0, 1, 1, 1, 1, 1, 1, 1), set_labels = ('A', 'B', 'C'))
    v.get_patch_by_id('100').set_alpha(1.0)
    v.get_patch_by_id('100').set_color('white')
    v.get_text_by_id('100').set_text('Unknown')
    v.labels[0].set_text('Set "A"')
    c = venn3_circles(sets=(0, 1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
    c[0].set_lw(1.0)
    c[0].set_ls('dotted')
    plt.title("Sample Venn diagram")
    plt.annotate('Unknown set', xy=v.get_text_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-70,-70), 
                ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))


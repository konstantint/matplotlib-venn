==============================================================
Hierarchical subset diagram plotting for Python/Matplotlib
==============================================================

Routines for plotting area-weighted diagrams of subsets and subsets of subsets.
This consistutes an extension to Venn diagrams in some sense (hierarchy), while a limitation 
in another (subsets only).

Installation
------------

The simplest way to install the package is via ``easy_install`` or ``pip``::

    $ easy_install matplotlib-subsets

Dependencies
------------

- ``numpy``, ``scipy``, ``matplotlib``.

Usage
-----
The package provides the function: ``treesets_rectangles``.

It takes a tree, where each node is defined as ((number-of-items-contained, 
label, dictionary-of-plotting-attributes), [child nodes...]).

For example::

	tree = ((120, '120', None), [
		((50, 'A50', None), []),
		((50, 'B50', None), [])
		])
	
	treesets_rectangles(tree)
	plt.savefig('example_tree.pdf', bbox_inches='tight')
	plt.close()

Here, the node '120' is of size 120. It has two subsets, 'A50' and 'B50', each of size 50.
No additional plotting attributes are given (e.g. the color of rectangles is chosen automatically).

See also
--------

* Report issues and submit fixes at Github: https://github.com/JohannesBuchner/matplotlib-subsets
* Venn diagram package: https://github.com/konstantint/matplotlib-venn


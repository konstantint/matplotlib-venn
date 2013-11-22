'''
Venn diagram plotting routines.
Test module (meant to be used via py.test).

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
'''
from matplotlib_venn._vennnested import *

def test_nested_example1():
	sets = [
		set(list('ABCDEFGH')),
		set(list('DEFG')),
		set(list('E')),
	]
	setsizes = [len(s) for s in sets]
	nestedsets_rectangles(setsizes, labels = [
		r'$\mathbf{%s}$ ($%d$)' % (string.ascii_uppercase[i], len(s)) 
			for i, s in enumerate(sets)])
	plt.savefig('example_nested.pdf', bbox_inches='tight')
	plt.close()

def test_tree_example1():
	tree = ((120, '100', None), [
		((50, 'A50', None), []),
		((50, 'B50', None), [])
		])
	
	treesets_rectangles(tree)
	plt.savefig('example_tree.pdf', bbox_inches='tight')
	plt.close()

def test_tree_example2():
	tree = ((120, '100', None), 
		[((50, 'A50', None), 
			[((20, 'AX20', None), [((8, 'AXM8', None), [((4, 'AXM4', None), [((2, 'AXM2', None), [])])]), ((8, 'AXN8', None), [])]), 
			 ((20, 'AY20', None), [])]),
		((50, 'B50', None), [((5, 'Bt', None), [])]*5)
		])
	
	plt.figure(figsize=(7,7))
	treesets_rectangles(tree)
	plt.savefig('example_tree2.pdf', bbox_inches='tight')
	plt.close()


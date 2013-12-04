'''
Subset diagram plotting routines.

Copyright 2013, Johannes Buchner.

Licensed under MIT license.
'''
import numpy
import warnings
import string

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyBboxPatch
import itertools
import scipy.stats

default_attrs = [dict(ec=c, fc='None', alpha=0.7) for c in ['black', 'blue', 'red', 'green', 'magenta']]
default_attrs = itertools.cycle(default_attrs)

def plot_box(bbox, label='', attrs={}):
	((xmin, ymin), (xmax, ymax)) = bbox.get_points()
	if attrs is None:
		attrs = default_attrs.next()
		print 'picked attrs:', attrs

	ax = plt.gca()
	print 'plotting %s at' % label, ((xmin, ymin), (xmax, ymax))
	p_fancy = FancyBboxPatch((xmin, ymin),
		xmax - xmin, ymax - ymin,
		boxstyle="round,pad=0.0, rounding_size=0.02",
		**attrs)
	ax.add_patch(p_fancy)
	ax.text(xmin + 0.01, ymax - 0.01, label, 
		horizontalalignment='left',
		verticalalignment='top',
		)
	
def nodesets_rectangles((node, children), bbox):
	# plot what the children make out in the bbox
	nodesize, nodelabel, nodeattrs = node
	
	nchildren = len(children)
	
	print 'node', node
	print '   has %d children:' % nchildren
	for i, c in enumerate(children):
		print '     %i: %s' % (i, c)
	((xmin0, ymin0), (xmax0, ymax0)) = bbox.get_points()
	deltay = ymax0 - ymin0
	deltax = xmax0 - xmin0
	if nchildren > 0:
		nsizechildren = sum([size for (size, label, attrs), grandchildren in children])
		empty = 1 - nsizechildren * 1. / nodesize
		fbuffer = (empty)**0.5
		yratio = deltay * (nsizechildren * 1. / nodesize) / deltax
		#yratio = deltax / deltax * (nsizechildren * 1. / nodesize)
		
		#padding = arearatio**0.5 / 10
		#shorteningratio = 
		#fbuffer = 0
		print 'yratio %s:' % str(node), yratio
		print 'deltax, deltay:', deltax, deltay
	for child, grandchildren in children:
		((xmin, ymin), (xmax, ymax)) = bbox.get_points()
		
		size, label, attrs = child
		arearatio = size * 1. / nodesize
		print 'arearatio of child:', arearatio
		if arearatio == 0:
			# skip empty children
			continue
		
		if nchildren == 1:
			# update borders, proportional to area
			print 'single subset: arearatio', arearatio
			# along a beta distribution; cdf is area
			a, b = 10, 1
			rv = scipy.stats.beta(a, b)
			fx = rv.ppf(arearatio)
			fy = arearatio / fx
			print 'fx, fy:', fx, fy
			# add padding if possible
			ypad = min(fy*0.02, 1 - fy)
			xpad = min(fx*0.02, 1 - fx)
			
			ymax = ymax - deltay * (1 - (fy + ypad))
			xmin = xmin + deltax * (1 - (fx + xpad))
			ymin = ymin + deltay * ypad
			xmax = xmax - deltax * xpad
		else:
			# update borders, split dependent on xratio
			# we prefer to split in y (because label text flows in x)
			# but split if ratio is too extreme
			if yratio > 0.4: # split in y
				print 'splitting in y: starting box', ((xmin0, ymin0), (xmax0, ymax0))
				#ymin,  ymax  = ymax0 - deltay * arearatio + 0.*deltay * fbuffer / 2, ymax0
				#ymax0, ymin0 = ymax0 - deltay * arearatio - 0.*deltay * fbuffer / 2, ymin0
				ymin,  ymax  = ymin0, ymin0 + deltay * arearatio - 1.*deltay * fbuffer / 40
				ymax0, ymin0 = ymax0, ymin0 + deltay * arearatio + 1.*deltay * fbuffer / 40
				#ymax0 = ymax0 - deltay * arearatio - deltay * fbuffer / 2
				#ymin = ymax0 + deltay * fbuffer / 2
				#xmax, xmin = xmax0, xmin0
				print 'splitting in y: child box', ((xmin, ymin), (xmax, ymax))
				print 'splitting in y: remaining box', ((xmin0, ymin0), (xmax0, ymax0))
			else:
				print 'splitting in x: starting box', ((xmin0, ymin0), (xmax0, ymax0))
				#xmin,  xmax  = xmin0, xmin0 + deltay * arearatio - 1.*deltax * fbuffer / 80
				xmin,  xmax  = xmax0 - deltax * arearatio + deltax * fbuffer / 40, xmax0
				xmax0, xmin0 = xmax0 - deltax * arearatio - deltax * fbuffer / 40, xmin0
				
				print 'splitting in x: child box', ((xmin, ymin), (xmax, ymax))
				print 'splitting in x: remaining box', ((xmin0, ymin0), (xmax0, ymax0))
		# recurse
		childbox = mtransforms.Bbox(((xmin, ymin), (xmax, ymax)))
		nodesets_rectangles((child, grandchildren), childbox)
		plot_box(bbox=childbox, attrs=attrs, label=label)

def treesets_rectangles(tree):
	((xmin, xmax), (ymin, ymax)) = [(0, 1), (0, 1)]
	superset = None
	ax = plt.gca()
	ax.set_aspect(1.)
	ax.set_xlim(-0.1, 1.1)
	ax.set_ylim(-0.1, 1.1)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.axis('off')
	# start with plotting root node
	root, children = tree
	size, label, attrs = root
	assert size > 0
	rootbox = mtransforms.Bbox([[xmin, ymin], [xmax, ymax]])
	nodesets_rectangles((root, children), rootbox)
	plot_box(bbox=rootbox, attrs=attrs, label=label)

def nestedsets_rectangles(setsizes, labels = None, attrs = None):
	nsets = len(setsizes)
	if labels is None:
		labels = list(string.ascii_uppercase)[:nsets]
	if attrs is None:
		attrs = [default_attrs.next() for i in range(nsets)]
		#itertools.cycle([dict(ec=c, fc='None', alpha=0.7)
		#	for c in ['black', 'blue', 'red', 'green', 'magenta']])

	tree = []
	for node in list(zip(setsizes, labels, attrs))[::-1]:
		tree = [[node, tree]]
	treesets_rectangles(tree[0])




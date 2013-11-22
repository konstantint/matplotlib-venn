'''
Venn diagram plotting routines.
Rounded square, nested venn plotter.

Copyright 2013, Johannes Buchner.

Licensed under MIT license.
'''
import numpy
import warnings
import string

from matplotlib_venn import venn2, venn3
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.patches import FancyBboxPatch

def venn_nested(sets, labels = None, attrs = None):
	# assume nested in order
	if labels is None:
		labels = list(string.ascii_uppercase)[:len(sets)]
	if attrs is None:
		attrs = [dict(ec=c, fc='None', alpha=0.7)
			for c in ['black', 'blue', 'red', 'green', 'magenta']]
	((xmin, xmax), (ymin, ymax)) = [(0, 1), (0, 1)]
	#bb = mtransforms.Bbox([[xmin, ymin], [xmax, ymax]])
	superset = None
	ax = plt.gca()
	ax.set_aspect(1.)
	ax.set_xlim(-0.1, 1.1)
	ax.set_ylim(-0.1, 1.1)
	ax.set_xticks([])
	ax.set_yticks([])
	plt.axis('off')
	for i, (s, l, a) in enumerate(zip(sets, labels, attrs)):
		if i == 0:
			superset = s
		else:
			# update borders, proportional to area
			ratio = len(s) * 1. / len(superset)
			f = (ratio * 8/3.)**0.5
			fx = 1/2. * f
			fy = 3/4. * f
			f = ratio**0.5
			deltay = ymax - ymin
			deltax = ymax - ymin
			ymax = ymax - deltay * (1 - fx*0.95)
			xmin = xmin + deltax * (1 - fy*0.95)
			ymin = ymin + deltay * (fx*0.05)
			xmax = xmax - deltax * (fy*0.05)
		p_fancy = FancyBboxPatch((xmin, ymin),
			xmax - xmin, ymax - ymin,
			boxstyle="round,pad=0.02, rounding_size=0.05",
			**a)
		ax.add_patch(p_fancy)
		ax.text(xmin, ymax, l, 
			horizontalalignment='left',
			verticalalignment='top',
			)
		#draw_bbox(ax, bb)
		#p_fancy.set_boxstyle("round,pad=0.01, rounding_size=0.05")
		

import setuptools

version = '1.0'
long_description = ''
try:
	long_description = open("README.rst").read()
except:
	pass

setuptools.setup(
	name='matplotlib-subsets',
	version=version,
	description="Functions for plotting area-proportional hierarchical subset diagrams in matplotlib.",
	long_description=long_description,
	classifiers=[  # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: MIT License',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 2',
		'Topic :: Scientific/Engineering :: Visualization'
	],
	platforms=['Platform Independent'],
	keywords='matplotlib plotting charts venn-diagrams subset tree hierarchy',
	author='Johannes Buchner',
	author_email='buchner.johannes@gmx.at',
	url='https://github.com/JohannesBuchner/matplotlib-subsets',
	license='MIT',
	packages=['matplotlib_subsets'],
	zip_safe=True,
	install_requires=['matplotlib', 'numpy', 'scipy'],
	)


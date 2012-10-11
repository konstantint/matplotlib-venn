'''
Venn diagram plotting routines.
Setup script.

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under BSD.
'''

from setuptools import setup, find_packages
import sys, os

from setuptools.command.test import test as TestCommand

class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True
    def run_tests(self):
        import pytest  #import here, cause outside the eggs aren't loaded
        pytest.main(self.test_args)

version = '0.1'

setup(name='matplotlib-venn',
      version=version,
      description="Methods for generating area-proportional 2- and 3-way venn diagrams.",
      long_description="""Methods for generating area-proportional 2- and 3-way venn diagrams.""",
      classifiers=[ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Topic :: Scientific/Engineering :: Visualization'
      ],
      keywords='matplotlib plotting venn diagrams',
      author='Konstantin Tretyakov',
      author_email='kt@ut.ee',
      url='http://kt.era.ee/',
      license='BSD',
      namespace_packages = ['matplotlib'],
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=True,
      install_requires=['matplotlib', 'numpy', 'scipy'],
      tests_require=['pytest'],
      cmdclass = {'test': PyTest},
      entry_points="""
      # -*- Entry points: -*-
      """,
      )

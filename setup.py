'''
Venn diagram plotting routines.
Setup script.

Copyright 2012, Konstantin Tretyakov.
http://kt.era.ee/

Licensed under MIT license.
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

version = '0.2'

setup(name='matplotlib-venn',
      version=version,
      description="Functions for plotting area-proportional two- and three-way Venn diagrams in matplotlib.",
      long_description=open("README.rst").read(),
      classifiers=[ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Topic :: Scientific/Engineering :: Visualization'
      ],
      platforms=['Platform Independent'],
      keywords='matplotlib plotting charts venn-diagrams',
      author='Konstantin Tretyakov',
      author_email='kt@ut.ee',
      url='https://github.com/konstantint/matplotlib-venn',
      license='MIT',
      namespace_packages = ['matplotlib'],
      packages=find_packages(exclude=['ez_setup', 'examples', 'tests']),
      include_package_data=True,
      zip_safe=True,
      install_requires=['matplotlib', 'numpy', 'scipy'],
      tests_require=['pytest'],
      cmdclass = {'test': PyTest},
      entry_points=''
      )

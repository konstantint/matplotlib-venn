====================================================
Developer notes for Python/Matplotlib
====================================================

Starting development
--------------------

The package is formatted as a standard Python setuptools package, so
you you can use::

    $ python setup.py develop
	
to temporarily add it to your Python path. To remove it from the path use::

    $ python setup.py develop -u


Running the tests
-----------------

The recommended way to run package test is via `py.test <http://pytest.org/latest/>`_.
If you have it installed, just typing::

    $ py.test 

from the current directory will suffice. Note that ``setup.cfg`` contains some configuration
for ``py.test``. You may change the settings there while developing some feature to speed-up test runs.
For example, adding the name of a particular module to the end of the ``addopts`` setting will
limit test runs to that module only.

If you do not have ``py.test`` installed, you may run the tests via::

    $ python setup.py test
	
However, this will install the ``py.test`` egg locally in this directory and takes a bit more time to run.

If you plan to contribute code, please, test that it works both for Python 2.x and Python 3.x.


Functional tests
-----------------

The functional tests for the package are developed using the ``ipython`` notebook interface
and stored in the ``tests/*.ipynb`` files. Those notebook files are executed automatically when ``py.test`` is run
from the code in ``tests/functional_test.py``.

To review and develop functional tests you therefore have to install ``ipython[notebook]``::

    $ pip install ipython[notebook]
	
In order for the notebook code to execute correctly, the ``matplotlib_venn`` and ``tests`` packages must be in 
your Python's scope, which will happen automatically if you did ``python setup.py develop`` before.


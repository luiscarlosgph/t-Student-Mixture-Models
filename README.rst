t-Student-Mixture-Models
========================

Implementation of the paper: 'Robust mixture modelling using the t-distribution', D. Peel and G. J. McLachlan. Compatible with Python 2.7 and Python 3.

Dependencies
============

-  scikit-learn v0.18.1
-  numpy v1.11.0
-  scipy v0.19.0
-  setuptools v36.0.1

Install
=======

Using pip (no need to clone this repo):

::

    pip install smm --user

Manually:

::

    git clone https://github.com/luiscarlosgph/t-Student-Mixture-Models.git
    cd t-Student-Mixture-Models
    python setup.py build
    python setup.py install --user

Usage
=====

See example in `src/smm/example.py <src/smm/example.py>`__.

::

    python src/smm/example.py

Tests
=====

To run the tests execute:

::

    python setup.py test

Coverage
========

Current coverage: 79%. To re-run the coverage test (Ubuntu Ubuntu
16.04.2 LTS):

::

    python-coverage run ./setup.py test
    python-coverage html

Then open 'htmlcov/index.html' and check the line 'src/smm/smm'.

Author
======

Luis Carlos Garcia-Peraza Herrera (luis.herrera.14@ucl.ac.uk).

License
=======

BSD 3-Clause License, see LICENSE file for more information.

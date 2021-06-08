t-Student-Mixture-Models
------------------------
[![Build Status](https://travis-ci.org/luiscarlosgph/t-Student-Mixture-Models.svg?branch=master)](https://travis-ci.org/luiscarlosgph/t-Student-Mixture-Models)
[![Documentation Status](https://readthedocs.org/projects/t-student-mixture-models/badge/?version=latest)](http://t-student-mixture-models.readthedocs.io/en/latest/?badge=latest)  
Implementation of the paper: 'Robust mixture modelling using the t distribution', D. Peel and G. J. McLachlan.

<!--
Dependencies
------------
* scikit-learn v0.18.1
* numpy v1.11.0
* scipy v0.19.0
* setuptools v36.0.1
-->

Install with pip
----------------
```
$ python3 -m pip install smm --user
```

Install from source
-------------------
```
$ git clone https://github.com/luiscarlosgph/t-Student-Mixture-Models.git
$ cd t-Student-Mixture-Models
$ python3 setup.py install --user
```

Usage
-----
See example in [src/example.py](src/example.py). 
```
$ python3 src/example.py
```

Tests
-----
To run the tests execute:
```
$ python3 setup.py test
```

Coverage
--------
Current coverage: 79%.
To re-run the coverage test (Ubuntu 16.04.2 LTS):
```
$ python-coverage run ./setup.py test
$ python-coverage html
```
Then open 'htmlcov/index.html' and check the line 'src/smm/smm'.

Documentation
-------------
See [t-Student-Mixture-Models documentation](http://t-student-mixture-models.readthedocs.io/en/latest).

Author
------
Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).

License
-------
BSD 3-Clause License, see [LICENSE](https://github.com/luiscarlosgph/t-Student-Mixture-Models/blob/master/LICENSE) file for more information.

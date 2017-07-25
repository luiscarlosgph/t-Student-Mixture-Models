#!/usr/bin/env python

import os
import setuptools
import unittest

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setuptools.setup(name='smm',
    version='0.1.1',
    description='t-Student-Mixture-Models',
    author='Luis C. Garcia-Peraza Herrera',
    author_email='luis.herrera.14@ucl.ac.uk',
    license='BSD 3-Clause License',
    url='https://github.com/luiscarlosgph/t-Student-Mixture-Models',
    long_description=open('README.md').read(),
    packages=['smm'],
    package_dir={'smm' : 'src/smm'}, 
    test_suite = 'tests',
    classifiers=[
        'Intended Audience :: Developers'
        'Intended Audience :: Science/Research'
        'License :: OSI Approved'
        'Operating System :: MacOS'
        'Operating System :: Microsoft :: Windows'
        'Operating System :: POSIX'
        'Operating System :: Unix'
        'Programming Language :: C'
        'Programming Language :: Python'
        'Programming Language :: Python :: 2'
        'Programming Language :: Python :: 2.6'
        'Programming Language :: Python :: 2.7'
        'Programming Language :: Python :: 3'
        'Programming Language :: Python :: 3.3'
        'Programming Language :: Python :: 3.4'
        'Topic :: Scientific/Engineering'
        'Topic :: Software Development'
         ],
)

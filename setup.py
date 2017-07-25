#!/usr/bin/env python

import os
import setuptools
import unittest


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def my_module_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_smm)
    return suite

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
)

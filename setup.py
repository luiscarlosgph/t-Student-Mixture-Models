#!/usr/bin/env python

import setuptools
import unittest

# Read the contents of the README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(name='smm',
    version='0.1.6',
    description='t-Student-Mixture-Models',
    author='Luis C. Garcia-Peraza Herrera',
    author_email='luis.herrera.14@ucl.ac.uk',
    license='BSD 3-Clause License',
    url='https://github.com/luiscarlosgph/t-Student-Mixture-Models',
    packages=['smm'],
    package_dir={'smm' : 'src/smm'}, 
    test_suite = 'tests',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires = ['sklearn'],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Programming Language :: C',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
         ],
)

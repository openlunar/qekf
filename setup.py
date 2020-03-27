from setuptools import setup, find_packages

import os
import numpy

setup(name='qekf',
      test_suite       = 'test.qekf_test_suite',
      install_requires = ['numpy', 'scipy'])

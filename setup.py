#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='jax_cosmo',
      version='0.0.1',
      description='Differentiable Python Cosmology Library',
      author='JAX Cosmo developers',
      packages=find_packages(),
      install_requires=['jax', 'jaxlib'],
      tests_require = ['pyccl']
     )

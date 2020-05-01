#!/usr/bin/env python
from setuptools import setup, find_packages

setup(name='jax_cosmo',
      version='0.0.1',
      description='Differentiable Python Cosmology Library',
      author='Francois Lanusse',
      packages=find_packages(),
      install_requires=['jax', 'jaxlib']
     )

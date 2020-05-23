#!/usr/bin/env python
import io
import os

from setuptools import find_packages
from setuptools import setup

this_directory = os.path.abspath(os.path.dirname(__file__))
with io.open(os.path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="jax-cosmo",
    description="Differentiable Python Cosmology Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="jax-cosmo developers",
    packages=find_packages(),
    url="https://github.com/DifferentiableUniverseInitiative/jax_cosmo",
    install_requires=["jax", "jaxlib"],
    tests_require=["pyccl"],
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
)

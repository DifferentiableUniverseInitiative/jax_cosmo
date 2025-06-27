# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

jax-cosmo is a differentiable cosmology library built on JAX, providing end-to-end automatic differentiation for cosmological computations. It enables efficient inference (HMC, VI) and survey optimization tasks by computing observables like angular power spectra and correlation functions for various tracers (lensing, clustering, CMB lensing).

## Development Commands

### Testing
```bash
pytest                    # Run all tests
pytest tests/test_*.py    # Run specific test file
```

### Code Formatting
```bash
black .                   # Format code with Black
```

### Installation
```bash
pip install -e .          # Install in development mode
pip install -e .[test]    # Install with test dependencies
```

### Documentation
```bash
cd docs && make html      # Build documentation with Sphinx
```

## Architecture

### Core Design Principles
- **JAX-first**: All functions must be JAX-compatible for automatic differentiation
- **Avoid Python loops**: Use `jax.vmap` or `jax.lax` for XLA-compatible operations
- **Functions over methods**: Enable `jax.grad(function)(cosmo, params)` pattern
- **Minimal caching**: Only through interpolation tables when needed

### Key Modules

**Core Infrastructure:**
- `core.py`: Cosmology container class and core utilities
- `parameters.py`: Parameter definitions and handling

**Physics Modules:**
- `background.py`: Background cosmology (distances, growth factors)
- `transfer.py`: Transfer functions (Eisenstein-Hu, etc.)
- `power.py`: Matter power spectrum calculations
- `angular_cl.py`: Limber integration and angular power spectra
- `bias.py`: Galaxy bias models

**Observational Probes:**
- `probes.py`: Tracer definitions (lensing, clustering, etc.)
- `redshift.py`: Redshift distribution handling
- `likelihood.py`: Likelihood functions

**Utilities:**
- `scipy/`: JAX implementations of scipy functions
- `sparse.py`: Sparse matrix operations
- `jax_utils.py`: JAX utility functions

### Container Class Pattern

The `container` class is fundamental to jax-cosmo's design:
- Stores traceable parameters in `self.params` (for differentiation)
- Stores static configuration in `self.config` (non-differentiable)
- JAX-compatible through custom pytree registration
- Enables `jax.grad(function)(container_object)` pattern

Example:
```python
class gaussian_nz(container):
    def __init__(self, z0, sigma, zmax=10, **kwargs):
        super().__init__(z0, sigma, zmax=zmax, **kwargs)
    
    def __call__(self, z):
        z0, sigma = self.params
        return jnp.exp(-0.5 * (z - z0)**2 / sigma**2)
```

### Two-Point Functions

All 2-point functions use the Limber approximation following CCL's generalized tracer approach. Each tracer is characterized by:
- Radial kernel function
- Ell-dependent prefactor  
- Transfer function

Probes are container objects that aggregate redshift distributions and parameters for different observational tracers.

## Development Guidelines

### Code Style
- Use Black for formatting (`black .`)
- Follow NumPy/SciPy docstring format
- Import order managed by `reorder_python_imports`
- Pre-commit hooks available for automated formatting

### JAX Best Practices
- Vectorize operations with `jax.vmap` instead of loops
- Use `jax.lax` control flow for loops that can't be vectorized  
- Avoid Python control flow in traced functions
- Prefer pure functions for differentiability
- Use `jnp` instead of `np` for all array operations

### Testing
- Tests use pytest framework
- Comparison tests against CCL (Core Cosmology Library)
- pytest.ini configures to suppress warnings
- Test dependencies include `pyccl`

### Dependencies
- Core: `jax`, `jaxlib`
- Testing: `pyccl` (for comparison tests)
- Documentation: Sphinx (see docs/requirements.txt)
# jax-cosmo

[![Join the chat at https://gitter.im/DifferentiableUniverseInitiative/jax_cosmo](https://badges.gitter.im/DifferentiableUniverseInitiative/jax_cosmo.svg)](https://gitter.im/DifferentiableUniverseInitiative/jax_cosmo?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) ![Python package](https://github.com/DifferentiableUniverseInitiative/jax_cosmo/workflows/Python%20package/badge.svg)  

A differentiable cosmology library in JAX.

**Note**: This package is still in the development phase, expect changes to the API. We hope to make this project a community effort, contributions of all kind are most welcome!
Have a look at the [GitHub issues](https://github.com/DifferentiableUniverseInitiative/jax_cosmo/issues) to see what is needed or if you have any thoughts on the design, and don't hesitate to join the [Gitter room](https://gitter.im/DifferentiableUniverseInitiative/jax_cosmo) for discussions.

## TL;DR

This is what `jax-cosmo` aims to do:

```python
data = #... some measured Cl data vector
nz1,nz2,nz3,nz4 = #.... redshift distributions of bins
def likelihood(cosmo):
  # Define a list of probes
  probes = [jax_cosmo.probes.WeakLensing([nz1, nz2, nz3, nz4]),
            jax_cosmo.probes.NumberCounts([nz1, nz2, nz3, nz4])]

  # Compute mean and covariance of angular Cls
  mu, cov = jax_cosmo.angular_cl.gaussian_cl_covariance(cosmo, ell, probes)

  # Return likelihood value
  return jax_cosmo.likelihood.gaussian_log_likelihood(data, mu, cov)

# Compute derivatives of the likelihood with respect to cosmological parameters
g = jax.grad(likelihood)(cosmo)

# Compute Fisher matrix of cosmological parameters
F = - jax.hessian(likelihood)(cosmo)
```
This is how you can compute gradients and hessians of any functions in `jax-cosmo`,
all of this without any finite differences.

Check out a full example here:

## What is JAX?

[JAX](https://github.com/google/jax) = NumPy + autodiff + GPU

JAX is a framework for automatic differentiation (like TensorFlow or PyTorch) but following the NumPy API, and using the GPU/TPU enable XLA backend.

What does that mean?
  - You write plain Python/NumPy code, no need to learn a different language
  - It runs on GPU, you don't need to do anything particular
  - You can take derivatives of any quantity with respect to any parameters by
  automatic differentiation.

Checkout the [JAX](https://github.com/google/jax) project page to learn more!

## Install

`jax-cosmo` is pure Python, so installing is a breeze:
```bash
$ pip install jax-cosmo
```

## Philosophy

Here are some of the design guidelines:
  - Implementation of equations should be human readable, and documentation should always live next to the implementation.
  - Should always be trivially installable: external dependencies should be kept
  to a minimum, especially the ones that require compilation or with restrictive licenses.
  - Keep API and implementation simple and intuitive, minimize user and developer
  surprise.
  - “Debugging is twice as hard as writing the code in the first place. Therefore, if you write the code as cleverly as possible, you are, by definition, not smart enough to debug it.” -Brian Kernighan, quote stolen from
  [here](https://flax.readthedocs.io/en/latest/philosophy.html).

## Contributing

`jax-cosmo` aims to be a community effort, contributions are most welcome and
can come in several forms
  - Bug reports
  - API design suggestions
  - (Pull) requests for more features
  - Examples and notebooks of cool things that can be done with the code

The issue page is a good place to start, but don't hesitate to come chat in the
Gitter room.

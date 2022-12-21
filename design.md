# Design document for jax-cosmo

This document details the API, implementation choices, and internal mechanisms.

## Objective

Provide a fully end-to-end automatically differentiable cosmology library,
providing observables (e.g. 2pt angular Cls or correlation functions) for a
variety of tracers (e.g. lensing, clustering, CMB lensing, etc.).

This tool will make it easy to perform efficient inference (e.g. HMC, VI), as well as a wide variety of survey optimization tasks (e.g. photoz binning).

## Related Work

There isn't any equivalent of this project so far, to the best of our
knowledge.

But there are a wide collection of non differentiable cosmology libraries.
  - CCL
  - Cosmosis
  - CosmicFish
  - ...

## Design Overview

### JAX

This section covers some of the design aspects related to the JAX backend. It is
probably good to have a look at the JAX [intro](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html).

#### JAX best practices

 - Loops are evil! avoid them at all cost! The problem is that normal Python
 loops will get unrolled into a very long computational graph. Instead, as
 much as possible, use batching with `jax.vmap` or the low level loop utilities
 in `jax.lax` for XLA compatible control loops.
 - Functions should be preferred to methods. Because we want to be able to do
 things like:
 ```python
 jax.grad(Omega_m)(cosmo, a)
 ```
 which will compute the derivative with respect to the cosmology. If cosmology
 wasn't an argument, it would be a lot more wordy:
 ```python
 def fn(cosmo):
    return cosmo.Omega_m(a)
 jax.grad(fn)
 ```
 - Careful with caching! Avoid it if possible, the only acceptable form of
 caching is by computing an interpolation table and returning the result of an
 interpolation. Only useful when needing consecutive calls to that table in the
 same function.

#### The container class

Here is a situation, we want to define a parametric redshift distribution, say
a Gaussian with mean `z0` and standard deviation `sigma`. This redshift distribution
needs to be used through many operations all the  way to the likelihood, so
we want a structure that can store these 2 parameters, and compatible with JAX
tracing.

So we define a `container` class, which is a generic structure holding some
parameters that need to be traced, and some static configuration arguments. The
`container` class knows how to pack and unpack its arguments, in a manner compatible
with the JAX custom types [(see here)](https://jax.readthedocs.io/en/latest/notebooks/JAX_pytrees.html)

The `container` class will store all the positional arguments it receives during
init in a list stored in `self.params`. These parameters are meant to be the
traceable arguments, so anything that might need to be differentiable should go
there. In addition, non traceable, configuration arguments, like a numerical precision
flag, or a computation flag, can be stored by providing keyword arguments to the
init. These arguments will be stored in `self.config`

Concretely, we can define our redshift distribution this way:
```python
class gaussian_nz(container):

  def __init__(self, z0, sigma, zmax=10, **kwargs):
    super(gaussian_nz, self).__init__(z0, sigma, # Traceable parameters
                                      zmax=zmax, **kwargs) # non-traceable configuration

  def __call__(self, z):
    z0, sigma = self.params
    return np.clip(exp(-0.5*( z - z0)**2/sigma**2),
                   0., self.config['zmax'])
```
Note that in this example, the `__init__` isn't doing anything, we just leave it
for readibility. JAX will know how to properly flatten and inflate this object
through the tracing process. You can for instance now do the following:
```python
# Define a likelihood, function of the redshift distribution
def likelihood(nz):
  ... # some computation that uses this nz
  return likelihood_value
>>> nz = gaussian_nz(1., 0.1)
>>> jax.grad(likelihood)(nz)
(0.5346, 0.1123 )
```
where what is the returned is the gradient of the redshift object.

In general, this container mechanism can be used to aggregate a bunch of
parameters in one place, in a way that JAX knows how to handle.

### Cosmology

In this section we cover aspects related to the cosmology API and implementation.

#### Code structure

Here are the main modules:

  - The `Cosmology` class: stores cosmological parameters, it is essentially an
  instance of the `container`.

  - The `background` module: hosts functions of the comology to compute various
  background related quantities.

  - The `transfer` module: Libary of transfer functions, e.g. EisensteinHu

  - The `probes` module: Hosts the definition of various probes, as defined in the next section

  - The `angular_cl` module: hosts the Limber integration code, and covariance tools

To these existing modules, we should add a `non_linear` for things like halofit.

#### Handling of 2pt functions

For now, and in the foreseable future, all 2pt functions are computed using the
Limber approximation.

We follow the structure adopted by [CCL](https://github.com/LSSTDESC/CCL) to define two point functions of generalized tracers, as proposed by David Alonso in this issue [#627](https://github.com/LSSTDESC/CCL/issues/627). To summarize, each
tracer (e.g. lensing, number count, etc.) is characterized by the following:
  - A radial kernel function
  - An ell dependent prefactor
  - A transfer function

In `jax-cosmo`, we define `probes` that are container
objects (i.e. which can be differentiated), gathering in particular a list of
redshift distributions, and any other necessary parameters.

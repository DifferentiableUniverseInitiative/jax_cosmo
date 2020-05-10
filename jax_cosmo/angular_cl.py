# This module contains functions to compute angular cls for various tracers
from functools import partial

import jax.numpy as np
from jax import vmap, lax, jit

import jax_cosmo.constants as const
from jax_cosmo.utils import z2a, a2z
from jax_cosmo.scipy.integrate import simps
import jax_cosmo.background as bkgrd
import jax_cosmo.power as power

def _get_cl_ordering(probes):
  """
  Utility function to get the indices for Cls from a list of probes
  """
  n_tracers = sum([p.n_tracers for p in probes])
  # Define an ordering for the blocks of the signal vector
  cl_index = []
  for i in range(n_tracers):
    for j in range(i, n_tracers):
      cl_index.append((i, j))
  return cl_index

def _get_cov_blocks_ordering(probes):
  """
  Utility function to get the ordering of the covariance matrix blocks
  """
  cl_index = _get_cl_ordering(probes)

  def find_index(a, b):
    if (a,b) in cl_index:
      return cl_index.index((a,b))
    else:
      return cl_index.index((b,a))

  cov_blocks = []
  for (i,j) in cl_index:
      for (m,n) in cl_index:
          cov_blocks.append((find_index(i,m),
                             find_index(j,n),
                             find_index(i,n),
                             find_index(j,m)))
  return cov_blocks


@partial(vmap, in_axes=(None, 0, None), out_axes=1)
def angular_cl(cosmo, ell, probes, amin=0.002):
  """
  Computes angular Cls for the provided probes

  All using the Limber approximation

  Returns
  -------

  cls: [ell, ncls]
  """
  def integrand(a):
    # Step 1: retrieve the associated comoving distance
    chi = bkgrd.radial_comoving_distance(cosmo, a)

    # Step 2: get the power spectrum for this combination of chi and a
    k = (ell+0.5) / np.clip(chi, 1.)

    # pk should have shape [na]
    pk = power.linear_matter_power(cosmo, k, a)

    # Compute the kernels for all probes
    kernels = np.vstack([ p.radial_kernel(cosmo, a2z(a)) *
                          p.ell_factor(ell) *
                          p.constant_factor(cosmo)
                         for p in probes])

    # Define an ordering for the blocks of the signal vector
    cl_index = np.array(_get_cl_ordering(probes))
    # Compute all combinations of tracers
    @jit
    def combine_kernels(inds):
      return kernels[inds[0]] * kernels[inds[1]]
    # Now kernels has shape [ncls, na]
    kernels = lax.map(combine_kernels, cl_index)

    result = pk * kernels * bkgrd.dchioverda(cosmo, a) / np.clip(chi**2, 1.)

    # We transpose the result just to make sure that na is first
    return result.T

  return simps(integrand, amin, 1., 512) / const.c**2

def noise_cl(ell, probes):
  """
  Computes noise contributions to auto-spectra
  """
  n_ell = len(ell)
  # Concatenate noise power for each tracer
  noise = np.concatenate([p.noise() for p in probes])
  # Define an ordering for the blocks of the signal vector
  cl_index = np.array(_get_cl_ordering(probes))
  # Only include a noise contribution for the auto-spectra
  def get_noise_cl(inds):
      i,j = inds
      delta = 1. - np.clip(np.abs(i-j), 0., 1.)
      return noise[i] * delta * np.ones(n_ell)
  return lax.map(get_noise_cl, cl_index)

def gaussian_cl_covariance(cosmo, ell, probes, f_sky=0.25, return_cls=True):
  """
  Computes a Gaussian covariance for the angular cls of the provided probes

  return_cls: (returns signal + noise cl, covariance)
  """
  ell = np.atleast_1d(ell)
  n_ell = len(ell)

  # Compute signal vectors
  cl_signal = angular_cl(cosmo, ell, probes)
  cl_noise = noise_cl(ell, probes)

  # Adding noise to auto-spectra
  cl_obs = cl_signal + cl_noise
  n_cls = cl_obs.shape[0]

  # Normalization of covariance
  norm = (2*ell + 1) * np.gradient(ell) * f_sky

  # Retrieve ordering for blocks of the covariance matrix
  cov_blocks = np.array(_get_cov_blocks_ordering(probes))

  def get_cov_block(inds):
      a, b, c, d = inds
      cov = (cl_obs[a]*cl_obs[b] + cl_obs[c]*cl_obs[d]) / norm
      return cov*np.eye(n_ell)

  cov_mat = lax.map(get_cov_block, cov_blocks)

  # Reshape covariance matrix into proper matrix
  cov_mat = cov_mat.reshape((n_cls, n_cls, n_ell, n_ell))
  cov_mat = cov_mat.transpose(axes=(0,2,1,3)).reshape((n_ell*n_cls,
                                                       n_ell*n_cls))
  if return_cls:
    return cl_obs.flatten(), cov_mat
  else:
    return cov_mat

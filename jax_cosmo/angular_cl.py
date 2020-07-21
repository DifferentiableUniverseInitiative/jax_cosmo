# This module contains functions to compute angular cls for various tracers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

import jax.numpy as np
from jax import jit
from jax import lax
from jax import vmap

import jax_cosmo.background as bkgrd
import jax_cosmo.constants as const
import jax_cosmo.power as power
import jax_cosmo.transfer as tklib
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.utils import a2z
from jax_cosmo.utils import z2a


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
        if (a, b) in cl_index:
            return cl_index.index((a, b))
        else:
            return cl_index.index((b, a))

    cov_blocks = []
    for (i, j) in cl_index:
        for (m, n) in cl_index:
            cov_blocks.append(
                (find_index(i, m), find_index(j, n), find_index(i, n), find_index(j, m))
            )
    return cov_blocks


def angular_cl(
    cosmo, ell, probes, transfer_fn=tklib.Eisenstein_Hu, nonlinear_fn=power.halofit
):
    """
    Computes angular Cls for the provided probes

    All using the Limber approximation

    Returns
    -------

    cls: [ell, ncls]
    """
    # Retrieve the maximum redshift probed
    zmax = max([p.zmax for p in probes])

    # We define a function that computes a single l, and vectorize it
    @partial(vmap, out_axes=1)
    def cl(ell):
        def integrand(a):
            # Step 1: retrieve the associated comoving distance
            chi = bkgrd.radial_comoving_distance(cosmo, a)

            # Step 2: get the power spectrum for this combination of chi and a
            k = (ell + 0.5) / np.clip(chi, 1.0)

            # pk should have shape [na]
            pk = power.nonlinear_matter_power(cosmo, k, a, transfer_fn, nonlinear_fn)

            # Compute the kernels for all probes
            kernels = np.vstack([p.kernel(cosmo, a2z(a), ell) for p in probes])

            # Define an ordering for the blocks of the signal vector
            cl_index = np.array(_get_cl_ordering(probes))
            # Compute all combinations of tracers
            def combine_kernels(inds):
                return kernels[inds[0]] * kernels[inds[1]]

            # Now kernels has shape [ncls, na]
            kernels = lax.map(combine_kernels, cl_index)

            result = pk * kernels * bkgrd.dchioverda(cosmo, a) / np.clip(chi ** 2, 1.0)

            # We transpose the result just to make sure that na is first
            return result.T

        return simps(integrand, z2a(zmax), 1.0, 512) / const.c ** 2

    return cl(ell)


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
        i, j = inds
        delta = 1.0 - np.clip(np.abs(i - j), 0.0, 1.0)
        return noise[i] * delta * np.ones(n_ell)

    return lax.map(get_noise_cl, cl_index)


def gaussian_cl_covariance(ell, probes, cl_signal, cl_noise, f_sky=0.25):
    """
    Computes a Gaussian covariance for the angular cls of the provided probes

    return_cls: (returns covariance)
    """
    ell = np.atleast_1d(ell)
    n_ell = len(ell)

    # Adding noise to auto-spectra
    cl_obs = cl_signal + cl_noise
    n_cls = cl_obs.shape[0]

    # Normalization of covariance
    norm = (2 * ell + 1) * np.gradient(ell) * f_sky

    # Retrieve ordering for blocks of the covariance matrix
    cov_blocks = np.array(_get_cov_blocks_ordering(probes))

    def get_cov_block(inds):
        a, b, c, d = inds
        cov = (cl_obs[a] * cl_obs[b] + cl_obs[c] * cl_obs[d]) / norm
        return cov * np.eye(n_ell)

    cov_mat = lax.map(get_cov_block, cov_blocks)

    # Reshape covariance matrix into proper matrix
    cov_mat = cov_mat.reshape((n_cls, n_cls, n_ell, n_ell))
    cov_mat = cov_mat.transpose(axes=(0, 2, 1, 3)).reshape(
        (n_ell * n_cls, n_ell * n_cls)
    )
    return cov_mat


def gaussian_cl_covariance_and_mean(
    cosmo,
    ell,
    probes,
    transfer_fn=tklib.Eisenstein_Hu,
    nonlinear_fn=power.halofit,
    f_sky=0.25,
):
    """
    Computes a Gaussian covariance for the angular cls of the provided probes

    return_cls: (returns signal + noise cl, covariance)
    """
    ell = np.atleast_1d(ell)
    n_ell = len(ell)

    # Compute signal vectors
    cl_signal = angular_cl(
        cosmo, ell, probes, transfer_fn=transfer_fn, nonlinear_fn=nonlinear_fn
    )
    cl_noise = noise_cl(ell, probes)

    # retrieve the covariance
    cov_mat = gaussian_cl_covariance(ell, probes, cl_signal, cl_noise, f_sky)

    return cl_signal.flatten(), cov_mat

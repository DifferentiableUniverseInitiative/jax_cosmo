# This module defines kernel functions for various tracers
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
from jax import jit
from jax import vmap
from jax.tree_util import register_pytree_node_class

import jax_cosmo.background as bkgrd
import jax_cosmo.constants as const
from jax_cosmo.jax_utils import container
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.utils import a2z
from jax_cosmo.utils import z2a

__all__ = ["WeakLensing", "NumberCounts"]


@jit
def weak_lensing_kernel(cosmo, pzs, z, ell):
    """
    Returns a weak lensing kernel
    """
    z = np.atleast_1d(z)
    zmax = max([pz.zmax for pz in pzs])
    # Retrieve comoving distance corresponding to z
    chi = bkgrd.radial_comoving_distance(cosmo, z2a(z))

    @vmap
    def integrand(z_prime):
        chi_prime = bkgrd.radial_comoving_distance(cosmo, z2a(z_prime))
        # Stack the dndz of all redshift bins
        dndz = np.stack([pz(z_prime) for pz in pzs], axis=0)
        return dndz * np.clip(chi_prime - chi, 0) / np.clip(chi_prime, 1.0)

    # Computes the radial weak lensing kernel
    radial_kernel = np.squeeze(simps(integrand, z, zmax, 256) * (1.0 + z) * chi)
    # Constant term
    constant_factor = 3.0 * const.H0 ** 2 * cosmo.Omega_m / 2.0 / const.c
    # Ell dependent factor
    ell_factor = np.sqrt((ell - 1) * (ell) * (ell + 1) * (ell + 2)) / (ell + 0.5) ** 2
    return constant_factor * ell_factor * radial_kernel


@jit
def density_kernel(cosmo, pzs, bias, z, ell):
    """
    Computes the number counts density kernel
    """
    # stack the dndz of all redshift bins
    dndz = np.stack([pz(z) for pz in pzs], axis=0)
    # Compute radial NLA kernel: same as clustering
    if isinstance(bias, list):
        # This is to handle the case where we get a bin-dependent bias
        b = np.stack([b(cosmo, z) for b in bias], axis=0)
    else:
        b = bias(cosmo, z)
    radial_kernel = dndz * b * bkgrd.H(cosmo, z2a(z))
    # Normalization,
    constant_factor = 1.0
    # Ell dependent factor
    ell_factor = 1.0
    return constant_factor * ell_factor * radial_kernel


@jit
def nla_kernel(cosmo, pzs, bias, z, ell):
    """
    Computes the NLA IA kernel
    """
    # stack the dndz of all redshift bins
    dndz = np.stack([pz(z) for pz in pzs], axis=0)
    # Compute radial NLA kernel: same as clustering
    if isinstance(bias, list):
        # This is to handle the case where we get a bin-dependent bias
        b = np.stack([b(cosmo, z) for b in bias], axis=0)
    else:
        b = bias(cosmo, z)
    radial_kernel = dndz * b * bkgrd.H(cosmo, z2a(z))
    # Apply common A_IA normalization to the kernel
    # Joachimi et al. (2011), arXiv: 1008.3491, Eq. 6.
    radial_kernel *= (
        -(5e-14 * const.rhocrit) * cosmo.Omega_m / bkgrd.growth_factor(cosmo, z2a(z))
    )
    # Constant factor
    constant_factor = 1.0
    # Ell dependent factor
    ell_factor = np.sqrt((ell - 1) * (ell) * (ell + 1) * (ell + 2)) / (ell + 0.5) ** 2
    return constant_factor * ell_factor * radial_kernel


@register_pytree_node_class
class WeakLensing(container):
    """
    Class representing a weak lensing probe, with a bunch of bins

    Parameters:
    -----------
    redshift_bins: list of nzredshift distributions
    ia_bias: (optional) if provided, IA will be added with the NLA model,
    either a single bias object or a list of same size as nzs
    multiplicative_bias: (optional) adds an (1+m) multiplicative bias, either single
    value or list of same length as redshift bins

    Configuration:
    --------------
    sigma_e: intrinsic galaxy ellipticity
    """

    def __init__(
        self,
        redshift_bins,
        ia_bias=None,
        multiplicative_bias=0.0,
        sigma_e=0.26,
        **kwargs
    ):
        # Depending on the Configuration we will trace or not the ia_bias in the
        # container
        if ia_bias is None:
            ia_enabled = False
            args = (redshift_bins, multiplicative_bias)
        else:
            ia_enabled = True
            args = (redshift_bins, multiplicative_bias, ia_bias)
        if "ia_enabled" not in kwargs.keys():
            kwargs["ia_enabled"] = ia_enabled
        super(WeakLensing, self).__init__(*args, sigma_e=sigma_e, **kwargs)

    @property
    def n_tracers(self):
        """
        Returns the number of tracers for this probe, i.e. redshift bins
        """
        # Extract parameters
        pzs = self.params[0]
        return len(pzs)

    @property
    def zmax(self):
        """
        Returns the maximum redsfhit probed by this probe
        """
        # Extract parameters
        pzs = self.params[0]
        return max([pz.zmax for pz in pzs])

    def kernel(self, cosmo, z, ell):
        """
        Compute the radial kernel for all nz bins in this probe.

        Returns:
        --------
        radial_kernel: shape (nbins, nz)
        """
        z = np.atleast_1d(z)
        # Extract parameters
        pzs, m = self.params[:2]
        kernel = weak_lensing_kernel(cosmo, pzs, z, ell)
        # If IA is enabled, we add the IA kernel
        if self.config["ia_enabled"]:
            bias = self.params[2]
            kernel += nla_kernel(cosmo, pzs, bias, z, ell)
        # Applies measurement systematics
        if isinstance(m, list):
            m = np.expand_dims(np.stack([mi for mi in m], axis=0), 1)
        kernel *= 1.0 + m
        return kernel

    def noise(self):
        """
        Returns the noise power for all redshifts
        return: shape [nbins]
        """
        # Extract parameters
        pzs = self.params[0]
        # retrieve number of galaxies in each bins
        ngals = np.array([pz.gals_per_steradian for pz in pzs])
        if isinstance(self.config["sigma_e"], list):
            sigma_e = np.array([s for s in self.config["sigma_e"]])
        else:
            sigma_e = self.config["sigma_e"]
        return sigma_e ** 2 / ngals


@register_pytree_node_class
class NumberCounts(container):
    """ Class representing a galaxy clustering probe, with a bunch of bins

    Parameters:
    -----------
    redshift_bins: nzredshift distributions

    Configuration:
    --------------
    has_rsd....
    """

    def __init__(self, redshift_bins, bias, has_rsd=False, **kwargs):
        super(NumberCounts, self).__init__(
            redshift_bins, bias, has_rsd=has_rsd, **kwargs
        )

    @property
    def zmax(self):
        """
        Returns the maximum redsfhit probed by this probe
        """
        # Extract parameters
        pzs = self.params[0]
        return max([pz.zmax for pz in pzs])

    @property
    def n_tracers(self):
        """ Returns the number of tracers for this probe, i.e. redshift bins
        """
        # Extract parameters
        pzs = self.params[0]
        return len(pzs)

    def kernel(self, cosmo, z, ell):
        """ Compute the radial kernel for all nz bins in this probe.

        Returns:
        --------
        radial_kernel: shape (nbins, nz)
        """
        z = np.atleast_1d(z)
        # Extract parameters
        pzs, bias = self.params
        # Retrieve density kernel
        kernel = density_kernel(cosmo, pzs, bias, z, ell)
        return kernel

    def noise(self):
        """ Returns the noise power for all redshifts
        return: shape [nbins]
        """
        # Extract parameters
        pzs = self.params[0]
        ngals = np.array([pz.gals_per_steradian for pz in pzs])
        return 1.0 / ngals

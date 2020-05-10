# This module defines kernel functions for various tracers

import jax.numpy as np
from jax import vmap, jit

from jax_cosmo.scipy.integrate import simps
import jax_cosmo.constants as const
import jax_cosmo.background as bkgrd
from jax_cosmo.utils import z2a, a2z
from jax_cosmo.jax_utils import container
from jax.tree_util import register_pytree_node_class

__all__ = ["WeakLensing", "NumberCounts"]

@register_pytree_node_class
class WeakLensing(container):
  """
  Class representing a weak lensing probe, with a bunch of bins

  Parameters:
  -----------
  redshift_bins: nzredshift distributions
  sigma_e: intrinsic galaxy ellipticity

  Configuration:
  --------------
  sigma_e: intrinsic galaxy ellipticity
  has_shear:, ia_bias, use_bias.... these are not functional
  """
  def __init__(self, redshift_bins,
               sigma_e=0.26,
               use_shear=True,
               **kwargs):
    super(WeakLensing, self).__init__(redshift_bins,
                                      sigma_e=sigma_e,
                                      use_shear=use_shear,
                                      **kwargs)
  @property
  def n_tracers(self):
    """
    Returns the number of tracers for this probe, i.e. redshift bins
    """
    # Extract parameters
    pzs = self.params[0]
    return len(pzs)

  def constant_factor(self, cosmo):
    return 3.0 * const.H0**2 * cosmo.Omega_m / 2.0 / const.c

  @jit
  def radial_kernel(self, cosmo, z):
    """
    Compute the radial kernel for all nz bins in this probe.

    Returns:
    --------
    radial_kernel: shape (nbins, nz)
    """
    z = np.atleast_1d(z)
    # Extract parameters
    pzs = self.params[0]
    # Find the maximum necessary redshift
    zmax = max([pz.zmax for pz in pzs])

    # Retrieve comoving distance corresponding to z
    chi = bkgrd.radial_comoving_distance(cosmo, z2a(z))

    @vmap
    def integrand(z_prime):
      chi_prime = bkgrd.radial_comoving_distance(cosmo, z2a(z_prime))
      # Stack the dndz of all redshift bins
      dndz = np.stack([pz(z_prime) for pz in pzs], axis=0)
      return dndz * np.clip(chi_prime - chi, 0) / np.clip(chi_prime, 1.)

    result = simps(integrand, z, zmax, 256) * (1. + z ) * chi

    return np.squeeze(result)

  def ell_factor(self, ell):
    """
    Computes the ell dependent factor for this probe.
    """
    return np.sqrt((ell-1)*(ell)*(ell+1)*(ell+2))/(ell+0.5)**2

  def noise(self):
    """
    Returns the noise power for all redshifts
    return: shape [nbins]
    """
    # Extract parameters
    pzs = self.params[0]

    # retrieve number of galaxies in each bins
    ngals = np.array([pz.gals_per_steradian for pz in pzs])

    # TODO: add mechanism for effective number density, maybe a bin dependent
    # efficiency
    return self.config['sigma_e']**2 / ngals


@register_pytree_node_class
class NumberCounts(container):
  """
  Class representing a galaxy clustering probe, with a bunch of bins

  Parameters:
  -----------
  redshift_bins: nzredshift distributions

  Configuration:
  --------------
  has_rsd....
  """
  def __init__(self, redshift_bins, bias,
               has_rsd=False,
               **kwargs):
    super(NumberCounts, self).__init__(redshift_bins,
                                       bias,
                                       has_rsd=has_rsd,
                                       **kwargs)
  @property
  def n_tracers(self):
    """
    Returns the number of tracers for this probe, i.e. redshift bins
    """
    # Extract parameters
    pzs = self.params[0]
    return len(pzs)

  def constant_factor(self, cosmo):
    return 1.0

  @jit
  def radial_kernel(self, cosmo, z):
    """
    Compute the radial kernel for all nz bins in this probe.

    Returns:
    --------
    radial_kernel: shape (nbins, nz)
    """
    z = np.atleast_1d(z)
    # Extract parameters
    pzs, bias = self.params

    # stack the dndz of all redshift bins
    dndz = np.stack([pz(z) for pz in pzs], axis=0)

    return dndz * bias(z) * bkgrd.H(cosmo, z2a(z))

  def ell_factor(self, ell):
    """
    Computes the ell dependent factor for this probe.
    """
    return 1.

  def noise(self):
    """
    Returns the noise power for all redshifts
    return: shape [nbins]
    """
    # Extract parameters
    pzs = self.params[0]
    ngals = np.array([pz.gals_per_steradian for pz in pzs])
    return 1./ngals

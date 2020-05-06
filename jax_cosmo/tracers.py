# This module defines kernel functions for various tracers

import jax.numpy as np

from jax_cosmo.scipy.integrate import simps
import jax_cosmo.constants as const
import jax_cosmo.background as bkgrd
from jax_cosmo.utils import z2a, a2z

__all__ = ["lensing_kernel", "lensing_tracer"]

def lensing_kernel(cosmo, z, nz):
  """
  Computes the lensing kernel W_L, for a flat universe
  """
  chi = bkgrd.radial_comoving_distance(cosmo, z2a(z))

  def integrand(z_prime):
    chi_prime = bkgrd.radial_comoving_distance(cosmo, z2a(z_prime))
    return  nz(z_prime) * np.clip(chi_prime - chi, 0) / (chi_prime + 1e-5)

  return np.squeeze(simps(integrand, z, nz.zmax, 256))


def get_lensing_tracer_fn(nz, use_IA=False):
  """
  Generate a lensing tracer function for the provided nz distribution
  """

  def fn(cosmo, ell, z):
    # Tracer specific \ell dependent factor
    ell_factor = np.sqrt((ell-1)*(ell)*(ell+1)*(ell+2))/(ell+0.5)**2

    # Contant factor
    cst_factor = 3.0 * const.H0**2 * cosmo.Omega_m / (2.0 * const.c**2)

    return cst_factor * ell_factor * lensing_kernel(cosmo, z, nz)

  return fn

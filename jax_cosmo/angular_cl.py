# This module contains functions to compute angular cls for various tracers
from functools import partial

import jax.numpy as np
from jax import vmap

from jax_cosmo.utils import z2a, a2z
from jax_cosmo.scipy.integrate import simps
import jax_cosmo.background as bkgrd
import jax_cosmo.power as power

@partial(vmap, in_axes=(None, 0, None, None))
def angular_cl(cosmo, ell, tracer_fn1, tracer_fn2, amin=0.002):
  """
  Computes angular Cls for the provided trace functions

  All using the Limber approximation

  TODO: support generic power spectra
  """
  @vmap
  def integrand(a):
    # Step 1: retrieve the associated comoving distance
    chi = bkgrd.radial_comoving_distance(cosmo, a)

    # Step 2: get the powers pectrum for this combination of chi and a
    k = (ell+0.5) / np.clip(chi, 1.)
    pk = power.linear_matter_power(cosmo, k, a)

    # Step 3: Get the kernels evaluated at given (ell, z)
    kernel = tracer_fn1(cosmo, ell, a2z(a)) * tracer_fn2(cosmo, ell, a2z(a))

    return np.squeeze(pk * kernel * bkgrd.dchioverda(cosmo, a)/a**2)

  # Integral over a.... would probably be better in log scale... idk
  return simps(integrand, amin, 1., 512)

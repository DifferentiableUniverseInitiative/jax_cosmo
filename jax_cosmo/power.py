# This module computes power spectra
import jax.numpy as np
import jax_cosmo.constants as const
from jax_cosmo.scipy.integrate import romb

import jax_cosmo.background as bkgrd
import jax_cosmo.transfer as tklib
#import jax_cosmo.nonlinear as nllib

__all__ = ['primordial_matter_power',
           'linear_matter_power']

def primordial_matter_power(cosmo, k):
  """ Primordial power spectrum
      Pk = k^n
  """
  return k**cosmo.n_s

def linear_matter_power(cosmo, k, a=1.0, transfer_fn=tklib.Eisenstein_Hu, **kwargs):
  r""" Computes the linear matter power spectrum.

  Parameters
  ----------
  k: array_like
      Wave number in h Mpc^{-1}

  a: array_like, optional
      Scale factor (def: 1.0)

  transfer_fn: transfer_fn(cosmo, k, **kwargs)
      Transfer function

  Returns
  -------
  pk: array_like
      Linear matter power spectrum at the specified scale
      and scale factor.

  """
  k = np.atleast_1d(k)
  a = np.atleast_1d(a)
  g = bkgrd.growth_factor(cosmo, a)
  t = transfer_fn(cosmo, k,**kwargs)

  pknorm = cosmo.sigma8**2/sigmasqr(cosmo, 8.0, transfer_fn, **kwargs)

  # if k.ndim == 1:
  #   pk = np.outer(primordial_matter_power(cosmo, k) * t**2,  g**2)
  # else:
  pk = primordial_matter_power(cosmo, k) * t**2 * g**2

  # Apply normalisation
  pk = pk*pknorm
  return pk.squeeze()

def sigmasqr(cosmo, R, transfer_fn, kmin=0.0001, kmax = 1000.0, ksteps=5, **kwargs):
  """ Computes the energy of the fluctuations within a sphere of R h^{-1} Mpc

  .. math::

     \\sigma^2(R)= \\frac{1}{2 \\pi^2} \\int_0^\\infty \\frac{dk}{k} k^3 P(k,z) W^2(kR)

  where

  .. math::

     W(kR) = \\frac{3j_1(kR)}{kR}
  """
  def int_sigma(logk):
    k = np.exp(logk)
    x = k * R
    w = 3.0*(np.sin(x) - x*np.cos(x))/(x*x*x)
    pk = transfer_fn(cosmo, k, **kwargs)**2 * primordial_matter_power(cosmo, k)
    return k * (k*w)**2 * pk
  y = romb(int_sigma, np.log10(kmin), np.log10(kmax), divmax=7)
  return 1.0/(2.0*np.pi**2.0) * y

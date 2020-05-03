# This module implements various functions for the background COSMOLOGY
import jax.numpy as np
from jax.experimental.ode import odeint

import jax_cosmo.constants as const

__all__=[
  'w',
  'f_de',
  'Esqr',
  'H',
  'Omega_m_a',
  'Omega_de_a',
  'radial_comoving_distance',
  'dchioverda',
  'transverse_comoving_distance',
  'angular_diameter_distance',
  'growth_factor'
]

def w(cosmo, a):
  r"""Dark Energy equation of state parameter using the Linder
  parametrisation.

  Parameters
  ----------
  cosmo: Cosmology
    Cosmological parameters structure

  a : array_like
      Scale factor

  Returns
  -------
  w : ndarray, or float if input scalar
      The Dark Energy equation of state parameter at the specified
      scale factor

  Notes
  -----

  The Linder parametrization :cite:`2003:Linder` for the Dark Energy
  equation of state :math:`p = w \rho` is given by:

  .. math::

      w(a) = w_0 + w (1 -a)
  """
  return cosmo.w0 + (1.0 - a) * cosmo.wa  # Equation (6) in Linder (2003)

def f_de(cosmo, a):
  r"""Evolution parameter for the Dark Energy density.

  Parameters
  ----------
  a : array_like
      Scale factor

  Returns
  -------
  f : ndarray, or float if input scalar
      The evolution parameter of the Dark Energy density as a function
      of scale factor

  Notes
  -----

  For a given parametrisation of the Dark Energy equation of state,
  the scaling of the Dark Energy density with time can be written as:

  .. math::

      \rho_{de}(a) \propto a^{f(a)}

  (see :cite:`2005:Percival`) where :math:`f(a)` is computed as
  :math:`f(a) = \frac{-3}{\ln(a)} \int_0^{\ln(a)} [1 + w(a^\prime)]
  d \ln(a^\prime)`. In the case of Linder's parametrisation for the
  dark energy in Eq. :eq:`linderParam` :math:`f(a)` becomes:

  .. math::

      f(a) = -3(1 + w_0) + 3 w \left[ \frac{a - 1}{ \ln(a) } - 1 \right]
  """
  # Just to make sure we are not diving by 0
  epsilon = np.finfo(np.float32).eps
  return -3.0*(1.0+cosmo.w0) + 3.0*cosmo.wa*((a-1.0)/np.log(a-epsilon) - 1.0)

def Esqr(cosmo, a):
  r"""Square of the scale factor dependent factor E(a) in the Hubble
  parameter.

  Parameters
  ----------
  a : array_like
      Scale factor

  Returns
  -------
  E^2 : ndarray, or float if input scalar
      Square of the scaling of the Hubble constant as a function of
      scale factor

  Notes
  -----

  The Hubble parameter at scale factor `a` is given by
  :math:`H^2(a) = E^2(a) H_o^2` where :math:`E^2` is obtained through
  Friedman's Equation (see :cite:`2005:Percival`) :

  .. math::

      E^2(a) = \Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_{de} a^{f(a)}

  where :math:`f(a)` is the Dark Energy evolution parameter computed
  by :py:meth:`.f_de`.
  """
  return cosmo.Omega_m*np.power(a, -3) + cosmo.Omega_k*np.power(a, -2) + \
      cosmo.Omega_de*np.power(a, f_de(cosmo, a))

def H(cosmo, a):
  r"""Hubble parameter [km/s/(Mpc/h)] at scale factor `a`

  Parameters
  ----------
  a : array_like
      Scale factor

  Returns
  -------
  H : ndarray, or float if input scalar
      Hubble parameter at the requested scale factor.
  """
  return const.H0 * np.sqrt(cosmo.Esqr(a))

def Omega_m_a(cosmo, a):
  r"""Matter density at scale factor `a`.

  Parameters
  ----------
  a : array_like
      Scale factor

  Returns
  -------
  Omega_m : ndarray, or float if input scalar
      Non-relativistic matter density at the requested scale factor

  Notes
  -----
  The evolution of matter density :math:`\Omega_m(a)` is given by:

  .. math::

      \Omega_m(a) = \frac{\Omega_m a^{-3}}{E^2(a)}

  see :cite:`2005:Percival` Eq. (6)
  """
  return cosmo.Omega_m * np.power(a, -3) / Esqr(cosmo, a)

def Omega_de_a(cosmo, a):
  r"""Dark Energy density at scale factor `a`.

  Parameters
  ----------
  a : array_like
      Scale factor

  Returns
  -------
  Omega_de : ndarray, or float if input scalar
      Dark Energy density at the requested scale factor

  Notes
  -----
  The evolution of Dark Energy density :math:`\Omega_{de}(a)` is given
  by:

  .. math::

      \Omega_{de}(a) = \frac{\Omega_{de} a^{f(a)}}{E^2(a)}

  where :math:`f(a)` is the Dark Energy evolution parameter computed by
  :py:meth:`.f_de` (see :cite:`2005:Percival` Eq. (6)).
  """
  return cosmo.Omega_de*np.power(a, f_de(cosmo, a))/Esqr(cosmo, a)

def radial_comoving_distance(cosmo, a):
  r"""Radial comoving distance in [Mpc/h] for a given scale factor.

  Parameters
  ----------
  a : array_like
      Scale factor

  Returns
  -------
  chi : ndarray, or float if input scalar
      Radial comoving distance corresponding to the specified scale
      factor.

  Notes
  -----
  The radial comoving distance is computed by performing the following
  integration:

  .. math::

      \chi(a) =  R_H \int_a^1 \frac{da^\prime}{{a^\prime}^2 E(a^\prime)}
  """
  def dchioverdlna(y, x, cosmo):
      xa = np.exp(x)
      return dchioverda(cosmo, xa) * xa
  # Let's turn a into an array so that we can compute several scales at once
  a = np.concatenate([np.atleast_1d(a), np.array([1.])])
  res = odeint(dchioverdlna, 0., np.log(a), cosmo)
  return np.squeeze(res[-1] - res[:-1])

def dchioverda(cosmo, a):
  r"""Derivative of the radial comoving distance with respect to the
  scale factor.

  Parameters
  ----------
  a : array_like
      Scale factor

  Returns
  -------
  dchi/da :  ndarray, or float if input scalar
      Derivative of the radial comoving distance with respect to the
      scale factor at the specified scale factor.

  Notes
  -----

  The expression for :math:`\frac{d \chi}{da}` is:

  .. math::

      \frac{d \chi}{da}(a) = \frac{R_H}{a^2 E(a)}
  """
  return const.rh/(a**2*np.sqrt(Esqr(cosmo, a)))

def transverse_comoving_distance(cosmo, a):
  r"""Transverse comoving distance in [Mpc/h] for a given scale factor.

  Parameters
  ----------
  a : array_like
      Scale factor

  Returns
  -------
  f_k : ndarray, or float if input scalar
      Transverse comoving distance corresponding to the specified
      scale factor.

  Notes
  -----
  The transverse comoving distance depends on the curvature of the
  universe and is related to the radial comoving distance through:

  .. math::

      f_k(a) = \left\lbrace
      \begin{matrix}
      R_H \frac{1}{\sqrt{\Omega_k}}\sinh(\sqrt{|\Omega_k|}\chi(a)R_H)&
          \mbox{for }\Omega_k > 0 \\
      \chi(a)&
          \mbox{for } \Omega_k = 0 \\
      R_H \frac{1}{\sqrt{\Omega_k}} \sin(\sqrt{|\Omega_k|}\chi(a)R_H)&
          \mbox{for } \Omega_k < 0
      \end{matrix}
      \right.
  """
  chi = radial_comoving_distance(cosmo, a)
  if cosmo.k < 0:      # Open universe
      return const.rh/cosmo.sqrtk*np.sinh(cosmo.sqrtk * chi/const.rh)
  elif cosmo.k > 0:    # Closed Universe
      return const.rh/cosmo.sqrtk*np.sin(cosmo.sqrtk * chi/const.rh)
  else:
      return chi

def angular_diameter_distance(cosmo, a):
  r"""Angular diameter distance in [Mpc/h] for a given scale factor.

  Parameters
  ----------
  a : array_like
      Scale factor

  Returns
  -------
  d_A : ndarray, or float if input scalar

  Notes
  -----
  Angular diameter distance is expressed in terms of the transverse
  comoving distance as:

  .. math::

      d_A(a) = a f_k(a)
  """
  return a * transverse_comoving_distance(cosmo, a)

def growth_factor(cosmo, a, amin=1e-3, eps=1e-4):
  """ Compute Growth factor at a given scale factor, normalised such
  that G(a=1) = 1.

  Parameters
  ----------
  a: array_like
    Scale factor

  amin: float
    Mininum scale factor, default 1e-3

  Returns
  -------
  G:  ndarray, or float if input scalar
      Growth factor computed at requested scale factor

  """
  a = np.atleast_1d(a)
  def D_derivs(y, x, cosmo):
      q = (2.0 - 0.5 * (Omega_m_a(cosmo, x) +
                        (1.0 + 3.0 * w(cosmo, x))
                        * Omega_de_a(cosmo, x)))/x
      r = 1.5*Omega_m_a(cosmo, x)/x/x
      return [y[1], -q * y[1] + r * y[0]]
  y0 = [amin, 1.0]
  a = np.concatenate([np.array([amin]), np.clip(a, amin+eps, 1.- eps), np.array([1.0])])
  y1, y2 = odeint(D_derivs, y0, a, cosmo)
  return y1[1:-1]/y1[-1]

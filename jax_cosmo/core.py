from collections import namedtuple
from jax.tree_util import register_pytree_node
import jax.numpy as np
from jax.experimental.ode import odeint

import jax_cosmo.constants as const
from jax_cosmo.utils import a2z, z2a

__all__ = ["Cosmology",
          "Background"]

# Let's define a cosmology as a namedtuple
Cosmology = namedtuple("Cosmology", [
   "Omega_c", # Cold dark matter density fraction.
   "Omega_b", # Baryonic matter density fraction.
   "h",       # Hubble constant divided by 100 km/s/Mpc; unitless.
   "n_s",     # Primordial scalar perturbation spectral index.
   "sigma8",  # Variance of matter density perturbations at an 8 Mpc/h scale
   "Omega_k", # Curvature density fraction.
   "w0",      # First order term of dark energy equation
   "wa",      # Second order term of dark energy equation of state
   ])
# Registering the cosmology for JAX
register_pytree_node(
    Cosmology,
    lambda xs: (tuple(xs), None),  # tell JAX how to unpack to an iterable
    lambda _, xs: Cosmology(*xs)   # tell JAX how to pack back into a Cosmology
)

class Background:

  def __init__(self, cosmo):
    """ Model for background cosmology.

    Parameters:
    -----------
    cosmo: Cosmology
        Cosmological parameters
    """
    self._parameters = cosmo

    # Setup derived parameters
    self._Omega_m = self.Omega_b + self.Omega_c
    self._Omega = 1.0 - self.Omega_k
    self._Omega_de = self._Omega - self._Omega_m

    # Sugiyama (1995, APJS, 100, 281)
    self._gamma = self.Omega_m*self.h * \
        np.exp(-self.Omega_b*(1. + np.sqrt(2.*self.h)/self.Omega_m))

    if self.Omega > 1.0:   # Closed universe
      self._k = 1.0
      self._sqrtk = np.sqrt(np.abs(self.Omega_k))
    elif self.Omega == 1.0:  # Flat universe
      self._k = 0
      self._sqrtk = 1.
    elif self.Omega < 1.0:  # Open Universe
      self._k = -1.0
      self._sqrtk = np.sqrt(np.abs(self.Omega_k))

    #############################################
    # Quantities computed from 1998:EisensteinHu
    # Provides : - k_eq   : scale of the particle horizon at equality epoch
    #            - z_eq   : redshift of equality epoch
    #            - R_eq   : ratio of the baryon to photon momentum density
    #                       at z_eq
    #            - z_d    : redshift of drag epoch
    #            - R_d    : ratio of the baryon to photon momentum density
    #                       at z_d
    #            - sh_d   : sound horizon at drag epoch
    #            - k_silk : Silk damping scale
    T_2_7_sqr = (const.tcmb/2.7)**2
    h2 = self.h**2
    w_m = self.Omega_m*h2
    w_b = self.Omega_b*h2

    self._k_eq = 7.46e-2*w_m/T_2_7_sqr / self.h     # Eq. (3) [h/Mpc]
    self._z_eq = 2.50e4*w_m/(T_2_7_sqr)**2          # Eq. (2)

    # z drag from Eq. (4)
    b1 = 0.313*np.power(w_m, -0.419)*(1.0+0.607*np.power(w_m, 0.674))
    b2 = 0.238*np.power(w_m, 0.223)
    self._z_d = 1291.0*np.power(w_m, 0.251)/(1.0+0.659*np.power(w_m, 0.828)) * \
        (1.0 + b1*np.power(w_b, b2))

    # Ratio of the baryon to photon momentum density at z_d  Eq. (5)
    self._R_d = 31.5 * w_b / (T_2_7_sqr)**2 * (1.e3/self._z_d)
    # Ratio of the baryon to photon momentum density at z_eq Eq. (5)
    self._R_eq = 31.5 * w_b / (T_2_7_sqr)**2 * (1.e3/self._z_eq)
    # Sound horizon at drag epoch in h^-1 Mpc Eq. (6)
    self._sh_d = 2.0/(3.0*self._k_eq) * np.sqrt(6.0/self._R_eq) * \
        np.log((np.sqrt(1.0 + self._R_d) + np.sqrt(self._R_eq + self._R_d)) /
            (1.0 + np.sqrt(self._R_eq)))
    # Eq. (7) but in [hMpc^{-1}]
    self._k_silk = 1.6 * np.power(w_b, 0.52) * np.power(w_m, 0.73) * \
        (1.0 + np.power(10.4*w_m, -0.95)) / self.h
    #############################################

    #############################################
    # Quantities computed from 1999:EfstathiouBond
    # Provides : - z_r   : redshift of recombination
    #            - sh_r  : sound horizon at recombination
    # z recombination from Eq. (20)
    g1 = 0.078*w_b**(-0.238) * (1.0 + 39.5*w_b**0.7630)**(-1)
    g2 = 0.56*(1.0 + 21.1*w_b**1.81)**(-1)
    a_r = 1.0/(1048.*(1.0+0.00124*w_b**(-0.738))*(1.0+g1*w_m**g2) + 1)
    self._z_r = a2z(a_r)

    # sound horizon at recombination from Eq. (18) and (19)
    a_eq = 1.0/(24185.0 * (1.6813/(1.0 + const.eta_nu)) * w_m)  # Eq. (18)
    R_eq = 30496.*w_b*a_eq  # Eq. (18)
    R_zr = 30496.*w_b*a_r   # Eq. (18)
    frac = (np.sqrt(1.0+R_zr)+np.sqrt(R_zr+R_eq))/(1.0+np.sqrt(R_eq))    # Eq. (19)
    self._sh_r = 4000.0/np.sqrt(w_b)*np.sqrt(a_eq)/np.sqrt(1.0+const.eta_nu) * \
        np.log(frac) * self.h  # Eq. (19) converted to Mpc/h
    #############################################

  def __str__(self):
    return 'FLRW Cosmology with the following parameters: \n' + \
        '    h:        ' + str(self.h) + ' \n' + \
        '    Omega_b:  ' + str(self.Omega_b) + ' \n' + \
        '    Omega_m:  ' + str(self.Omega_m) + ' \n' + \
        '    Omega_de: ' + str(self.Omega_de) + ' \n' + \
        '    w0:       ' + str(self.w0) + ' \n' + \
        '    wa:       ' + str(self.wa) + ' \n' + \
        '    n:        ' + str(self.n_s) + ' \n' + \
        '    sigma8:   ' + str(self.sigma8)


  def w(self, a):
    r"""Dark Energy equation of state parameter using the Linder
    parametrisation.

    Parameters
    ----------
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
    return self.w0 + (1.0 - a) * self.wa  # Equation (6) in Linder (2003)

  def f_de(self, a):
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
    epsilon = 0.000000001
    return -3.0*(1.0+self.w0) + 3.0*self.wa*((a-1.0)/np.log(a-epsilon) - 1.0)

  def Esqr(self, a):
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
    return self.Omega_m*np.power(a, -3) + self.Omega_k*np.power(a, -2) + \
        self.Omega_de*np.power(a, self.f_de(a))

  def H(self, a):
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
    return const.H0 * np.sqrt(self.Esqr(a))

  def Omega_m_a(self, a):
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
    return self.Omega_m * np.power(a, -3) / self.Esqr(a)

  def Omega_de_a(self, a):
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
    return self.Omega_de*np.power(a, self.f_de(a))/self.Esqr(a)

  def radial_comoving_distance(self, a):
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
    def dchioverdlna(y, x):
        xa = np.exp(x)
        return self.dchioverda(xa) * xa
    # Let's turn a into an array so that we can compute several scales at once
    a = np.concatenate([np.atleast_1d(a), np.array([1.])])
    res = odeint(dchioverdlna, 0., np.log(a))
    return np.squeeze(res[-1] - res[:-1])

  def dchioverda(self, a):
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
    return const.rh/(a**2*np.sqrt(self.Esqr(a)))

  def transverse_comoving_distance(self, a):
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
    chi = self.radial_comoving_distance(a)
    if self.k < 0:      # Open universe
        return const.rh/self.sqrtk*np.sinh(self.sqrtk * chi/const.rh)
    elif self.k > 0:    # Closed Universe
        return const.rh/self.sqrtk*np.sin(self.sqrtk * chi/const.rh)
    else:
        return chi

  def angular_diameter_distance(self, a):
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
    return a * self.transverse_comoving_distance(a)

  def growth_factor(self, a, amin=1e-3):
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
    def D_derivs(y, x):
        q = (2.0 - 0.5 * (self.Omega_m_a(x) +
                          (1.0 + 3.0 * self.w(x))
                          * self.Omega_de_a(x)))/x
        r = 1.5*self.Omega_m_a(x)/x/x
        return [y[1], -q * y[1] + r * y[0]]
    y0 = [amin, 1.0]
    a = np.concatenate([np.array([amin]), a ,np.array([1.0])])
    y1, y2 = odeint(D_derivs, y0, a)
    return y1[1:-1]/y1[-1]

  @property
  def sh_d(self):
    r"""
    Sound horizon at drag epoch in Mpc/h

    Computed from Equation (6) in :cite:`1998:EisensteinHu` :

    .. math ::

        r_s(z_d) = \frac{2}{3 k_{eq}} \sqrt{ \frac{6}{R_{eq}} } \ln \frac{ \sqrt{1 + R_d} + \sqrt{R_d + R_{eq}}}{1 + \sqrt{R_{eq}}}

    where :math:`R_d` and :math:`R_{eq}` are respectively the ratio of baryon to photon momentum density at drag epoch and equality epoch (see Equation (5) in :cite:`1998:EisensteinHu`)
    and :math:`k_{eq}` is the scale of the scale of the particle horizon at equality epoch.
    """
    return self._sh_d

  @property
  def sh_r(self):
    r"""
    Sound horizon at recombination in Mpc/h

    Computed from Equation (19) in :cite:`1999:EfstathiouBond` :

    .. math ::

        r_s(z_r) = \frac{4000 \sqrt{a_{equ}}}{\sqrt{\omega_b (1 + \eta_\nu)}} \ln \frac{ \sqrt{1 + R_r} + \sqrt{R_r + R_{eq}}}{1 + \sqrt{R_{eq}}}

    where :math:`R_r` and :math:`R_{eq}` are respectively the ratio of baryon to photon momentum density at recombination epoch and equality epoch (see Equation (18) in :cite:`1999:EfstathiouBond`)
    and :math:`\eta_{\nu}` denotes the relative densities of massless neutrinos and photons.
    """
    return self._sh_r

  @property
  def parameters(self):
    return self._parameters

  # Cosmological parameters, base and derived
  @property
  def Omega(self):
    return self._Omega

  @property
  def Omega_b(self):
    return self._parameters.Omega_b

  @property
  def Omega_c(self):
    return self._parameters.Omega_c

  @property
  def Omega_m(self):
    return self._Omega_m

  @property
  def Omega_de(self):
    return self._Omega_de

  @property
  def Omega_k(self):
    return self._parameters.Omega_k

  @property
  def k(self):
    return self._k

  @property
  def sqrtk(self):
    return self._sqrtk

  @property
  def h(self):
    return self._parameters.h

  @property
  def w0(self):
    return self._parameters.w0

  @property
  def wa(self):
    return self._parameters.wa

  @property
  def n_s(self):
    return self._parameters.n_s

  @property
  def sigma8(self):
    return self._parameters.sigma8

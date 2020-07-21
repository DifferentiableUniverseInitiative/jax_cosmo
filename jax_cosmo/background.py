# This module implements various functions for the background COSMOLOGY
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np

import jax_cosmo.constants as const
from jax_cosmo.scipy.interpolate import interp
from jax_cosmo.scipy.ode import odeint

__all__ = [
    "w",
    "f_de",
    "Esqr",
    "H",
    "Omega_m_a",
    "Omega_de_a",
    "radial_comoving_distance",
    "dchioverda",
    "transverse_comoving_distance",
    "angular_diameter_distance",
    "growth_factor",
    "growth_rate",
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
    return -3.0 * (1.0 + cosmo.w0) + 3.0 * cosmo.wa * (
        (a - 1.0) / np.log(a - epsilon) - 1.0
    )


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
    return (
        cosmo.Omega_m * np.power(a, -3)
        + cosmo.Omega_k * np.power(a, -2)
        + cosmo.Omega_de * np.power(a, f_de(cosmo, a))
    )


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
    return const.H0 * np.sqrt(Esqr(cosmo, a))


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
    return cosmo.Omega_de * np.power(a, f_de(cosmo, a)) / Esqr(cosmo, a)


def radial_comoving_distance(cosmo, a, log10_amin=-3, steps=256):
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
    # Check if distances have already been computed
    if not "background.radial_comoving_distance" in cosmo._workspace.keys():
        # Compute tabulated array
        atab = np.logspace(log10_amin, 0.0, steps)

        def dchioverdlna(y, x):
            xa = np.exp(x)
            return dchioverda(cosmo, xa) * xa

        chitab = odeint(dchioverdlna, 0.0, np.log(atab))
        # np.clip(- 3000*np.log(atab), 0, 10000)#odeint(dchioverdlna, 0., np.log(atab), cosmo)
        chitab = chitab[-1] - chitab

        cache = {"a": atab, "chi": chitab}
        cosmo._workspace["background.radial_comoving_distance"] = cache
    else:
        cache = cosmo._workspace["background.radial_comoving_distance"]

    a = np.atleast_1d(a)
    # Return the results as an interpolation of the table
    return np.clip(interp(a, cache["a"], cache["chi"]), 0.0)


def a_of_chi(cosmo, chi):
    r""" Computes the scale factor for corresponding (array) of radial comoving
    distance by reverse linear interpolation.

    Parameters:
    -----------
    cosmo: Cosmology
      Cosmological parameters

    chi: array-like
      radial comoving distance to query.

    Returns:
    --------
    a : array-like
      Scale factors corresponding to requested distances
    """
    # Check if distances have already been computed, force computation otherwise
    if not "background.radial_comoving_distance" in cosmo._workspace.keys():
        radial_comoving_distance(cosmo, 1.0)
    cache = cosmo._workspace["background.radial_comoving_distance"]
    chi = np.atleast_1d(chi)
    return interp(chi, cache["chi"], cache["a"])


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
    return const.rh / (a ** 2 * np.sqrt(Esqr(cosmo, a)))


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
    if cosmo.k < 0:  # Open universe
        return const.rh / cosmo.sqrtk * np.sinh(cosmo.sqrtk * chi / const.rh)
    elif cosmo.k > 0:  # Closed Universe
        return const.rh / cosmo.sqrtk * np.sin(cosmo.sqrtk * chi / const.rh)
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


def growth_factor(cosmo, a):
    """ Compute linear growth factor D(a) at a given scale factor,
    normalized such that D(a=1) = 1.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    D:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor

    Notes
    -----
    The growth computation will depend on the cosmology parametrization, for
    instance if the $\gamma$ parameter is defined, the growth will be computed
    assuming the $f = \Omega^\gamma$ growth rate, otherwise the usual ODE for
    growth will be solved.
    """
    if cosmo._flags["gamma_growth"]:
        return _growth_factor_gamma(cosmo, a)
    else:
        return _growth_factor_ODE(cosmo, a)


def growth_rate(cosmo, a):
    """ Compute growth rate dD/dlna at a given scale factor.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    f:  ndarray, or float if input scalar
        Growth rate computed at requested scale factor

    Notes
    -----
    The growth computation will depend on the cosmology parametrization, for
    instance if the $\gamma$ parameter is defined, the growth will be computed
    assuming the $f = \Omega^\gamma$ growth rate, otherwise the usual ODE for
    growth will be solved.

    The LCDM approximation to the growth rate :math:`f_{\gamma}(a)` is given by:

    .. math::

        f_{\gamma}(a) = \Omega_m^{\gamma} (a)

     with :math: `\gamma` in LCDM, given approximately by:
     .. math::

        \gamma = 0.55

    see :cite:`2019:Euclid Preparation VII, eqn.32`
    """
    if cosmo._flags["gamma_growth"]:
        return _growth_rate_gamma(cosmo, a)
    else:
        return _growth_rate_ODE(cosmo, a)


def _growth_factor_ODE(cosmo, a, log10_amin=-3, steps=128, eps=1e-4):
    """ Compute linear growth factor D(a) at a given scale factor,
    normalised such that D(a=1) = 1.

    Parameters
    ----------
    a: array_like
      Scale factor

    amin: float
      Mininum scale factor, default 1e-3

    Returns
    -------
    D:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor
    """
    # Check if growth has already been computed
    if not "background.growth_factor" in cosmo._workspace.keys():
        # Compute tabulated array
        atab = np.logspace(log10_amin, 0.0, steps)

        def D_derivs(y, x):
            q = (
                2.0
                - 0.5
                * (
                    Omega_m_a(cosmo, x)
                    + (1.0 + 3.0 * w(cosmo, x)) * Omega_de_a(cosmo, x)
                )
            ) / x
            r = 1.5 * Omega_m_a(cosmo, x) / x / x
            return np.array([y[1], -q * y[1] + r * y[0]])

        y0 = np.array([atab[0], 1.0])
        y = odeint(D_derivs, y0, atab)
        y1 = y[:, 0]
        gtab = y1 / y1[-1]
        # To transform from dD/da to dlnD/dlna: dlnD/dlna = a / D dD/da
        ftab = y[:, 1] / y1[-1] * atab / gtab

        cache = {"a": atab, "g": gtab, "f": ftab}
        cosmo._workspace["background.growth_factor"] = cache
    else:
        cache = cosmo._workspace["background.growth_factor"]
    return np.clip(interp(a, cache["a"], cache["g"]), 0.0, 1.0)


def _growth_rate_ODE(cosmo, a):
    """ Compute growth rate dD/dlna at a given scale factor by solving the linear
    growth ODE.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    f:  ndarray, or float if input scalar
        Growth rate computed at requested scale factor
    """
    # Check if growth has already been computed, if not, compute it
    if not "background.growth_factor" in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
    cache = cosmo._workspace["background.growth_factor"]
    return interp(a, cache["a"], cache["f"])


def _growth_factor_gamma(cosmo, a, log10_amin=-3, steps=128):
    r""" Computes growth factor by integrating the growth rate provided by the
    \gamma parametrization. Normalized such that D( a=1) =1

    Parameters
    ----------
    a: array_like
      Scale factor

    amin: float
      Mininum scale factor, default 1e-3

    Returns
    -------
    D:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor

    """
    # Check if growth has already been computed, if not, compute it
    if not "background.growth_factor" in cosmo._workspace.keys():
        # Compute tabulated array
        atab = np.logspace(log10_amin, 0.0, steps)

        def integrand(y, loga):
            xa = np.exp(loga)
            return _growth_rate_gamma(cosmo, xa)

        gtab = np.exp(odeint(integrand, np.log(atab[0]), np.log(atab)))
        gtab = gtab / gtab[-1]  # Normalize to a=1.
        cache = {"a": atab, "g": gtab}
        cosmo._workspace["background.growth_factor"] = cache
    else:
        cache = cosmo._workspace["background.growth_factor"]
    return np.clip(interp(a, cache["a"], cache["g"]), 0.0, 1.0)


def _growth_rate_gamma(cosmo, a):
    r"""Growth rate approximation at scale factor `a`.

    Parameters
    ----------
    cosmo: `Cosmology`
        Cosmology object

    a : array_like
        Scale factor

    Returns
    -------
    f_gamma : ndarray, or float if input scalar
        Growth rate approximation at the requested scale factor

    Notes
    -----
    The LCDM approximation to the growth rate :math:`f_{\gamma}(a)` is given by:

    .. math::

        f_{\gamma}(a) = \Omega_m^{\gamma} (a)

     with :math: `\gamma` in LCDM, given approximately by:
     .. math::

        \gamma = 0.55

    see :cite:`2019:Euclid Preparation VII, eqn.32`
    """
    return Omega_m_a(cosmo, a) ** cosmo.gamma

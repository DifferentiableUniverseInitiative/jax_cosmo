"""This module implements various functions for the cosmological background
and linear perturbations.

"""
import jax.numpy as np
from jax import lax

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
    "a_of_chi",
    "dchioverda",
    "transverse_comoving_distance",
    "angular_diameter_distance",
    "growth_factor",
    "growth_rate",
]


def w(cosmo, a):
    r"""Dark energy equation of state parameter using the Linder
    parametrisation.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    w : ndarray, or float if input scalar
        Dark energy equation of state parameters at specified scale factors.

    Notes
    -----
    The Linder parametrization :cite:`2003:Linder` for the dark energy
    equation of state :math:`p = w \rho` is given by:

    .. math::

        w(a) = w_0 + w_a (1 - a)

    """
    return cosmo.w0 + (1.0 - a) * cosmo.wa  # Equation (6) in Linder (2003)


def f_de(cosmo, a):
    r"""Evolution parameter for the dark energy density.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    f : ndarray, or float if input scalar
        Dark energy density evolution parameters at specified scale factors.

    Notes
    -----
    For a given parametrisation of the dark energy equation of state,
    the scaling of the dark energy density with time can be written as:

    .. math::

        \rho_{de}(a) = \rho_{de}(a=1) e^{f(a)}

    (see :cite:`2005:Percival` and note the difference in the exponent base
    in the parametrizations) where :math:`f(a)` is computed as
    :math:`f(a) = -3 \int_0^{\ln(a)} [1 + w(a')] d \ln(a')`.
    In the case of Linder's parametrisation for the dark energy
    in Eq. :eq:`linderParam` :math:`f(a)` becomes:

    .. math::

        f(a) = -3 (1 + w_0 + w_a) \ln(a) + 3 w_a (a - 1)

    """
    return -3.0 * (1.0 + cosmo.w0 + cosmo.wa) * np.log(a) + 3.0 * cosmo.wa * (a - 1.0)


def Esqr(cosmo, a):
    r"""Squared time scaling factors E(a) of the Hubble expansion.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    E^2 : ndarray, or float if input scalar
        Squared scaling of the Hubble expansion at specified scale factors.

    Notes
    -----
    The Hubble parameter at scale factor `a` is given by
    :math:`H^2(a) = E^2(a) H_0^2` where :math:`E^2` is obtained through
    Friedman's Equation (see :cite:`2005:Percival`) :

    .. math::

        E^2(a) = \Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_{de} e^{f(a)}

    where :math:`f(a)` is the dark energy evolution parameter computed
    by :py:meth:`.f_de`.

    """
    return (
        cosmo.Omega_m * np.power(a, -3)
        + cosmo.Omega_k * np.power(a, -2)
        + cosmo.Omega_de * np.exp(f_de(cosmo, a))
    )


def H(cosmo, a):
    r"""Hubble expansion rate [km/s/(Mpc/h)] at given scale factors.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    H : ndarray, or float if input scalar
        Hubble parameters at specified scale factors.

    """
    return const.H0 * np.sqrt(Esqr(cosmo, a))


def Omega_m_a(cosmo, a):
    r"""Matter density at given scale factors.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    Omega_m : ndarray, or float if input scalar
        Non-relativistic matter density at specified scale factors.

    Notes
    -----
    The evolution of matter density :math:`\Omega_m(a)` is given by:

    .. math::

        \Omega_m(a) = \frac{\Omega_m a^{-3}}{E^2(a)}

    see :cite:`2005:Percival` Eq. (6)

    """
    return cosmo.Omega_m * np.power(a, -3) / Esqr(cosmo, a)


def Omega_de_a(cosmo, a):
    r"""Dark energy density at given scale factors.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    Omega_de : ndarray, or float if input scalar
        Dark energy density at specified scale factors.

    Notes
    -----
    The evolution of dark energy density :math:`\Omega_{de}(a)` is given
    by:

    .. math::

        \Omega_{de}(a) = \frac{\Omega_{de} e^{f(a)}}{E^2(a)}

    where :math:`f(a)` is the dark energy evolution parameter computed by
    :py:meth:`.f_de` (see :cite:`2005:Percival` Eq. (6)).

    """
    return cosmo.Omega_de * np.exp(f_de(cosmo, a)) / Esqr(cosmo, a)


def radial_comoving_distance(cosmo, a, log10_amin=-3, steps=256):
    r"""Radial comoving distances in [Mpc/h] at given scale factors.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    cosmo : Cosmology
        Cosmological parameters with cached computations.
    chi : ndarray, or float if input scalar
        Radial comoving distances at specified scale factors.

    Notes
    -----
    The radial comoving distances is computed by performing the following
    integration:

    .. math::

        \chi(a) =  R_H \int_a^1 \frac{da^\prime}{{a^\prime}^2 E(a^\prime)}

    """
    # Check if distances have already been computed
    key = "background.radial_comoving_distance"
    if not cosmo.is_cached(key):
        # Compute tabulated array
        atab = np.logspace(log10_amin, 0.0, steps)

        def dchioverdlna(y, x):
            xa = np.exp(x)
            return dchioverda(cosmo, xa) * xa

        chitab = odeint(dchioverdlna, 0.0, np.log(atab))
        # np.clip(- 3000*np.log(atab), 0, 10000)#odeint(dchioverdlna, 0., np.log(atab), cosmo)
        chitab = chitab[-1] - chitab

        value = {"a": atab, "chi": chitab}
        cosmo = cosmo.cache_set(key, value)
    else:
        value = cosmo.cache_get(key)

    a = np.atleast_1d(a)
    # Return the results as an interpolation of the table
    chi = np.clip(interp(a, value["a"], value["chi"]), 0.0)
    return cosmo, chi


def a_of_chi(cosmo, chi):
    r"""Computes the scale factors at given radial comoving distances by
    reverse linear interpolation.

    Parameters:
    -----------
    cosmo : Cosmology
        Cosmological parameters.
    chi : array-like
        Radial comoving distances to query.

    Returns:
    --------
    cosmo : Cosmology
        Cosmological parameters with cached computations.
    a : array-like
        Scale factors at specified distances.

    """
    # Check if distances have already been computed, force computation otherwise
    key = "background.radial_comoving_distance"
    if not cosmo.is_cached(key):
        cosmo, _ = radial_comoving_distance(cosmo, 1.0)
    value = cosmo.cache_get(key)

    chi = np.atleast_1d(chi)
    a = interp(chi, value["chi"], value["a"])
    return cosmo, a


def dchioverda(cosmo, a):
    r"""Derivative of the radial comoving distances with respect to the
    scale factors.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    dchi/da : ndarray, or float if input scalar
        Derivative of the radial comoving distances with respect to the
        scale factors at specified scale factors.

    Notes
    -----
    The expression for :math:`\frac{d \chi}{da}` is:

    .. math::

        \frac{d \chi}{da}(a) = \frac{R_H}{a^2 E(a)}

    """
    return const.rh / (a ** 2 * np.sqrt(Esqr(cosmo, a)))


def transverse_comoving_distance(cosmo, a):
    r"""Transverse comoving distances in [Mpc/h] for given scale factors.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    cosmo : Cosmology
        Cosmological parameters with cached computations.
    f_k : ndarray, or float if input scalar
        Transverse comoving distances at specified scale factors.

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

    def open_universe(chi):
        return const.rh / cosmo.sqrtk * np.sinh(cosmo.sqrtk * chi / const.rh)

    def flat_universe(chi):
        return chi

    def close_universe(chi):
        return const.rh / cosmo.sqrtk * np.sin(cosmo.sqrtk * chi / const.rh)

    branches = (open_universe, flat_universe, close_universe)

    cosmo, chi = radial_comoving_distance(cosmo, a)

    f_k = lax.switch(cosmo.k + 1, branches, chi)
    return cosmo, f_k


def angular_diameter_distance(cosmo, a):
    r"""Angular diameter distances in [Mpc/h] for given scale factors.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    cosmo : Cosmology
        Cosmological parameters with cached computations.
    d_A : ndarray, or float if input scalar
        Angular diameter distances at specified scale factors.

    Notes
    -----
    Angular diameter distance is expressed in terms of the transverse
    comoving distance as:

    .. math::

        d_A(a) = a f_k(a)

    """
    cosmo, f_k = transverse_comoving_distance(cosmo, a)
    d_A = a * f_k
    return cosmo, d_A


def growth_factor(cosmo, a):
    r"""Compute linear growth factors :math:`D(a)` at given scale factors,
    normalized such that :math:`D(a=1) = 1`.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmology parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    cosmo : Cosmology
        Cosmological parameters with cached computations.
    D : ndarray, or float if input scalar
        Growth factors at specified scale factors.

    Notes
    -----
    The growth computation depends on the cosmology parametrization:
    if the :math:`\gamma` parameter is defined, the growth history is computed
    assuming the growth rate :math:`f = \Omega_m(a)^\gamma`, otherwise the
    usual ODE for growth will be solved.

    """
    if cosmo.gamma is not None:
        cosmo, D = _growth_factor_gamma(cosmo, a)
    else:
        cosmo, D = _growth_factor_ODE(cosmo, a)
    return cosmo, D


def growth_rate(cosmo, a):
    r"""Compute growth rates :math:`dD/d\ln\ a` at given scale factors.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmology parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    cosmo : Cosmology
        Cosmological parameters with cached computations.
    f : ndarray, or float if input scalar
        Growth rate at specified scale factors.

    Notes
    -----
    The growth computation depends on the cosmology parametrization:
    if the :math:`\gamma` parameter is defined, the growth history is computed
    assuming the growth rate :math:`f = \Omega_m(a)^\gamma`, otherwise the
    usual ODE for growth will be solved.

    The LCDM approximation to the growth rate :math:`f_{\gamma}(a)` is given by:

    .. math::

        f_{\gamma}(a) = \Omega_m^{\gamma} (a)

     with :math:`\gamma` in LCDM, given approximately by:
     .. math::

        \gamma = 0.55

    see :cite:`2019:Euclid Preparation VII, eqn.32`

    """
    if cosmo.gamma is not None:
        f = _growth_rate_gamma(cosmo, a)
    else:
        cosmo, f = _growth_rate_ODE(cosmo, a)
    return cosmo, f


def _growth_factor_ODE(cosmo, a, log10_amin=-3, steps=128, eps=1e-4):
    r"""Compute linear growth factors :math:`D(a)` at given scale factors,
    normalised such that :math:`D(a=1) = 1`.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.
    amin : float
        Mininum scale factor, default 1e-3.

    Returns
    -------
    cosmo : Cosmology
        Cosmological parameters with cached computations.
    D : ndarray, or float if input scalar
        Growth factors at specified scale factors.

    """
    # Check if growth has already been computed
    key = "background.growth_factor"
    if not cosmo.is_cached(key):
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

        value = {"a": atab, "g": gtab, "f": ftab}
        cosmo = cosmo.cache_set(key, value)
    else:
        value = cosmo.cache_get(key)

    D = np.clip(interp(a, value["a"], value["g"]), 0.0, 1.0)
    return cosmo, D


def _growth_rate_ODE(cosmo, a):
    r"""Compute growth rates :math:`dD/d\ln\ a` at given scale factors by
    solving the linear growth ODE.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmology parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    cosmo : Cosmology
        Cosmological parameters with cached computations.
    f : ndarray, or float if input scalar
        Growth rates at specified scale factors.

    """
    # Check if growth has already been computed, if not, compute it
    key = "background.growth_factor"
    if not cosmo.is_cached(key):
        cosmo, _ = _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
    value = cosmo.cache_get(key)

    f = interp(a, value["a"], value["f"])
    return cosmo, f


def _growth_factor_gamma(cosmo, a, log10_amin=-3, steps=128):
    r"""Growth factors by integrating the :math:`\gamma`-parametrized growth
    rates, normalized such that :math:`D(a=1) = 1`.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmological parameters.
    a : array_like
        Scale factors.
    amin : float
        Mininum scale factor, default 1e-3

    Returns
    -------
    cosmo : Cosmology
        Cosmological parameters with cached computations.
    D : ndarray, or float if input scalar
        Growth factors at specified scale factors.

    """
    # Check if growth has already been computed, if not, compute it
    key = "background.growth_factor"
    if not cosmo.is_cached(key):
        # Compute tabulated array
        atab = np.logspace(log10_amin, 0.0, steps)

        def integrand(y, loga):
            xa = np.exp(loga)
            return _growth_rate_gamma(cosmo, xa)

        gtab = np.exp(odeint(integrand, np.log(atab[0]), np.log(atab)))
        gtab = gtab / gtab[-1]  # Normalize to a=1.
        value = {"a": atab, "g": gtab}
        cosmo = cosmo.cache_set(key, value)
    else:
        value = cosmo.cache_get(key)

    D = np.clip(interp(a, value["a"], value["g"]), 0.0, 1.0)
    return cosmo, D


def _growth_rate_gamma(cosmo, a):
    r"""Growth rate approximation at given scale factors.

    Parameters
    ----------
    cosmo : Cosmology
        Cosmology parameters.
    a : array_like
        Scale factors.

    Returns
    -------
    f_gamma : ndarray, or float if input scalar
        Growth rate approximation at specified scale factors.

    Notes
    -----
    The LCDM approximation to the growth rate :math:`f_{\gamma}(a)` is given by:

    .. math::

        f_{\gamma}(a) = \Omega_m^{\gamma} (a)

     with :math:`\gamma` in LCDM, given approximately by:
     .. math::

        \gamma = 0.55

    see :cite:`2019:Euclid Preparation VII, eqn.32`

    """
    return Omega_m_a(cosmo, a) ** cosmo.gamma

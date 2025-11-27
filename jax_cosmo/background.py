# This module implements various functions for the background COSMOLOGY
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
    "dchioverda",
    "transverse_comoving_distance",
    "angular_diameter_distance",
    "growth_factor",
    "growth_rate",
    "growth_factor_second",
    "growth_rate_second",
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

        w(a) = w_0 + w_a (1 - a)
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

        E^2(a) = \Omega_m a^{-3} + \Omega_k a^{-2} + \Omega_{de} e^{f(a)}

    where :math:`f(a)` is the Dark Energy evolution parameter computed
    by :py:meth:`.f_de`.
    """
    return (
        cosmo.Omega_m * np.power(a, -3)
        + cosmo.Omega_k * np.power(a, -2)
        + cosmo.Omega_de * np.exp(f_de(cosmo, a))
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

        \Omega_{de}(a) = \frac{\Omega_{de} e^{f(a)}}{E^2(a)}

    where :math:`f(a)` is the Dark Energy evolution parameter computed by
    :py:meth:`.f_de` (see :cite:`2005:Percival` Eq. (6)).
    """
    return cosmo.Omega_de * np.exp(f_de(cosmo, a)) / Esqr(cosmo, a)


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
    r"""Computes the scale factor for corresponding (array) of radial comoving
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
    return const.rh / (a**2 * np.sqrt(Esqr(cosmo, a)))


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
    index = cosmo.k + 1

    def open_universe(chi):
        return const.rh / cosmo.sqrtk * np.sinh(cosmo.sqrtk * chi / const.rh)

    def flat_universe(chi):
        return chi

    def close_universe(chi):
        return const.rh / cosmo.sqrtk * np.sin(cosmo.sqrtk * chi / const.rh)

    branches = (open_universe, flat_universe, close_universe)

    chi = radial_comoving_distance(cosmo, a)

    return lax.switch(cosmo.k + 1, branches, chi)


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
    r"""Compute linear growth factor D(a) at a given scale factor,
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
    r"""Compute growth rate dD/dlna at a given scale factor.

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


def growth_factor_second(cosmo, a):
    r"""Compute second-order growth factor D2(a) at a given scale factor,
    normalized such that D2(a=1) = 1.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    D2:  ndarray, or float if input scalar
        Second-order growth factor computed at requested scale factor

    Notes
    -----
    The second-order growth factor satisfies the ODE:

    .. math::

        D_2'' + q D_2' - r D_2 = -r D_1^2

    where q and r are the same coefficients as in the first-order equation.
    The second-order growth is important for 2LPT (second-order Lagrangian
    perturbation theory) calculations.

    Note: gamma parametrization is not supported for second-order growth.
    """
    if cosmo._flags["gamma_growth"]:
        raise NotImplementedError(
            "Gamma growth parametrization is not implemented for second-order growth. "
            "Use a cosmology without gamma_growth flag."
        )
    return _growth_factor_second_ODE(cosmo, a)


def growth_rate_second(cosmo, a):
    r"""Compute second-order growth rate f2 = d ln(D2)/d ln(a) at a given scale factor.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    f2:  ndarray, or float if input scalar
        Second-order growth rate computed at requested scale factor

    Notes
    -----
    The second-order growth rate is defined as:

    .. math::

        f_2(a) = \frac{d \ln D_2}{d \ln a}

    Note: gamma parametrization is not supported for second-order growth.
    """
    if cosmo._flags["gamma_growth"]:
        raise NotImplementedError(
            "Gamma growth parametrization is not implemented for second-order growth. "
            "Use a cosmology without gamma_growth flag."
        )
    return _growth_rate_second_ODE(cosmo, a)


def _growth_factor_ODE(cosmo, a, log10_amin=-3, steps=128, eps=1e-4):
    """Compute linear growth factor D(a) at a given scale factor,
    normalised such that D(a=1) = 1.

    This function also computes second-order growth factors simultaneously
    by solving the coupled ODE system.

    Parameters
    ----------
    a: array_like
      Scale factor

    log10_amin: float
      Log10 of minimum scale factor, default -3

    steps: int
      Number of integration steps, default 128

    eps: float
      Small regularization parameter (unused, kept for API compatibility)

    Returns
    -------
    D:  ndarray, or float if input scalar
        Growth factor computed at requested scale factor
    """
    # Check if growth has already been computed
    if "background.growth_factor" not in cosmo._workspace.keys():
        # Compute tabulated array
        atab = np.logspace(log10_amin, 0.0, steps)

        def D_derivs(y, x):
            """Coupled ODE system for first and second order growth factors.

            State vector y has shape (2, 2):
              y[0] = [D1, D2]     (growth factors)
              y[1] = [dD1/da, dD2/da]  (derivatives)
            """
            q = (
                2.0
                - 0.5
                * (
                    Omega_m_a(cosmo, x)
                    + (1.0 + 3.0 * w(cosmo, x)) * Omega_de_a(cosmo, x)
                )
            ) / x
            r = 1.5 * Omega_m_a(cosmo, x) / x / x

            g1, g2 = y[0]
            f1, f2 = y[1]

            # First order: D1'' + q*D1' - r*D1 = 0
            dy1da = [f1, -q * f1 + r * g1]
            # Second order: D2'' + q*D2' - r*D2 = -r*D1^2
            dy2da = [f2, -q * f2 + r * g2 - r * g1**2]

            return np.array([[dy1da[0], dy2da[0]], [dy1da[1], dy2da[1]]])

        # Initial conditions at early times (matter-dominated era)
        # D1 ~ a, D2 ~ -3/7 * a^2
        y0 = np.array([[atab[0], -3.0 / 7 * atab[0] ** 2], [1.0, -6.0 / 7 * atab[0]]])
        y = odeint(D_derivs, y0, atab)

        # Compute second derivatives for h and h2
        dyda2 = D_derivs(np.transpose(y, (1, 2, 0)), atab)
        dyda2 = np.transpose(dyda2, (2, 0, 1))

        # Extract and normalize first order growth
        y1 = y[:, 0, 0]
        gtab = y1 / y1[-1]

        # Extract and normalize second order growth
        y2 = y[:, 0, 1]
        g2tab = y2 / y2[-1]

        # Growth rates: transform from dD/da to dlnD/dlna = a/D * dD/da
        ftab = y[:, 1, 0] / y1[-1] * atab / gtab
        f2tab = y[:, 1, 1] / y2[-1] * atab / g2tab

        # Second derivatives (normalized)
        htab = dyda2[:, 1, 0] / y1[-1] * atab / gtab
        h2tab = dyda2[:, 1, 1] / y2[-1] * atab / g2tab

        cache = {
            "a": atab,
            "g": gtab,
            "f": ftab,
            "h": htab,
            "g2": g2tab,
            "f2": f2tab,
            "h2": h2tab,
        }
        cosmo._workspace["background.growth_factor"] = cache
    else:
        cache = cosmo._workspace["background.growth_factor"]

    return np.clip(interp(a, cache["a"], cache["g"]), 0.0, 1.0)


def _growth_rate_ODE(cosmo, a):
    """Compute growth rate dD/dlna at a given scale factor by solving the linear
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


def _growth_factor_second_ODE(cosmo, a):
    """Compute second-order growth factor D2(a) by solving the coupled ODE system.

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    D2:  ndarray, or float if input scalar
        Second-order growth factor computed at requested scale factor
    """
    # Ensure growth factors have been computed (this populates g2)
    if "background.growth_factor" not in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
    cache = cosmo._workspace["background.growth_factor"]
    return np.clip(interp(a, cache["a"], cache["g2"]), 0.0, 1.0)


def _growth_rate_second_ODE(cosmo, a):
    """Compute second-order growth rate f2 = d ln(D2)/d ln(a).

    Parameters
    ----------
    cosmo: `Cosmology`
      Cosmology object

    a: array_like
      Scale factor

    Returns
    -------
    f2:  ndarray, or float if input scalar
        Second-order growth rate computed at requested scale factor
    """
    # Ensure growth factors have been computed (this populates f2)
    if "background.growth_factor" not in cosmo._workspace.keys():
        _growth_factor_ODE(cosmo, np.atleast_1d(1.0))
    cache = cosmo._workspace["background.growth_factor"]
    return interp(a, cache["a"], cache["f2"])


def _growth_factor_gamma(cosmo, a, log10_amin=-3, steps=128):
    r"""Computes growth factor by integrating the growth rate provided by the
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

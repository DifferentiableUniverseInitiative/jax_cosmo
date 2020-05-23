# This module computes power spectra
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax
import jax.numpy as np

import jax_cosmo.background as bkgrd
import jax_cosmo.constants as const
import jax_cosmo.transfer as tklib
from jax_cosmo.scipy.integrate import romb
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.scipy.interpolate import interp

__all__ = ["primordial_matter_power", "linear_matter_power", "nonlinear_matter_power"]


def primordial_matter_power(cosmo, k):
    """ Primordial power spectrum
        Pk = k^n
    """
    return k ** cosmo.n_s


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
    t = transfer_fn(cosmo, k, **kwargs)

    pknorm = cosmo.sigma8 ** 2 / sigmasqr(cosmo, 8.0, transfer_fn, **kwargs)

    pk = primordial_matter_power(cosmo, k) * t ** 2 * g ** 2

    # Apply normalisation
    pk = pk * pknorm
    return pk.squeeze()


def sigmasqr(cosmo, R, transfer_fn, kmin=0.0001, kmax=1000.0, ksteps=5, **kwargs):
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
        w = 3.0 * (np.sin(x) - x * np.cos(x)) / (x * x * x)
        pk = transfer_fn(cosmo, k, **kwargs) ** 2 * primordial_matter_power(cosmo, k)
        return k * (k * w) ** 2 * pk

    y = romb(int_sigma, np.log10(kmin), np.log10(kmax), divmax=7)
    return 1.0 / (2.0 * np.pi ** 2.0) * y


def linear(cosmo, k, a, transfer_fn):
    """Linear matter power spectrum
    """
    return linear_matter_power(cosmo, k, a, transfer_fn)


def _halofit_parameters(cosmo, a, transfer_fn):
    r""" Computes the non linear scale,
         effective spectral index,
         spectral curvature
    """
    # Step 1: Finding the non linear scale for which sigma(R)=1
    # That's our search range for the non linear scale
    r = np.logspace(-3, 1, 256)

    @jax.vmap
    def R_nl(a):
        def int_sigma(logk):
            k = np.exp(logk)
            y = np.outer(k, r)
            pk = linear_matter_power(cosmo, k, transfer_fn=transfer_fn)
            g = bkgrd.growth_factor(cosmo, np.atleast_1d(a))
            return (
                np.expand_dims(pk * k ** 3, axis=1)
                * np.exp(-(y ** 2))
                / (2.0 * np.pi ** 2)
                * g ** 2
            )

        sigma = simps(int_sigma, np.log(1e-4), np.log(1e4), 256)
        root = interp(np.atleast_1d(1.0), sigma, r)
        return root

    # Compute non linear scale
    k_nl = 1.0 / R_nl(np.atleast_1d(a)).squeeze()

    # Step 2: Retrieve the spectral index and spectral curvature
    def integrand(logk):
        k = np.exp(logk)
        y = np.outer(k, 1.0 / k_nl)
        pk = linear_matter_power(cosmo, k, transfer_fn=transfer_fn)
        g = np.expand_dims(bkgrd.growth_factor(cosmo, np.atleast_1d(a)), 0)
        res = (
            np.expand_dims(pk * k ** 3, axis=1)
            * np.exp(-(y ** 2))
            * g ** 2
            / (2.0 * np.pi ** 2)
        )
        dneff_dlogk = 2 * res * y ** 2
        dC_dlogk = 4 * res * (y ** 2 - y ** 4)
        return np.stack([dneff_dlogk, dC_dlogk], axis=1)

    res = simps(integrand, np.log(1e-4), np.log(1e4), 256)

    n_eff = res[0] - 3.0
    C = res[0] ** 2 + res[1]

    return k_nl, n_eff, C


def halofit(cosmo, k, a, transfer_fn, prescription="takahashi2012"):
    r""" Computes the non linear halofit correction to the matter power spectrum.

    Parameters
    ----------
    k: array_like
        Wave number in h Mpc^{-1}

    a: array_like, optional
        Scale factor (def: 1.0)

    prescription: str, optional
        Either 'smith2003' or 'takahashi2012'

    Returns
    -------
    pk: array_like
        Non linear matter power spectrum at the specified scale
        and scale factor.

    Notes
    -----
    The non linear corrections are implemented following :cite:`2003:smith`

    """
    a = np.atleast_1d(a)

    # Compute the linear power spectrum
    pklin = linear_matter_power(cosmo, k, a, transfer_fn)

    # Compute non linear scale, effective spectral index and curvature
    k_nl, n, C = _halofit_parameters(cosmo, a, transfer_fn)

    om_m = bkgrd.Omega_m_a(cosmo, a)
    om_de = bkgrd.Omega_de_a(cosmo, a)
    w = bkgrd.w(cosmo, a)
    frac = om_de / (1.0 - om_m)

    if prescription == "smith2003":
        # eq C9 to C18
        a_n = 10 ** (
            1.4861
            + 1.8369 * n
            + 1.6762 * n ** 2
            + 0.7940 * n ** 3
            + 0.1670 * n ** 4
            - 0.6206 * C
        )
        b_n = 10 ** (0.9463 + 0.9466 * n + 0.3084 * n ** 2 - 0.9400 * C)
        c_n = 10 ** (-0.2807 + 0.6669 * n + 0.3214 * n ** 2 - 0.0793 * C)
        gamma_n = 0.8649 + 0.2989 * n + 0.1631 * C
        alpha_n = 1.3884 + 0.3700 * n - 0.1452 * n ** 2
        beta_n = 0.8291 + 0.9854 * n + 0.3401 * n ** 2
        mu_n = 10 ** (-3.5442 + 0.1908 * n)
        nu_n = 10 ** (0.9585 + 1.2857 * n)
    elif prescription == "takahashi2012":
        a_n = 10 ** (
            1.5222
            + 2.8553 * n
            + 2.3706 * n ** 2
            + 0.9903 * n ** 3
            + 0.2250 * n ** 4
            - 0.6038 * C
            + 0.1749 * om_de * (1 + w)
        )
        b_n = 10 ** (
            -0.5642
            + 0.5864 * n
            + 0.5716 * n ** 2
            - 1.5474 * C
            + 0.2279 * om_de * (1 + w)
        )
        c_n = 10 ** (0.3698 + 2.0404 * n + 0.8161 * n ** 2 + 0.5869 * C)
        gamma_n = 0.1971 - 0.0843 * n + 0.8460 * C
        alpha_n = np.abs(6.0835 + 1.3373 * n - 0.1959 * n ** 2 - 5.5274 * C)
        beta_n = (
            2.0379
            - 0.7354 * n
            + 0.3157 * n ** 2
            + 1.2490 * n ** 3
            + 0.3980 * n ** 4
            - 0.1682 * C
        )
        mu_n = 0.0
        nu_n = 10 ** (5.2105 + 3.6902 * n)
    else:
        raise NotImplementedError

    f1a = om_m ** (-0.0732)
    f2a = om_m ** (-0.1423)
    f3a = om_m ** 0.0725
    f1b = om_m ** (-0.0307)
    f2b = om_m ** (-0.0585)
    f3b = om_m ** (0.0743)

    if prescription == "takahashi2012":
        f1 = f1b
        f2 = f2b
        f3 = f3b
    elif prescription == "smith2003":
        f1 = frac * f1b + (1 - frac) * f1a
        f2 = frac * f2b + (1 - frac) * f2a
        f3 = frac * f3b + (1 - frac) * f3a
    else:
        raise NotImplementedError

    f = lambda x: x / 4.0 + x ** 2 / 8.0

    d2l = k ** 3 * pklin / (2.0 * np.pi ** 2)

    y = k / k_nl

    # Eq C2
    d2q = d2l * ((1.0 + d2l) ** beta_n / (1 + alpha_n * d2l)) * np.exp(-f(y))
    d2hprime = (
        a_n * y ** (3 * f1) / (1.0 + b_n * y ** f2 + (c_n * f3 * y) ** (3.0 - gamma_n))
    )
    d2h = d2hprime / (1.0 + mu_n / y + nu_n / y ** 2)
    # Eq. C1
    d2nl = d2q + d2h
    pk_nl = 2.0 * np.pi ** 2 / k ** 3 * d2nl

    return pk_nl.squeeze()


def nonlinear_matter_power(
    cosmo, k, a=1.0, transfer_fn=tklib.Eisenstein_Hu, nonlinear_fn=halofit
):
    """ Computes the non-linear matter power spectrum.

    This function is just a wrapper over several nonlinear power spectra.
    """
    return nonlinear_fn(cosmo, k, a, transfer_fn=transfer_fn)

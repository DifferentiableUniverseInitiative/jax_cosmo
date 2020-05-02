# This module contains analytic nonlinear prescriptions for the power spectrum
import jax.numpy as np


def _smith_parameters(self, a,  **kwargs):
  r""" Computes the non linear scale, effective spectral index
  and spectral curvature"""
  a = atleast_1d(a)
  R_nl = zeros_like(a)
  n = zeros_like(a)
  C = zeros_like(a)

  ksamp = logspace(log10(self._kmin), log10(self._kmax), 1024)
  pklog = interp1d(log(ksamp), ksamp**3 *
                   self.pk_lin(ksamp, **kwargs) / (2.0*pi**2))
  g = self.G(a)

  def int_sigma(logk, r, _g):
      y = exp(logk)*r
      return pklog(logk) * _g**2 * exp(-y**2)

  def int_neff(logk, r, _g):
      y = exp(logk)*r
      return pklog(logk) * _g**2 * y**2 * exp(-y**2)

  def int_C(logk, r, _g):
      y = exp(logk)*r
      return pklog(logk) * _g**2 * (y**2 - y**4) * exp(-y**2)

  for i in range(R_nl.size):
      sigm = lambda r: romberg(int_sigma, log(self._kmin), log(self._kmax),
                               args=(exp(r), g[i]), rtol=1e-4, vec_func=True) - 1
      R_nl[i] = exp(brentq(sigm, -5, 1.5, rtol=1e-4))

      n[i] = 2.0 * romberg(int_neff, log(self._kmin), log(self._kmax),
                              args=(R_nl[i], g[i]), rtol=1e-4,  vec_func=True) - 3

      C[i] = (3 + n[i])**2 + 4 * romberg(int_C, log(self._kmin), log(self._kmax),
                                            args=(R_nl[i], g[i]), rtol=1e-4,  vec_func=True)
  k_nl = 1.0/R_nl
  return k_nl, n, C


def smith2003(cosmo, k, a=1.0):
    r""" Computes the non linear matter power spectrum.

    Parameters
    ----------
    k: array_like
        Wave number in h Mpc^{-1}

    a: array_like, optional
        Scale factor (def: 1.0)

    nl_type: str, optional
        Type of non linear corrections. Only 'smith2003' is implemented

    type: str, optional
        Type of transfer function. Either 'eisenhu' or 'eisenhu_osc'
        (def: 'eisenhu_osc')

    Returns
    -------
    pk: array_like
        Non linear matter power spectrum at the specified scale
        and scale factor.

    Notes
    -----
    The non linear corrections are implemented following :cite:`2003:smith`

    """
    k = atleast_1d(k)
    a = atleast_1d(a)
    pklin = self.pk_lin(k, a, **kwargs)

    if (nl_type == 'smith2003'):

        # Compute non linear scale, effective spectral index and curvature
        k_nl, n, C = self._smith_parameters(a)

        om_m = self.Omega_m_a(a)
        frac = self.Omega_de_a(a)/(1.0 - om_m)

        # eq C9 to C18
        a_n = 10**(1.4861 + 1.8369*n + 1.6762*n**2 + 0.7940*n**3 +
                   0.1670*n**4 - 0.6206*C)
        b_n = 10**(0.9463 + 0.9466*n + 0.3084*n**2 - 0.9400*C)
        c_n = 10**(-0.2807 + 0.6669*n + 0.3214*n**2 - 0.0793*C)
        gamma_n = 0.8649 + 0.2989*n + 0.1631*C
        alpha_n = 1.3884 + 0.3700*n - 0.1452*n**2
        beta_n = 0.8291 + 0.9854*n + 0.3401*n**2
        mu_n = 10**(-3.5442 + 0.1908*n)
        nu_n = 10**(0.9585 + 1.2857*n)

        f1a = om_m**(-0.0732)
        f2a = om_m**(-0.1423)
        f3a = om_m**0.0725
        f1b = om_m**(-0.0307)
        f2b = om_m**(-0.0585)
        f3b = om_m**(0.0743)

        f1 = frac*f1b + (1-frac)*f1a
        f2 = frac*f2b + (1-frac)*f2a
        f3 = frac*f3b + (1-frac)*f3a

        f = lambda x: x/4. + x**2/8.

        d2l = einsum('i...,i...->i...', k**3, pklin / (2.0*pi**2))
        if k.ndim > 1:
            y = k/k_nl
        else:
            y = outer(k, 1.0/k_nl).squeeze()
        # Eq C2
        d2q = d2l * ((1.0+d2l)**beta_n/(1+alpha_n*d2l)) * exp(-f(y))
        d2hprime = a_n*y**(3*f1)/(1.0 + b_n * y**f2 +
                                  (c_n*f3*y)**(3.0 - gamma_n))
        d2h = d2hprime / (1.0 + mu_n/y + nu_n/y**2)
        # Eq. C1
        d2nl = d2q + d2h
        pk_nl = einsum('i...,i...->i...', 2.0*pi**2/k**3, d2nl)
    else:
        print("unknown non linear prescription")
        pk_nl = pklin

    return pk_nl.squeeze()

import jax.numpy as jnp
import numpy as np
import pyccl as ccl
from numpy.testing import assert_allclose

import jax_cosmo.power as power
import jax_cosmo.transfer as tklib
from jax_cosmo import Cosmology


def test_eisenstein_hu():
    # We first define equivalent CCL and jax_cosmo cosmologies
    cosmo_ccl = ccl.Cosmology(
        Omega_c=0.3,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=0,
        transfer_function="eisenstein_hu",
        matter_power_spectrum="linear",
    )

    cosmo_jax = Cosmology(
        Omega_c=0.3,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )

    # Test array of scales
    k = np.logspace(-4, 2, 512)

    # Computing matter power spectrum
    pk_ccl = ccl.linear_matter_power(cosmo_ccl, k, 1.0)
    pk_jax = (
        power.linear_matter_power(cosmo_jax, k / cosmo_jax.h, a=1.0) / cosmo_jax.h**3
    )

    assert_allclose(pk_ccl, pk_jax, rtol=0.5e-2)


def test_halofit():
    # We first define equivalent CCL and jax_cosmo cosmologies
    cosmo_ccl = ccl.Cosmology(
        Omega_c=0.3,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=0,
        transfer_function="eisenstein_hu",
        matter_power_spectrum="halofit",
    )

    cosmo_jax = Cosmology(
        Omega_c=0.3,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )

    # Test array of scales
    k = np.logspace(-4, 2, 512)

    # Computing matter power spectrum
    pk_ccl = ccl.nonlin_matter_power(cosmo_ccl, k, 1.0)
    pk_jax = (
        power.nonlinear_matter_power(
            cosmo_jax,
            k / cosmo_jax.h,
            a=1.0,
            transfer_fn=tklib.Eisenstein_Hu,
            nonlinear_fn=power.halofit,
        )
        / cosmo_jax.h**3
    )

    assert_allclose(pk_ccl, pk_jax, rtol=0.5e-2)


def test_halofit_nl_scales():
    """This test checks that the halofit correction doesn't become
    unstable when the nonlinear scales are very small.

    Note: CCL is not accurate by default when a is too small,
    it will clip the non linear scales at some kmax around 100.
    """
    # We first define equivalent CCL and jax_cosmo cosmologies
    cosmo_ccl = ccl.Cosmology(
        Omega_c=0.3,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Neff=0,
        transfer_function="eisenstein_hu",
        matter_power_spectrum="halofit",
    )

    cosmo_jax = Cosmology(
        Omega_c=0.3,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )

    # Test array of scales
    k = np.logspace(-4, 2, 512)

    # Computing matter power spectrum
    pk_ccl = ccl.nonlin_matter_power(cosmo_ccl, k, 0.1)
    pk_jax = (
        power.nonlinear_matter_power(
            cosmo_jax,
            k / cosmo_jax.h,
            a=0.1,
            transfer_fn=tklib.Eisenstein_Hu,
            nonlinear_fn=power.halofit,
        )
        / cosmo_jax.h**3
    )
    assert_allclose(pk_ccl, pk_jax, rtol=0.5e-2)

    # Asserting that even if a is very small, we dont get NaNs
    # Computing matter power spectrum
    pk_ccl = ccl.nonlin_matter_power(cosmo_ccl, k, 0.01)
    pk_jax = (
        power.nonlinear_matter_power(
            cosmo_jax,
            k / cosmo_jax.h,
            a=0.01,
            transfer_fn=tklib.Eisenstein_Hu,
            nonlinear_fn=power.halofit,
        )
        / cosmo_jax.h**3
    )
    # We relax the test here, because actually CCL is not accurate in this regime
    assert_allclose(pk_ccl, pk_jax, rtol=2e-2)

import jax.numpy as jnp
import numpy as np
import pyccl as ccl
from numpy.testing import assert_allclose

import jax_cosmo.background as bkgrd
from jax_cosmo import Cosmology


def test_H():
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
        wa=2.0,  # non-zero wa
    )

    cosmo_jax = Cosmology(
        Omega_c=0.3,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.0,
        w0=-1.0,
        wa=2.0,  # non-zero wa
    )

    # Test array of scale factors
    a = np.linspace(0.01, 1.0)

    H_ccl = ccl.h_over_h0(cosmo_ccl, a)
    H_jax = bkgrd.H(cosmo_jax, a) / 100.0
    assert_allclose(H_ccl, H_jax, rtol=1.0e-3)


def test_distances_flat():
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

    # Test array of scale factors
    a = np.linspace(0.01, 1.0)

    chi_ccl = ccl.comoving_radial_distance(cosmo_ccl, a)
    chi_jax = bkgrd.radial_comoving_distance(cosmo_jax, a) / cosmo_jax.h
    assert_allclose(chi_ccl, chi_jax, rtol=0.5e-2)

    chi_ccl = ccl.comoving_angular_distance(cosmo_ccl, a)
    chi_jax = bkgrd.transverse_comoving_distance(cosmo_jax, a) / cosmo_jax.h
    assert_allclose(chi_ccl, chi_jax, rtol=0.5e-2)

    chi_ccl = ccl.angular_diameter_distance(cosmo_ccl, a)
    chi_jax = bkgrd.angular_diameter_distance(cosmo_jax, a) / cosmo_jax.h
    assert_allclose(chi_ccl, chi_jax, rtol=0.5e-2)


def test_growth():
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

    # Test array of scale factors
    a = np.linspace(0.01, 1.0)

    gccl = ccl.growth_factor(cosmo_ccl, a)
    gjax = bkgrd.growth_factor(cosmo_jax, a)

    assert_allclose(gccl, gjax, rtol=1e-2)


def test_growth_rate():
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

    # Test array of scale factors
    a = np.linspace(0.01, 1.0)

    fccl = ccl.growth_rate(cosmo_ccl, a)
    fjax = bkgrd.growth_rate(cosmo_jax, a)

    assert_allclose(fccl, fjax, rtol=1e-2)


def test_growth_rate_gamma():
    # We test consistency of both effective growth
    # parametrisation for LCDM
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
        gamma=0.55,
    )

    # Test array of scale factors
    a = np.linspace(0.01, 1.0)

    fccl = ccl.growth_rate(cosmo_ccl, a)
    fjax = bkgrd.growth_rate(cosmo_jax, a)

    assert_allclose(fccl, fjax, rtol=1e-2)


def test_growth_gamma():
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
        gamma=0.55,
    )

    # Test array of scale factors
    a = np.linspace(0.01, 1.0)

    gccl = ccl.growth_factor(cosmo_ccl, a)
    gjax = bkgrd.growth_factor(cosmo_jax, a)

    assert_allclose(gccl, gjax, rtol=1e-2)


def test_growth_factor_second():
    # Test second-order growth factor
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

    # Test array of scale factors
    a = np.linspace(0.01, 1.0)

    d1 = bkgrd.growth_factor(cosmo_jax, a)
    d2 = bkgrd.growth_factor_second(cosmo_jax, a)

    # Test normalization: D2(a=1) = 1
    assert_allclose(
        bkgrd.growth_factor_second(cosmo_jax, jnp.atleast_1d(1.0)), 1.0, rtol=1e-10
    )

    # Did not find a pyccl equivalent to test against, so I test like so
    # (matter-dominated), D2_norm / D1_norm^2 ≈ 1
    # This is because both are normalized to 1 at a=1
    d1_early = bkgrd.growth_factor(cosmo_jax, a)
    d2_early = bkgrd.growth_factor_second(cosmo_jax, a)
    ratio = d2_early / d1_early**2
    assert_allclose(ratio, 1.0, rtol=1e-2, atol=1e-5)


def test_growth_rate_second():
    # Test second-order growth rate
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

    # Test array of scale factors
    a = np.linspace(0.1, 1.0, 100)

    d2 = bkgrd.growth_factor_second(cosmo_jax, a)
    f2 = bkgrd.growth_rate_second(cosmo_jax, a)

    # did not find a pyccl equivalent to test against, so I test like so
    # Test numerical consistency: f2 = d ln(D2)/d ln(a)
    f2_numerical = np.gradient(np.log(np.abs(d2)), np.log(a))
    assert_allclose(f2, f2_numerical, rtol=5e-2)

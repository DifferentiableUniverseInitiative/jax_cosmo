import jax.numpy as jnp
import numpy as np
import pyccl as ccl
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

import jax_cosmo.background as bkgrd
from jax_cosmo import Cosmology
from jax_cosmo import probes
from jax_cosmo.angular_cl import angular_cl
from jax_cosmo.angular_cl import gaussian_cl_covariance
from jax_cosmo.bias import constant_linear_bias
from jax_cosmo.bias import inverse_growth_linear_bias
from jax_cosmo.redshift import delta_nz
from jax_cosmo.redshift import kde_nz
from jax_cosmo.redshift import smail_nz
from jax_cosmo.sparse import to_dense


def test_lensing_cl():
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

    # Define a redshift distribution
    nz = smail_nz(1.0, 2.0, 1.0)
    z = np.linspace(0, 5.0, 1024)
    tracer_ccl = ccl.WeakLensingTracer(cosmo_ccl, (z, nz(z)), use_A_ia=False)
    tracer_jax = probes.WeakLensing([nz])

    # Get an ell range for the cls
    ell = np.logspace(0.1, 4)

    # Compute the cls
    cl_ccl = ccl.angular_cl(cosmo_ccl, tracer_ccl, tracer_ccl, ell)
    cl_jax = angular_cl(cosmo_jax, ell, [tracer_jax])

    assert_allclose(cl_ccl, cl_jax[0], rtol=1.0e-2)


def test_lensing_cl_delta():
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

    # Define a redshift distribution
    z0 = 1.0
    z = np.linspace(0, 5.0, 1024)
    pz = np.zeros_like(z)
    pz[np.argmin(abs(z0 - z))] = 1.0
    nzs_s = kde_nz(z, pz, bw=0.01)
    nz = delta_nz(z0)
    nz_smail1 = smail_nz(1.0, 2.0, 1.0)
    nz_smail2 = smail_nz(1.4, 2.0, 1.0)
    tracer_ccl = ccl.WeakLensingTracer(cosmo_ccl, (z, nzs_s(z)), use_A_ia=False)
    tracer_cclb = ccl.WeakLensingTracer(cosmo_ccl, (z, nz_smail2(z)), use_A_ia=False)
    tracer_jax = probes.WeakLensing([nz])
    tracer_jaxb = probes.WeakLensing([nz, nz_smail1, nz_smail2])

    # Get an ell range for the cls
    ell = np.logspace(0.1, 4)

    # Compute the cls
    cl_ccl = ccl.angular_cl(cosmo_ccl, tracer_ccl, tracer_ccl, ell)
    cl_jax = angular_cl(cosmo_jax, ell, [tracer_jax])
    assert_allclose(cl_ccl, cl_jax[0], rtol=1.0e-2)

    # Also test if several nzs are provided
    cl_ccl = ccl.angular_cl(cosmo_ccl, tracer_cclb, tracer_cclb, ell)
    cl_jax = angular_cl(cosmo_jax, ell, [tracer_jaxb])
    assert_allclose(cl_ccl, cl_jax[-1], rtol=1.0e-2)


def test_lensing_cl_IA():
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

    # Define a redshift distribution
    nz = smail_nz(1.0, 2.0, 1.0)
    z = np.linspace(0, 5.0, 1024)

    # Pretty big IA to highlight potential differences stemming from IA
    # implementation
    bias = inverse_growth_linear_bias(10.0)

    tracer_ccl = ccl.WeakLensingTracer(
        cosmo_ccl, (z, nz(z)), ia_bias=(z, bias(cosmo_jax, z))
    )
    tracer_jax = probes.WeakLensing([nz], bias)

    # Get an ell range for the cls
    ell = np.logspace(0.1, 4)

    # Compute the cls
    cl_ccl = ccl.angular_cl(cosmo_ccl, tracer_ccl, tracer_ccl, ell)
    cl_jax = angular_cl(cosmo_jax, ell, [tracer_jax])

    assert_allclose(cl_ccl, cl_jax[0], rtol=1e-2)


def test_clustering_cl():
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

    # Define a redshift distribution
    nz = smail_nz(1.0, 2.0, 1.0)

    # And a bias model
    bias = constant_linear_bias(1.0)

    z = np.linspace(0, 5.0, 1024)
    tracer_ccl = ccl.NumberCountsTracer(
        cosmo_ccl, has_rsd=False, dndz=(z, nz(z)), bias=(z, bias(cosmo_jax, z))
    )
    tracer_jax = probes.NumberCounts([nz], bias)

    # Get an ell range for the cls
    ell = np.logspace(0.1, 4)

    # Compute the cls
    cl_ccl = ccl.angular_cl(cosmo_ccl, tracer_ccl, tracer_ccl, ell)
    cl_jax = angular_cl(cosmo_jax, ell, [tracer_jax])

    assert_allclose(cl_ccl, cl_jax[0], rtol=1e-2)


def test_sparse_cov():
    n_ell = 25
    ell = jnp.logspace(1, 3, n_ell)
    nz1 = smail_nz(1.0, 2.0, 1.0)
    nz2 = smail_nz(1.0, 2.0, 0.5)
    n_cls = 3
    P = [probes.NumberCounts([nz1, nz2], constant_linear_bias(1.0))]
    cl_signal = jnp.ones((n_cls, n_ell))
    cl_noise = jnp.ones_like(cl_signal)
    cov_dense = gaussian_cl_covariance(ell, P, cl_signal, cl_noise, sparse=False)
    cov_sparse = gaussian_cl_covariance(ell, P, cl_signal, cl_noise, sparse=True)
    assert cov_sparse.shape == (n_cls, n_cls, n_ell)
    assert_array_equal(to_dense(cov_sparse), cov_dense)

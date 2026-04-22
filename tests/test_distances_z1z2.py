import jax.numpy as jnp
import numpy as np
import pyccl as ccl
from numpy.testing import assert_allclose
import pytest

import jax_cosmo.background as bkgrd
from jax_cosmo import Cosmology


def test_comoving_distance_z1z2_flat():
    """Test comoving distance between two redshifts for flat cosmology."""
    # Define equivalent CCL and jax_cosmo cosmologies
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

    # Test arrays of redshifts
    z1 = np.array([0.1, 0.2, 0.5])
    z2 = np.array([0.5, 1.0, 1.5])

    # Compare with manually calculated distances
    for i in range(len(z1)):
        d1_ccl = ccl.comoving_radial_distance(cosmo_ccl, 1.0/(1.0 + z1[i]))
        d2_ccl = ccl.comoving_radial_distance(cosmo_ccl, 1.0/(1.0 + z2[i]))
        expected = abs(d2_ccl - d1_ccl)
        
        result_jax = bkgrd.comoving_distance_z1z2(cosmo_jax, z1[i], z2[i]) / cosmo_jax.h
        assert_allclose(expected, result_jax, rtol=1e-2)


def test_comoving_distance_z1z2_curved():
    """Test comoving distance between two redshifts for curved cosmology."""
    # Define curved cosmology
    cosmo_ccl = ccl.Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.1,  # Open universe
        Neff=0,
        transfer_function="eisenstein_hu",
        matter_power_spectrum="linear",
    )

    cosmo_jax = Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.1,
        w0=-1.0,
        wa=0.0,
    )

    # Test a single pair of redshifts
    z1, z2 = 0.2, 0.8
    
    # Compare with manually calculated distances
    d1_ccl = ccl.comoving_radial_distance(cosmo_ccl, 1.0/(1.0 + z1))
    d2_ccl = ccl.comoving_radial_distance(cosmo_ccl, 1.0/(1.0 + z2))
    expected = abs(d2_ccl - d1_ccl)
    
    result_jax = bkgrd.comoving_distance_z1z2(cosmo_jax, z1, z2) / cosmo_jax.h
    assert_allclose(expected, result_jax, rtol=1e-2)


def test_angular_diameter_distance_z1z2_consistency():
    """Test consistency of z1z2 angular diameter distance with single redshift version."""
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

    # Test that angular_diameter_distance_z1z2(0, z) gives the same result as
    # angular_diameter_distance(z)
    z = 1.0
    a = 1.0 / (1.0 + z)
    
    result_single = bkgrd.angular_diameter_distance(cosmo_jax, a)
    result_z1z2 = bkgrd.angular_diameter_distance_z1z2(cosmo_jax, 0.0, z)
    
    assert_allclose(result_single, result_z1z2, rtol=1e-6)


def test_angular_diameter_distance_z1z2_properties():
    """Test basic properties of z1z2 angular diameter distance."""
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

    # Test that distance is positive for z2 > z1
    z1, z2 = 0.2, 0.8
    result = bkgrd.angular_diameter_distance_z1z2(cosmo_jax, z1, z2)
    assert result > 0

    # Test that distance is zero for z1 = z2
    z = 0.5
    result = bkgrd.angular_diameter_distance_z1z2(cosmo_jax, z, z)
    assert_allclose(result, 0.0, atol=1e-10)


def test_lensing_efficiency():
    """Test a practical lensing application using the z1z2 functions."""
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

    # Typical lensing scenario: lens at z=0.3, source at z=1.2
    z_lens = 0.3
    z_source = 1.2

    # Calculate lensing efficiency factor: D_ls / D_s
    d_ls = bkgrd.angular_diameter_distance_z1z2(cosmo_jax, z_lens, z_source)
    d_s = bkgrd.angular_diameter_distance_z1z2(cosmo_jax, 0.0, z_source)
    
    lensing_efficiency = d_ls / d_s
    
    # Basic sanity checks
    assert 0 < lensing_efficiency < 1  # Should be between 0 and 1
    assert lensing_efficiency > 0.5  # For these redshifts, should be reasonably large


def test_vectorization():
    """Test that the functions work with vectorized inputs."""
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

    # Test with arrays
    z1 = np.array([0.1, 0.2, 0.3])
    z2 = np.array([0.5, 0.8, 1.0])

    # These should not raise errors and should return arrays of the same shape
    result_comoving = bkgrd.comoving_distance_z1z2(cosmo_jax, z1, z2)
    result_transverse = bkgrd.comoving_transverse_distance_z1z2(cosmo_jax, z1, z2)
    result_angular = bkgrd.angular_diameter_distance_z1z2(cosmo_jax, z1, z2)
    
    assert result_comoving.shape == z1.shape
    assert result_transverse.shape == z1.shape
    assert result_angular.shape == z1.shape
    
    # All distances should be positive
    assert np.all(result_comoving > 0)
    assert np.all(result_transverse > 0)
    assert np.all(result_angular > 0)


def test_curvature_effects():
    """Test that curvature affects transverse distance but not comoving distance."""
    # Flat cosmology
    cosmo_flat = Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )
    
    # Open cosmology  
    cosmo_open = Cosmology(
        Omega_c=0.2,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.05,
        w0=-1.0,
        wa=0.0,
    )

    z1, z2 = 0.2, 0.8

    # Comoving distances should be the same regardless of curvature
    # (they only depend on the integration of H(z))
    dc_flat = bkgrd.comoving_distance_z1z2(cosmo_flat, z1, z2)
    dc_open = bkgrd.comoving_distance_z1z2(cosmo_open, z1, z2)
    
    # Transverse distances should be different
    dt_flat = bkgrd.comoving_transverse_distance_z1z2(cosmo_flat, z1, z2)
    dt_open = bkgrd.comoving_transverse_distance_z1z2(cosmo_open, z1, z2)
    
    # For flat universe, transverse = comoving
    assert_allclose(dc_flat, dt_flat, rtol=1e-12)
    
    # For open universe, transverse > comoving
    assert dt_open > dc_open
    
    # The actual values should be different between cosmologies
    # (not exactly the same due to different matter densities)
    assert not np.allclose(dc_flat, dc_open, rtol=1e-2)
    assert not np.allclose(dt_flat, dt_open, rtol=1e-2)
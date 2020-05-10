import numpy as np
import jax.numpy as jnp
from numpy.testing import assert_allclose
from jax_cosmo import Cosmology
import jax_cosmo.background as bkgrd
from jax_cosmo.redshift import smail_nz
from jax_cosmo.angular_cl import angular_cl
from jax_cosmo.bias import constant_linear_bias
from jax_cosmo import probes
import pyccl as ccl

def test_lensing_cl():
  # We first define equivalent CCL and jax_cosmo cosmologies
  cosmo_ccl = ccl.Cosmology(
    Omega_c=0.3, Omega_b=0.05, h=0.7, sigma8 = 0.8, n_s=0.96, Neff=0,
    transfer_function='eisenstein_hu', matter_power_spectrum='linear')

  cosmo_jax = Cosmology(Omega_c=0.3, Omega_b=0.05, h=0.7, sigma8 = 0.8, n_s=0.96,
                         Omega_k=0., w0=-1., wa=0.)

  # Define a redshift distribution
  nz = smail_nz(1., 2., 1.)
  z = np.linspace(0,5.,1024)
  tracer_ccl = ccl.WeakLensingTracer(cosmo_ccl, (z, nz(z)), use_A_ia=False)
  tracer_jax = probes.WeakLensing([nz])

  # Get an ell range for the cls
  ell = np.logspace(0.1,4)

  # Compute the cls
  cl_ccl = ccl.angular_cl(cosmo_ccl, tracer_ccl, tracer_ccl, ell)
  cl_jax = angular_cl(cosmo_jax, ell, [tracer_jax])

  assert_allclose(cl_ccl, cl_jax[0], rtol=1e-2)


def test_clustering_cl():
  # We first define equivalent CCL and jax_cosmo cosmologies
  cosmo_ccl = ccl.Cosmology(
    Omega_c=0.3, Omega_b=0.05, h=0.7, sigma8 = 0.8, n_s=0.96, Neff=0,
    transfer_function='eisenstein_hu', matter_power_spectrum='linear')

  cosmo_jax = Cosmology(Omega_c=0.3, Omega_b=0.05, h=0.7, sigma8 = 0.8, n_s=0.96,
                         Omega_k=0., w0=-1., wa=0.)

  # Define a redshift distribution
  nz = smail_nz(1., 2., 1.)

  # And a bias model
  bias = constant_linear_bias(1.)

  z = np.linspace(0,5.,1024)
  tracer_ccl = ccl.NumberCountsTracer(cosmo_ccl,
                                      has_rsd=False,
                                      dndz=(z, nz(z)),
                                      bias=(z, bias(z)))
  tracer_jax = probes.NumberCounts([nz], bias)

  # Get an ell range for the cls
  ell = np.logspace(0.1,4)

  # Compute the cls
  cl_ccl = ccl.angular_cl(cosmo_ccl, tracer_ccl, tracer_ccl, ell)
  cl_jax = angular_cl(cosmo_jax, ell, [tracer_jax])

  assert_allclose(cl_ccl, cl_jax[0], rtol=1e-2)

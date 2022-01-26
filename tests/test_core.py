from dataclasses import FrozenInstanceError

from numpy.testing import assert_raises

from jax_cosmo import Cosmology


def test_Cosmology_immutability():
    cosmo = Cosmology(
        Omega_c=0.3,
        Omega_b=0.05,
        h=0.67,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )

    with assert_raises(FrozenInstanceError):
        cosmo.h = 0.74  # Hubble doesn't budge on the tension

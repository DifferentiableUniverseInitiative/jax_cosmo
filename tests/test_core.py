from dataclasses import FrozenInstanceError

from numpy.testing import assert_raises

from jax_cosmo import Cosmology


def test_Cosmology_immutability():
    cosmo = Cosmology(
        Omega_c=0.25,
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


def test_cache():
    cosmo = Cosmology(
        Omega_c=0.25,
        Omega_b=0.05,
        h=0.7,
        sigma8=0.8,
        n_s=0.96,
        Omega_k=0.0,
        w0=-1.0,
        wa=0.0,
    )

    cosmo = cosmo.cache_set("a", 1)
    cosmo = cosmo.cache_set("c", 3)
    assert cosmo.is_cached("a")
    assert not cosmo.is_cached("b")
    assert cosmo.cache_get("c") == 3

    cosmo = cosmo.cache_set("b", 2)
    cosmo_no_c = cosmo.cache_del("c")
    assert cosmo.is_cached("b")
    assert not cosmo_no_c.is_cached("c")
    assert (
        cosmo_no_c is not cosmo
        and cosmo_no_c == cosmo
        and cosmo_no_c._cache != cosmo._cache
    )

    cosmo = cosmo.cache_clear()
    assert len(cosmo._cache) == 0

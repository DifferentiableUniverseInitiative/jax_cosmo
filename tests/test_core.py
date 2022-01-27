from dataclasses import FrozenInstanceError

from numpy.testing import assert_raises

from jax_cosmo import Configuration
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


def test_Conguration_immutability():
    config = Configuration()

    with assert_raises(FrozenInstanceError):
        config.log10_a_max = 0.0


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

    def assert_pure(c1, c2):
        assert c1 is not c2 and c1 == c2 and c1._cache != c2._cache

    cosmo = cosmo.cache_set("a", 1)
    cosmo = cosmo.cache_set("c", 3)

    assert cosmo.is_cached("a")
    assert not cosmo.is_cached("b")
    assert cosmo.cache_get("c") == 3

    cosmo_add_b = cosmo.cache_set("b", 2)
    cosmo_del_c = cosmo.cache_del("c")
    cosmo_clean = cosmo.cache_clear()

    assert not cosmo_del_c.is_cached("c")
    assert_pure(cosmo_add_b, cosmo)  # test set purity
    assert_pure(cosmo_del_c, cosmo)  # test del purity
    assert_pure(cosmo_clean, cosmo)  # test clear purity
    assert len(cosmo_clean._cache) == 0

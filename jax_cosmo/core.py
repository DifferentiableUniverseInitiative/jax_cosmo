"""This module implements the Cosmology type containing cosmological parameters
and cached computations, and the Configuration type containing configuration parameters.

"""
from dataclasses import dataclass
from dataclasses import field
from dataclasses import replace
from functools import partial
from pprint import pformat
from typing import Any
from typing import Optional

import jax.numpy as np

from jax_cosmo.dataclasses import pytree_dataclass

__all__ = ["Cosmology", "Configuration"]


@dataclass(frozen=True)
class Configuration:
    """Configuration parameters, that are not to be traced by JAX.

    Parameters
    ----------
    log10_a_min : float, optional
        Minimum in scale factor logspace range
    log10_a_max : float, optional
        Maximum in scale factor logspace range
    log10_a_num : int, optional
        Number of samples in scale factor logspace range
    growth_rtol : float, optional
        Relative error tolerance for solving growth ODEs
    growth_atol : float, optional
        Absolute error tolerance for solving growth ODEs

    log10_k_min : float, optional
        Minimum in wavenumber logspace range
    log10_k_max : float, optional
        Maximum in wavenumber logspace range

    """

    log10_a_min: float = -3.0
    log10_a_max: float = 0.0
    log10_a_steps: int = 256  # TODO revisit after improving odeint and interpolation
    growth_atol: float = 0.0
    growth_rtol: float = 1e-4

    log10_k_min: float = -4.0
    log10_k_max: float = 3.0


@partial(pytree_dataclass, aux_fields="config", frozen=True)
class Cosmology:
    """
    Cosmology parameter type, containing primary, secondary, derived parameters,
    cached computations, and configurations; immutable as a frozen dataclass.

    Parameters
    ----------
    Omega_c : float
        Cold dark matter density fraction.
    Omega_b : float
        Baryonic matter density fraction.
    h : float
        Hubble constant divided by 100 km/s/Mpc; unitless.
    n_s : float
        Primordial scalar perturbation spectral index.
    sigma8 : float
        RMS of matter density perturbations in an 8 Mpc/h spherical tophat.
    Omega_k : float
        Curvature density fraction.
    w0 : float
        First order term of dark energy equation.
    wa : float
        Second order term of dark energy equation of state.
    gamma : float, optional
        Exponent of growth rate fitting formula.
    config : Configuration, optional
        Configuration parameters.

    Notes
    -----

    If `gamma` is specified, the growth rate fitting formula
    :math:`dlnD/dlna = \Omega_m(a)^\gamma` will be used to model the growth history.
    Otherwise the linear growth factor and growth rate will be solved by ODE.

    """

    # Primary parameters
    Omega_c: float
    Omega_b: float
    h: float
    n_s: float
    sigma8: float
    Omega_k: float
    w0: float
    wa: float

    # Secondary optional parameters
    gamma: Optional[float] = None

    # cache for intermediate computations;
    # users should not access it directly but use the class methods instead
    _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    # configuration parameters, immutable (frozen dataclass)
    config: Configuration = field(default_factory=Configuration)

    def __str__(self):
        return pformat(self, indent=4, width=1)  # for python >= 3.10

    def is_cached(self, key):
        return key in self._cache

    def cache_get(self, key):
        return self._cache[key]

    def cache_set(self, key, value):
        """Add key-value pair to cache and return a new ``Cosmology`` instance."""
        cache = self._cache.copy()
        cache[key] = value
        return replace(self, _cache=cache)

    def cache_del(self, key):
        """Remove key from cache and return a new ``Cosmology`` instance."""
        cache = self._cache.copy()
        del cache[key]
        return replace(self, _cache=cache)

    def cache_clear(self):
        """Return a new ``Cosmology`` instance with empty cache."""
        return replace(self, _cache={})

    # Derived parameters
    @property
    def Omega(self):
        return 1.0 - self.Omega_k

    @property
    def Omega_m(self):
        return self.Omega_b + self.Omega_c

    @property
    def Omega_de(self):
        return self.Omega - self.Omega_m

    @property
    def k(self):
        return -np.sign(self.Omega_k).astype(np.int8)

    @property
    def sqrtk(self):
        return np.sqrt(np.abs(self.Omega_k))

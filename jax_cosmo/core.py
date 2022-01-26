from dataclasses import field
from dataclasses import replace
from functools import partial
from pprint import pformat
from typing import Any
from typing import Optional

import jax.numpy as np

from jax_cosmo.dataclasses import pytree_dataclass

__all__ = ["Cosmology"]


@partial(pytree_dataclass, frozen=True)
class Cosmology:
    """
    Cosmology parameter class, including primary, secondary, and derived parameters; immutable.

    Parameters:
    -----------
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

    Notes:
    ------

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

    def __str__(self):
        return pformat(self, indent=4, width=1)  # for python >= 3.10

    def is_cached(self, key):
        return key in self._cache

    def cache_get(self, key):
        return self._cache[key]

    def cache_set(self, key, value):
        cache = self._cache
        cache[key] = value
        return replace(self, _cache=cache)

    def cache_clear(self):
        cache = self._cache
        cache.clear()
        return replace(self, _cache=cache)

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

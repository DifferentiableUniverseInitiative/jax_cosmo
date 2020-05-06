# Module to define redshift distributions we can differentiate through
from abc import ABC, abstractmethod
import jax.numpy as np

from jax.tree_util import register_pytree_node_class
from jax_cosmo.scipy.integrate import simps

class redshift_distribution(ABC):

  def __init__(self, zmax=10., **kwargs):
    """
    Initialize the parameters of the redshift distribution
    """
    self._params = kwargs
    self._norm = None
    self._zmax = zmax

  @abstractmethod
  def pz_fn(self, z, **kwargs):
    """
    Un-normalized n(z) function provided by sub classes
    """
    pass

  def __call__(self, z):
    """
    Computes the normalized n(z)
    """
    if self._norm is None:
      self._norm = simps(lambda t: self.pz_fn(t, **(self._params)),
                         0., self._zmax, 256)
    return self.pz_fn(z, **(self._params))/self._norm

  def __repr__(self):
    return "".join(["%s : %f ; "%(k, self._params[k]) for k in self._params.keys()])

  @property
  def zmax(self):
    return self._zmax

  # Operations for flattening/unflattening representation
  def tree_flatten(self):
      return ((self._params, self._zmax) , None)

  @classmethod
  def tree_unflatten(cls, aux_data, children):
      params, zmax = list(children)
      return cls(zmax=zmax, **params)

@register_pytree_node_class
class smail_nz(redshift_distribution):
  def pz_fn(self, z, **p):
    return z**p['a'] * np.exp(-(z / p['z0'])**p['b'])

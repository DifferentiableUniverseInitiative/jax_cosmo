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

@register_pytree_node_class
class kde_nz(redshift_distribution):
  """
  A redshift distribution based on a KDE estimate of the nz of a given catalog
  currently uses a Gaussian kernel, TODO: add more if necessary
  """

  def _kernel(self, bw, X, x):
    """
    Gaussian kernel for KDE
    """
    return (1. / np.sqrt(2 * np.pi )/bw) * np.exp(-(X - x)**2 / (bw**2 * 2.))

  def pz_fn(self, z, **p):
    if 'weight' in p.keys():
      w = np.atleast_1d(p['weight'])
    else:
      w = np.ones_like(p['zcat'])
    q = np.sum(w)
    X = np.expand_dims(p['zcat'], axis=-1)
    k = self._kernel(p['bw'], X , z)
    return np.dot(k.T, w)/(q)

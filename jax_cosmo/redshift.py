# Module to define redshift distributions we can differentiate through
from abc import ABC, abstractmethod
import jax.numpy as np

from jax.tree_util import register_pytree_node_class
from jax_cosmo.scipy.integrate import simps
from jax_cosmo.jax_utils import container

class redshift_distribution(container):

  def __init__(self, *args, zmax=10., **kwargs):
    """
    Initialize the parameters of the redshift distribution
    """
    self._norm = None
    super(redshift_distribution, self).__init__(*args,
                                                zmax=zmax,
                                                **kwargs)
  @abstractmethod
  def pz_fn(self, z):
    """
    Un-normalized n(z) function provided by sub classes
    """
    pass

  def __call__(self, z):
    """
    Computes the normalized n(z)
    """
    if self._norm is None:
      self._norm = simps(lambda t: self.pz_fn(t),
                         0., self.config['zmax'], 256)
    return self.pz_fn(z)/self._norm

  @property
  def zmax(self):
    return self.config['zmax']
    
@register_pytree_node_class
class smail_nz(redshift_distribution):
  """
  Defines a smail distribution with these arguments

  Parameters:
  -----------
  a:

  b:

  z0
  """
  def pz_fn(self, z):
    a, b, z0 = self.params
    return z**a * np.exp(-(z / z0)**b)

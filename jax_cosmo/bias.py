# This module contains implementations of galaxy bias
import jax.numpy as np
from jax.tree_util import register_pytree_node_class
from jax_cosmo.utils import a2z, z2a
from jax_cosmo.jax_utils import container
import jax_cosmo.background as bkgrd

@register_pytree_node_class
class constant_linear_bias(container):
  """
  Class representing a linear bias

  Parameters:
  -----------
  b: redshift independent bias value
  """
  def __call__(self, cosmo, z):
    """
    """
    b = self.params[0]
    return b * np.ones_like(z)

@register_pytree_node_class
class inverse_growth_linear_bias(container):
  """
  TODO: what's a better name for this?
  Class representing an inverse bias in 1/growth(a)

  Parameters:
  -----------
  cosmo: cosmology
  b: redshift independent bias value at z=0
  """
  def __call__(self, cosmo, z):
    """
    """
    b = self.params[0]
    return b / bkgrd.growth_factor(cosmo, z2a(z))

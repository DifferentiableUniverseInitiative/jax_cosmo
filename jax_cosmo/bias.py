# This module contains implementations of galaxy bias
import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from jax_cosmo.jax_utils import container

@register_pytree_node_class
class constant_linear_bias(container):
  """
  Class representing a linear bias

  Parameters:
  -----------
  b: redshift independent bias value
  """
  def __call__(self, z):
    """
    """
    b = self.params[0]
    return b * np.ones_like(z)

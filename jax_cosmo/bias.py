# This module contains implementations of galaxy bias
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
from jax.tree_util import register_pytree_node_class

import jax_cosmo.background as bkgrd
from jax_cosmo.jax_utils import container
from jax_cosmo.utils import a2z
from jax_cosmo.utils import z2a


@register_pytree_node_class
class constant_linear_bias(container):
    """
    Class representing a linear bias

    Parameters:
    -----------
    b: redshift independent bias value
    """

    def __call__(self, cosmo, z):
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
        b = self.params[0]
        return b / bkgrd.growth_factor(cosmo, z2a(z))


@register_pytree_node_class
class des_y1_ia_bias(container):
    """
    https://arxiv.org/pdf/1708.01538.pdf Sec. VII.B

    Parameters:
    -----------
    cosmo: cosmology
    A: amplitude
    eta: redshift dependent slope
    z0: pivot redshift
    """

    def __call__(self, cosmo, z):
        A, eta, z0 = self.params
        return A * ((1.0 + z) / (1.0 + z0)) ** eta

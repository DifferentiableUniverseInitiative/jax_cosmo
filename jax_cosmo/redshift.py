# Module to define redshift distributions we can differentiate through
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABC
from abc import abstractmethod

import jax.numpy as np
from jax.tree_util import register_pytree_node_class

from jax_cosmo.jax_utils import container
from jax_cosmo.scipy.integrate import simps

steradian_to_arcmin2 = 11818102.86004228

__all__ = ["smail_nz", "kde_nz"]


class redshift_distribution(container):
    def __init__(self, *args, gals_per_arcmin2=1.0, zmax=10.0, **kwargs):
        """ Initialize the parameters of the redshift distribution
        """
        self._norm = None
        self._gals_per_arcmin2 = gals_per_arcmin2
        super(redshift_distribution, self).__init__(*args, zmax=zmax, **kwargs)

    @abstractmethod
    def pz_fn(self, z):
        """ Un-normalized n(z) function provided by sub classes
        """
        pass

    def __call__(self, z):
        """ Computes the normalized n(z)
        """
        if self._norm is None:
            self._norm = simps(lambda t: self.pz_fn(t), 0.0, self.config["zmax"], 256)
        return self.pz_fn(z) / self._norm

    @property
    def zmax(self):
        return self.config["zmax"]

    @property
    def gals_per_arcmin2(self):
        """ Returns the number density of galaxies in gals/sq arcmin
        TODO: find a better name
        """
        return self._gals_per_arcmin2

    @property
    def gals_per_steradian(self):
        """ Returns the number density of galaxies in steradian
        """
        return self._gals_per_arcmin2 * steradian_to_arcmin2

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        children = (self.params, self._gals_per_arcmin2)
        aux_data = self.config
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        args, gals_per_arcmin2 = children
        return cls(*args, gals_per_arcmin2=gals_per_arcmin2, **aux_data)


@register_pytree_node_class
class smail_nz(redshift_distribution):
    """ Defines a smail distribution with these arguments
    Parameters:
    -----------
    a:

    b:

    z0:

    gals_per_arcmin2: number of galaxies per sq arcmin
    """

    def pz_fn(self, z):
        a, b, z0 = self.params
        return z ** a * np.exp(-((z / z0) ** b))


@register_pytree_node_class
class kde_nz(redshift_distribution):
    """ A redshift distribution based on a KDE estimate of the nz of a
    given catalog currently uses a Gaussian kernel.
    TODO: add more if necessary

    Parameters:
    -----------
    zcat: redshift catalog
    weights: weight for each galaxy between 0 and 1

    Configuration:
    --------------
    bw: Bandwidth for the KDE

    Example:
    nz = kde_nz(redshift_catalog, w, bw=0.1)
    """

    def _kernel(self, bw, X, x):
        """ Gaussian kernel for KDE
        """
        return (1.0 / np.sqrt(2 * np.pi) / bw) * np.exp(
            -((X - x) ** 2) / (bw ** 2 * 2.0)
        )

    def pz_fn(self, z):
        # Extract parameters
        zcat, weight = self.params[:2]
        w = np.atleast_1d(weight)
        q = np.sum(w)
        X = np.expand_dims(zcat, axis=-1)
        k = self._kernel(self.config["bw"], X, z)
        return np.dot(k.T, w) / (q)


@register_pytree_node_class
class systematic_shift(redshift_distribution):
    """ Implements a systematic shift in a redshift distribution
    TODO: Find a better name for this

    Arguments:
    redshift_distribution
    mean_bias
    """

    def pz_fn(self, z):
        parent_pz, bias = self.params[:2]
        return parent_pz.pz_fn(np.clip(z - bias, 0))

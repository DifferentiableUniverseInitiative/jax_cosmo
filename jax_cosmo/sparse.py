"""Support for sparse matrices composed of square blocks that are individually diagonal.

The motivating example is a Gaussian covariance matrix computed in angular_cl.
The sparse matrix is represented as a 3D array of shape (ny, nx, ndiag) composed
of ny x nx square blocks of size ndiag x ndiag.  The vector at [ny, nx] is the
diagonal of the corresponding block.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
from jax import jit
from jax import vmap


@jit
def to_dense(sparse):
    """Convert a sparse matrix to its dense equivalent.

    Parameters
    ----------
    sparse : array
        3D array of shape (ny, nx, ndiag) of block diagonal elements.

    Returns
    -------
    array
        2D array of shape (ny * ndiag, nx * ndiag) with the same dtype
        as the input array.
    """
    return np.vstack(vmap(lambda row: np.hstack(vmap(np.diag)(row)))(sparse))

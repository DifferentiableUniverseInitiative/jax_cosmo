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


def check_sparse(sparse, square=False):
    """Check for a valid sparse matrix.
    """
    sparse = np.array(sparse)
    if sparse.ndim != 3:
        raise ValueError("Expected 3D array of sparse diagonals.")
    if square and (sparse.shape[0] != sparse.shape[1]):
        raise ValueError("Can only invert a square matrix.")
    return sparse


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
    sparse = check_sparse(sparse)
    return np.vstack(vmap(lambda row: np.hstack(vmap(np.diag)(row)))(sparse))


@jit
def inv(sparse):
    """Calculate the inverse of a square matrix in sparse format.

    We currently assume that the matrix is invertible and you should not
    trust the answer unless you know this is true (because jax.numpy.linalg.inv
    has this behavior).

    Parameters
    ----------
    sparse : array
        3D array of shape (n, n, ndiag) of block diagonal elements.

    Returns
    -------
    array
        3D array of shape (n, n, ndiag) of block diagonal elements
        representing the inverse matrix.
    """
    sparse = check_sparse(sparse, square=True)
    return np.transpose(np.linalg.inv(np.transpose(sparse, (2, 0, 1))), (1, 2, 0))

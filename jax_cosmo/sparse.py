"""Support for sparse matrices composed of square blocks that are individually diagonal.

The motivating example is a Gaussian covariance matrix computed in angular_cl.
The sparse matrix is represented as a 3D array of shape (ny, nx, ndiag) composed
of ny x nx square blocks of size ndiag x ndiag.  The vector at [ny, nx] is the
diagonal of the corresponding block.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import jax.numpy as np
from jax import jit
from jax import vmap


def check_sparse(sparse, square=False):
    """Check for a valid sparse matrix.
    """
    sparse = np.asarray(sparse)
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


@jit
def vecdot(sparse, vec):
    """Multiply a sparse matrix by a vector.

    Parameters
    ----------
    sparse : array
        3D array of shape (ny, nx, ndiag) of block diagonal elements.
    vec : array
        1D array of shape (nx).

    Returns
    -------
    array
        1D array of shape (ny).
    """
    sparse = check_sparse(sparse)
    vec = np.asarray(vec)
    if vec.ndim != 1 or sparse.shape[1] * sparse.shape[2] != vec.size:
        raise ValueError("Vector has the wrong shape for this sparse matrix.")
    return vmap(
        lambda row, vec: np.sum(vmap(np.multiply)(row, vec.reshape(row.shape)), axis=0),
        in_axes=(0, None),
    )(sparse, vec).reshape(-1)


@jit
def matmul(sparse1, sparse2):
    """Multiply sparse matrices and return a sparse result.

    Parameters
    ----------
    sparse1 : array
        3D array of shape (a, b, ndiag) of block diagonal elements.
    sparse2 : array
        3D array of shape (b, c, ndiag) of block diagonal elements.

    Returns
    -------
    array
        3D array of shape (a, c, ndiag) of block diagonal elements.
    """
    sparse1 = check_sparse(sparse1)
    sparse2 = check_sparse(sparse2)
    if sparse1.shape[1] != sparse2.shape[0]:
        raise ValueError("Matrix shapes are not compatible for multiplication.")
    return vmap(
        # Sparse multiply row @ col
        vmap(
            # Sparse multiply blocks B1 and B2
            lambda B1, B2: np.sum(np.multiply(B1, B2), axis=0),
            (0, None),
            0,
        ),
        (None, 1),
        1,
    )(sparse1, sparse2)


# We split the determinant calculation for a matrix with N x N blocks
# into n pieces that can be evaluated in parallel, following eqn (2.2)
# of https://arxiv.org/abs/1112.4379.  First build a helper function
# to calculate one piece indexed by 0 <= k < N:
@functools.partial(jit, static_argnums=(1, 2, 3))
def _block_det(sparse, k, N, P):
    u = sparse[k : k + 1, k + 1 : N, 0:P]
    S = sparse[k + 1 : N, k + 1 : N, 0:P]
    v = sparse[k + 1 : N, k : k + 1, 0:P]
    Sinv_v = matmul(inv(S), v)
    return np.product(sparse[k, k] - matmul(u, Sinv_v))


@jit
def det(sparse):
    """Calculate the determinant of a sparse matrix.

    Parameters
    ----------
    sparse : array
        3D array of shape (ny, nx, ndiag) of block diagonal elements.

    Returns
    -------
    float
        Determinant result.
    """
    sparse = check_sparse(sparse, square=True)
    N, _, P = sparse.shape
    result = np.product(sparse[-1, -1])
    # The individual blocks can be calculated in any order so there
    # should be a better way to express this using lax.map but I
    # can't get it to work without "concretization" errors.
    for i in range(N - 1):
        result *= _block_det(sparse, i, N, P)
    return result

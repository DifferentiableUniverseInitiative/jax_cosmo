"""Support for sparse matrices composed of square blocks that are individually diagonal.

The motivating example is a Gaussian covariance matrix computed in angular_cl.
The sparse matrix is represented as a 3D array of shape (ny, nx, ndiag) composed
of ny x nx square blocks of size ndiag x ndiag.  The vector at [ny, nx] is the
diagonal of the corresponding block.  The memory savings is a factor of ndiag
and most algorithms are sped up by a comparable factor.

We do not assume that the corresponding dense matrix is square or symmetric, even
though a covariance has these properties, since this streamlines the implementation
for a relatively small (factor of 2) increase in memory.

This sparse format is not one of those currently supported by scipy.sparse.
The scipy.sparse dia format has a similar memory efficiency but does not take
advantage of the block structure we exploit here for efficient operations.

For dot products involving a sparse matrix, use :func:`dot` to automatically
select the correct jit-compiled algorithm, with some input validation. All
pairs of vector, dense matrix and at least one sparse matrix are
supported. The special bilinear form (dense, sparse, dense) is also supported.

You can also use the lower-level algorithms (with no input validation) directly:
 - :fun:`sparse_dot_vec`
 - :fun:`sparse_dot_dense`
 - :fun:`vec_dot_sparse`
 - :fun:`dense_dot_sparse`
 - :fun:`sparse_dot_sparse`
 - :fun:`dense_dot_sparse_dot_dense`
"""
import functools

import jax.numpy as np
from jax import jit
from jax import vmap


def is_sparse(sparse):
    """Test if the input is interpretable as a sparse matrix."""
    return np.asarray(sparse).ndim == 3


def check_sparse(sparse, square=False):
    """Check for a valid sparse matrix."""
    sparse = np.asarray(sparse)
    if sparse.ndim != 3:
        raise ValueError("Expected 3D array of sparse diagonals.")
    if square and (sparse.shape[0] != sparse.shape[1]):
        raise ValueError("Expected a square matrix.")
    return sparse


@jit
def to_dense(sparse):
    """Convert a sparse matrix to its dense equivalent.

    Parameters
    ----------
    sparse : array
        3D array of shape (a, b, ndiag) of block diagonal elements.

    Returns
    -------
    array
        2D array of shape (a * ndiag, b * ndiag) with the same dtype
        as the input array.
    """
    sparse = check_sparse(sparse)
    return np.vstack(vmap(lambda row: np.hstack(vmap(np.diag)(row)))(sparse))


@jit
def dot(*args):
    """Calculate A @ B where at least one of A or B is sparse.

    All combinations of vector, dense matrix and at least one
    sparse matrix are supported. The bilinear form A @ B @ C is
    also supported where A and C are dense and B is sparse.

    Checks the inputs and calls the appropriate *_dot_* specialized
    jit-compiled method defined below.  Input types are identified
    by their array dimension: 1 = vector, 2 = dense matrix,
    3 = sparse matrix.

    Returns a dense 1D or 2D array except where A and B are both
    sparse matrices, when the result is also a sparse matrix.

    Parameters
    ----------
    args
        2 or 3 arrays to multiply.

    Returns
    -------
    array
        Result of A @ B or A @ B @ C.
    """
    if len(args) == 2:
        A, B = args
        A = np.asarray(A)
        B = np.asarray(B)
        if is_sparse(A):
            Acols = A.shape[1] * A.shape[2]
        else:
            if A.ndim < 1 or A.ndim > 2:
                raise ValueError(f"A has invalid dimension {A.ndim} (expected 1 or 2).")
            Acols = A.shape[-1]
        if is_sparse(B):
            Brows = B.shape[0] * B.shape[2]
        else:
            if B.ndim < 1 or B.ndim > 2:
                raise ValueError(f"B has invalid dimension {B.ndim} (expected 1 or 2).")
            Brows = B.shape[0]
        if Acols != Brows:
            raise ValueError(
                f"Shapes of A {A.shape} and B {B.shape} not compatible for dot product."
            )
        if is_sparse(A):
            if is_sparse(B):
                return sparse_dot_sparse(A, B)
            else:
                return sparse_dot_vec(A, B) if B.ndim == 1 else sparse_dot_dense(A, B)
        else:
            return vec_dot_sparse(A, B) if A.ndim == 1 else dense_dot_sparse(A, B)
    elif len(args) == 3:
        A, B, C = args
        if A.ndim != 2 or B.ndim != 3 or C.ndim != 2:
            raise ValueError("Can only handle dense @ sparse @ dense bilinear form.")
        if (
            A.shape[1] != B.shape[0] * B.shape[2]
            or B.shape[1] * B.shape[2] != C.shape[0]
        ):
            raise ValueError(
                "Shapes of A {A.shape}, B {B.shape}, C {C.shape} not compatible for dot product."
            )
        return dense_dot_sparse_dot_dense(A, B, C)
    else:
        raise ValueError(f"Expected 2 or 3 input arrays but got {len(args)}.")


@jit
def sparse_dot_vec(sparse, vec):
    """Calculate M @ v where M is a sparse matrix.

    Inputs must be jax numpy arrays. No error checking is performed.
    Use :func:`dot` for a more convenient front-end with error checking.

    Parameters
    ----------
    sparse : array
        3D array of shape (a, b, ndiag) of block diagonal elements.
    vec : array
        1D array of shape (b * ndiag).

    Returns
    -------
    array
        1D array of shape (a * ndiag).
    """
    return vmap(
        lambda row, vec: np.sum(vmap(np.multiply)(row, vec.reshape(row.shape)), axis=0),
        in_axes=(0, None),
    )(sparse, vec).reshape(-1)


@jit
def sparse_dot_dense(sparse, dense):
    """Calculate A @ B where A is sparse and B is dense and return dense.

    Inputs must be jax numpy arrays. No error checking is performed.
    Use :func:`dot` for a more convenient front-end with error checking.

    Parameters
    ----------
    sparse : array
        3D array of shape (a, b, ndiag) of block diagonal elements.
    dense : array
        2D array of shape (b * ndiag, c).

    Returns
    -------
    array
        2D array of shape (a * ndiag, c).
    """
    return vmap(sparse_dot_vec, (None, 1), 1)(sparse, dense)


@jit
def vec_dot_sparse(vec, sparse):
    """Calculate vec @ M where M is a sparse matrix.

    Inputs must be jax numpy arrays. No error checking is performed.
    Use :func:`dot` for a more convenient front-end with error checking.

    Parameters
    ----------
    vec : array
        1D array of shape (a * ndiag).
    sparse : array
        3D array of shape (a, b, ndiag) of block diagonal elements.

    Returns
    -------
    array
        1D array of shape (b * ndiag).
    """
    return vmap(
        lambda vec, col: np.sum(vmap(np.multiply)(vec.reshape(col.shape), col), axis=0),
        in_axes=(None, 1),
    )(vec, sparse).reshape(-1)


@jit
def dense_dot_sparse(dense, sparse):
    """Calculate A @ B where A is dense and B is sparse and return dense.

    Inputs must be jax numpy arrays. No error checking is performed.
    Use :func:`dot` for a more convenient front-end with error checking.

    Parameters
    ----------
    dense : array
        2D array of shape (a * ndiag, b * ndiag).
    sparse : array
        3D array of shape (b, c, ndiag) of block diagonal elements.

    Returns
    -------
    array
        2D array of shape (a * ndiag, c * ndiag).
    """
    return vmap(vec_dot_sparse, (0, None), 0)(dense, sparse)


@jit
def sparse_dot_sparse(sparse1, sparse2):
    """Calculate A @ B where A and B are both sparse and return sparse.

    Inputs must be jax numpy arrays. No error checking is performed.
    Use :func:`dot` for a more convenient front-end with error checking.

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


@jit
def dense_dot_sparse_dot_dense(X, Y, Z):
    """Calculate the bilinear form X @ Y @ Z where B is sparse.

    Inputs must be jax numpy arrays. No error checking is performed.

    Parameters
    ----------
    X : array
        2D array of shape (a, b * ndiag) with dense matrix elements.
    Y : array
        3D array of shape (b, c, ndiag) with sparse matrix elements.
    Z : array
        2D array of shape (c * ndiag, d) with dense matrix elements.

    Returns
    -------
    array
        2D array of shape (a, d) with dense matrix elements.
    """
    return vmap(
        vmap(
            lambda row, sparse, col: np.dot(row, sparse_dot_vec(sparse, col)),
            (None, None, 1),
        ),
        (0, None, None),
    )(X, Y, Z)


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


# We split the determinant calculation for a matrix with N x N blocks
# into n pieces that can be evaluated in parallel, following eqn (2.2)
# of https://arxiv.org/abs/1112.4379.  First build a helper function
# to calculate one piece indexed by 0 <= k < N:
@functools.partial(jit, static_argnums=(1, 2, 3))
def _block_det(sparse, k, N, P):
    u = sparse[k : k + 1, k + 1 : N, 0:P]
    S = sparse[k + 1 : N, k + 1 : N, 0:P]
    v = sparse[k + 1 : N, k : k + 1, 0:P]
    Sinv_v = sparse_dot_sparse(inv(S), v)
    M = sparse[k, k] - sparse_dot_sparse(u, Sinv_v)
    sign = np.product(np.sign(M))
    logdet = np.sum(np.log(np.abs(M)))
    return sign, logdet


@jit
def slogdet(sparse):
    """Calculate the log(determinant) of a sparse matrix.

    Based on equation (2.2) of https://arxiv.org/abs/1112.4379

    For a zero sparse matrix, the result of this computation is
    currently undefined and will return nan.
    TODO: support null matrix as input.

    Parameters
    ----------
    sparse : array
        3D array of shape (ny, nx, ndiag) of block diagonal elements.

    Returns
    -------
    tuple
        Tuple (sign, logdet) such that sign * exp(logdet) is the
        determinant.
    """
    sparse = check_sparse(sparse, square=True)
    N, _, P = sparse.shape
    sign = np.product(np.sign(sparse[-1, -1]))
    logdet = np.sum(np.log(np.abs(sparse[-1, -1])))
    # The individual blocks can be calculated in any order so there
    # should be a better way to express this using lax.map but I
    # can't get it to work without "concretization" errors.
    for i in range(N - 1):
        s, ld = _block_det(sparse, i, N, P)
        sign *= s
        logdet += ld
    return sign, logdet


@jit
def det(sparse):
    """Calculate the determinant of a sparse matrix.

    Uses :func:`slogdet`.

    For a zero sparse matrix, the result of this computation is
    currently undefined and will return nan.

    Parameters
    ----------
    sparse : array
        3D array of shape (ny, nx, ndiag) of block diagonal elements.

    Returns
    -------
    float
        Determinant result.
    """
    sign, logdet = slogdet(sparse)
    return sign * np.exp(logdet)

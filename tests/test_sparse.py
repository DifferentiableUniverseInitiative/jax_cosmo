import jax.numpy as jnp
import numpy as numpy
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from numpy.testing import assert_raises

from jax_cosmo.sparse import *


def test_to_dense():
    X_sparse = jnp.array(
        [[[1, 2, 3], [4, 5, 6], [-1, -2, -3]], [[1, 2, 3], [-4, -5, -6], [7, 8, 9]]]
    )
    X_dense = to_dense(X_sparse)
    X_answer = jnp.array(
        [
            [1, 0, 0, 4, 0, 0, -1, 0, 0],
            [0, 2, 0, 0, 5, 0, 0, -2, 0],
            [0, 0, 3, 0, 0, 6, 0, 0, -3],
            [1, 0, 0, -4, 0, 0, 7, 0, 0],
            [0, 2, 0, 0, -5, 0, 0, 8, 0],
            [0, 0, 3, 0, 0, -6, 0, 0, 9],
        ]
    )
    assert_array_equal(X_dense, X_answer)

    with assert_raises(ValueError):
        to_dense([1, 2, 3])

    with assert_raises(ValueError):
        to_dense(jnp.ones((2, 3, 4, 5)))


def test_dot():
    X1 = [[[1.0, 2], [3, 4], [5, 6]], [[4, 5], [6, 7], [8, 9]]]
    X2 = [[[1.0, -2], [3, -4]], [[5, 4], [6, -7]], [[5, 6], [9, 8]]]
    X1d = to_dense(X1)
    X2d = to_dense(X2)
    v1 = np.arange(6)
    v2 = np.arange(4)

    assert_allclose(X2d @ v2, dot(X2, v2))
    assert_allclose(X1d @ v1, dot(X1, v1))
    assert_allclose(v2 @ X1d, dot(v2, X1))
    assert_allclose(v1 @ X2d, dot(v1, X2))
    assert_allclose(X1d @ X2d, dot(X1, X2d))
    assert_allclose(X1d @ X2d, dot(X1d, X2))
    assert_allclose(X1d @ X2d, to_dense(dot(X1, X2)))
    assert_allclose(X2d @ X1d, to_dense(dot(X2, X1)))


def test_bilinear():
    X1 = [[[1.0, 2], [3, 4], [5, 6]], [[4, 5], [6, 7], [8, 9]]]
    X2 = [[[1.0, -2], [3, -4]], [[5, 4], [6, -7]], [[5, 6], [9, 8]]]
    X1d = to_dense(X1)
    X2d = to_dense(X2)
    X12 = dot(X2, X1)
    X21 = dot(X1, X2)
    X12d = X2d @ X1d
    X21d = X1d @ X2d
    assert_allclose(X1d @ X12d @ X2d, dot(X1d, X12, X2d))
    assert_allclose(X2d @ X21d @ X1d, dot(X2d, X21, X1d))


def test_inv():
    X_sparse = jnp.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [2.0, 2.0]]])
    X_inv_sparse = inv(X_sparse)
    X_answer = jnp.array([[[2.0, 2.0], [-1.0, -1.0]], [[-1.0, -1.0], [1.0, 1.0]]])
    assert_allclose(X_inv_sparse, X_answer)

    with assert_raises(ValueError):
        inv(jnp.ones((2, 3, 4)))


def test_det():
    X = np.array(
        [
            [[1, 2, 3], [4, 5, 6], [-1, 7, -2]],
            [[1, 2, 3], [-4, -5, -6], [2, -3, 9]],
            [[7, 8, 9], [5, -4, 6], [-3, -2, -1]],
        ]
    )
    assert_array_equal(-det(-X), det(X))
    # TODO: Add proper support for 0 matrix
    # assert_array_equal(det(0.0 * X), 0.0)
    assert_allclose(det(X), np.linalg.det(to_dense(X)), rtol=1e-6)

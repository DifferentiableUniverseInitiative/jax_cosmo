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


def test_inv():
    X_sparse = jnp.array([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [2.0, 2.0]]])
    X_inv_sparse = inv(X_sparse)
    X_answer = jnp.array([[[2.0, 2.0], [-1.0, -1.0]], [[-1.0, -1.0], [1.0, 1.0]]])
    assert_allclose(X_inv_sparse, X_answer)

    with assert_raises(ValueError):
        inv(jnp.ones((2, 3, 4)))


def test_vecdot():
    X_sparse = [
        [[1, 2, 3], [4, 5, 6], [-1, -2, -3]],
        [[1, 2, 3], [-4, -5, -6], [7, 8, 9]],
    ]
    y_in = [1, 0.1, -1, 2, 0.2, -2, 3, 0.3, -3]
    y_out = to_dense(X_sparse).dot(jnp.array(y_in))
    assert_allclose(y_out, vecdot(X_sparse, y_in))

    with assert_raises(ValueError):
        vecdot(X_sparse, jnp.ones(5))


def test_matmul():
    X1 = [[[1.0, 2, 3], [4, 5, 6], [0, -1, 0]], [[-4, -5, -6], [3, 2, 1], [1, 0, 1]]]
    X2 = [
        [[1.0, 2, 3], [4, 5, 6]],
        [[-4, -5, -6], [3, 2, 1]],
        [[-4, -5, -6], [3, 2, 1]],
    ]
    assert_allclose(to_dense(matmul(X1, X2)), to_dense(X1) @ to_dense(X2))


def test_det():
    X = [
        [[1, 2, 3], [4, 5, 6], [-1, 7, -2]],
        [[1, 2, 3], [-4, -5, -6], [2, -3, 9]],
        [[7, 8, 9], [5, -4, 6], [-3, -2, -1]],
    ]
    assert_allclose(det(X), np.linalg.det(to_dense(X)), rtol=1e-6)

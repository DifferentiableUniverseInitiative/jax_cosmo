import jax.numpy as jnp
import numpy as numpy
from numpy.testing import assert_allclose, assert_array_equal, assert_raises

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

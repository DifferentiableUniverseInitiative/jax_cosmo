# This module tests the InterpolatedUnivariateSpline implementation against
# SciPy
from jax.config import config

config.update("jax_enable_x64", True)
import jax
import jax.numpy as np
from numpy.testing import assert_allclose

from jax_cosmo.scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline as RefSpline


def _testing_function(x):
    return np.sin(x) ** 2


def test_linear_spline():
    # We sample some irregularly sampled points
    x = np.logspace(-2, 1, 64)
    y = _testing_function(x)

    spl = InterpolatedUnivariateSpline(x, y, k=1)
    spl_ref = RefSpline(x, y, k=1)

    # Vector of points at which to interpolate, note that this goes outside of
    # the interpolation data, so we are also testing extrapolation
    t = np.linspace(-1, 11, 128)

    assert_allclose(spl_ref(t), spl(t), rtol=1e-10)

    # Test the antiderivative, up to integration constant
    a = spl_ref.antiderivative()(t) - spl_ref.antiderivative()(0.01)
    b = spl.antiderivative(t) - spl.antiderivative(0.01)
    assert_allclose(a, b, rtol=1e-10)


def test_quadratic_spline():
    # We sample some irregularly sampled points
    x = np.logspace(-2, 1, 64)
    y = _testing_function(x)

    spl = InterpolatedUnivariateSpline(x, y, k=2)
    spl_ref = RefSpline(x, y, k=2)

    # Vector of points at which to interpolate, note that this goes outside of
    # the interpolation data, so we are also testing extrapolation
    t = np.linspace(-1, 11, 128)

    assert_allclose(spl_ref(t), spl(t), rtol=1e-10)

    # Test the antiderivative, up to integration constant
    a = spl_ref.antiderivative()(t) - spl_ref.antiderivative()(0.01)
    b = spl.antiderivative(t) - spl.antiderivative(0.01)
    assert_allclose(a, b, rtol=1e-10)


def test_cubic_spline():
    # We sample some irregularly sampled points
    x = np.logspace(-2, 1, 64)
    y = _testing_function(x)

    spl = InterpolatedUnivariateSpline(x, y, k=3)
    spl_ref = RefSpline(x, y, k=3)

    # Vector of points at which to interpolate, note that this goes outside of
    # the interpolation data, so we are also testing extrapolation
    t = np.linspace(-1, 11, 128)

    assert_allclose(spl_ref(t), spl(t), rtol=1e-10)

    # Test the antiderivative, up to integration constant
    a = spl_ref.antiderivative()(t) - spl_ref.antiderivative()(0.01)
    b = spl.antiderivative(t) - spl.antiderivative(0.01)
    assert_allclose(a, b, rtol=1e-10)


def test_spline_pytree():
    """
    Test that we can interpolate over pytrees.
    """

    # Time and data structure to interpolate over.
    ts = np.linspace(0, 1, 10)
    us = {
        "a": np.linspace(0.0, 1.0, 10),
        "b": {
            "b0": np.linspace(0.0, 0.1, 10),
            "b1": np.linspace(0.0, 0.2, 10),
        },
    }

    # Generate a pytree of splines with the same structure as "us".
    spline_order = 1
    spline_tree = jax.tree_util.tree_map(
        lambda u: InterpolatedUnivariateSpline(ts, u, spline_order), us
    )

    def eval_splines(t):
        return jax.tree_util.tree_map(
            lambda sp: sp(t),
            spline_tree,
            is_leaf=lambda obj: isinstance(obj, InterpolatedUnivariateSpline),
        )

    # Evaluate the splines at t=0.0.
    out0 = eval_splines(0.0)
    assert out0 == {
        "a": 0.0,
        "b": {
            "b0": 0.0,
            "b1": 0.0,
        },
    }

    # Evaluate the splines at t=0.5.
    out05 = eval_splines(0.5)
    assert out05 == {
        "a": 0.5,
        "b": {
            "b0": 0.05,
            "b1": 0.1,
        },
    }

    # Evaluate the splines at t=1.0.
    out1 = eval_splines(1.0)
    assert out1 == {
        "a": 1.0,
        "b": {
            "b0": 0.1,
            "b1": 0.2,
        },
    }

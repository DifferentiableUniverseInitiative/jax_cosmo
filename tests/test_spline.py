# This module tests the InterpolatedUnivariateSpline implementation against
# SciPy
from jax.config import config

config.update("jax_enable_x64", True)
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

# This module contains some missing ops from jax
import functools
import jax.numpy as np
from jax import jit
from jax import vmap
from jax import grad
from jax import ops

__all__ = ["interp"]

@functools.partial(vmap, in_axes=(0, None, None))
def interp(x, xp, fp):
  """
  Simple equivalent of np.interp that compute a linear interpolation.

  We are not doing any checks, so make sure your query points are lying
  inside the array.

  TODO: Implement proper interpolation!

  x, xp, fp need to be 1d arrays
  """
  # First we find the nearest neighbour
  ind = np.argmin((x - xp)**2)

  # Perform linear interpolation
  ind = np.clip(ind, 1, len(xp)-2)

  xi = xp[ind]
  # Figure out if we are on the right or the left of nearest
  s = np.sign(np.clip(x, xp[1], xp[-2]) - xi).astype(np.int64)
  a = (fp[ind + np.copysign(1,s)] - fp[ind])/(xp[ind+ np.copysign(1,s)] - xp[ind])
  b = fp[ind] - a*xp[ind]
  return a*x + b


class InterpolatedUnivariateSpline(object):
    def __init__(self, x, y, k, endpoints="natural"):
        """JAX implementation of kth-order spline interpolation.

        This class aims to reproduce scipy's InterpolatedUnivariateSpline
        functionality using JAX. Not all of the original class's features
        have been implemented yet, notably
        - `w`    : no weights are used in the spline fitting.
        - `bbox` : we assume the boundary to always be [x[0], x[-1]].
        - `ext`  : extrapolation is always active, i.e., `ext` = 0.
        - `k`    : orders `k` > 3 are not available.
        - `check_finite` : no such check is performed.

        (The relevant lines from the original docstring have been included
        in the following.)

        Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
        Spline function passes through all provided points. Equivalent to
        `UnivariateSpline` with s = 0.

        Parameters
        ----------
        x : (N,) array_like
            Input dimension of data points -- must be strictly increasing
        y : (N,) array_like
            input dimension of data points
        k : int, optional
            Degree of the smoothing spline.  Must be 1 <= `k` <= 3.
        endpoints : str, optional, one of {'natural', 'not-a-knot'}
            Endpoint condition for cubic splines, i.e., `k` = 3.
            'natural' endpoints enforce a vanishing second derivative
            of the spline at the two endpoints, while 'not-a-knot'
            ensures that the third derivatives are equal for the two
            left-most `x` of the domain, as well as for the two
            right-most `x`. The original scipy implementation uses
            'not-a-knot'.

        See Also
        --------
        UnivariateSpline : Superclass -- allows knots to be selected by a
            smoothing condition
        LSQUnivariateSpline : spline for which knots are user-selected
        splrep : An older, non object-oriented wrapping of FITPACK
        splev, sproot, splint, spalde
        BivariateSpline : A similar class for two-dimensional spline interpolation

        Notes
        -----
        The number of data points must be larger than the spline degree `k`.

        The general form of the spline can be written as
            f[i](t) = a[i] + b[i] * t + c[i] * t**2 + d[i] * t**3,
            0 <= t <= 1,
            i = 0, ..., m-1,
        where d = 0 for `k` = 2, and c = d = 0 for `k` = 1.

        The unknown coefficients (a, b, c, d) define a symmetric, diagonal
        linear system of equations, Ab = s, where b are the values of the first
        derivatives of the spline at anchor points x (i.e., t = 0). In all
        three `k` cases, the coefficients defining each spline piece can be
        expressed in terms of only b[i], b[i+1], y[i], and y[i+1]. They are
        solved for using `np.linalg.solve` when `k` = {2, 3}.

        """
        # Verify inputs
        k = int(k)
        assert k in (1, 2, 3), "Order k must be in {1, 2, 3}."
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        assert len(x) == len(y), "Input arrays must be the same length."
        assert x.ndim == 1 and y.ndim == 1, "Input arrays must be 1D."
        n_data = len(x)

        # Build the matrix for the linear system of equations depending on k
        if k == 2:
            assert n_data > 2, "Not enough input points for quadratic spline."

            # Construct the bi-diagonal matrix A
            main_diag = np.ones(n_data)
            off_diag = np.ones(n_data - 1)
            A = np.diag(main_diag) + np.diag(off_diag, k=1)

            # Build the RHS vector s
            s = np.zeros(n_data)
            s = ops.index_update(s, ops.index[:-1], 2 * (y[1:] - y[:-1]))

        if k == 3:
            assert n_data > 3, "Not enough input points for cubic spline."

            # Construct the tri-diagonal matrix A
            main_diag = 2 * np.ones(n_data)
            main_diag = ops.index_update(main_diag, ops.index[1:-1], 4)
            off_diag = np.ones(n_data - 1)
            A = np.diag(main_diag) + np.diag(off_diag, k=-1) + np.diag(off_diag, k=1)

            # Build the RHS vector s
            s = np.zeros(n_data)
            s = ops.index_update(s, ops.index[1:-1], 3 * (y[2:] - y[:-2]))
            s = ops.index_update(s, ops.index[0], 3 * (y[1] - y[0]))
            s = ops.index_update(s, ops.index[-1], 3 * (y[-1] - y[-2]))

            # Modify first and last equations depending on endpoints
            if endpoints not in ("natural", "not-a-knot"):
                print("Warning : endpoints not recognized. Using natural.")
            if endpoints == "not-a-knot":
                A = ops.index_update(A, ops.index[0, :3], [1, 0, -1])
                A = ops.index_update(A, ops.index[-1, -3:], [1, 0, -1])
                s = ops.index_update(s, ops.index[0],
                                     2 * (2 * y[1] - y[0] - y[2]))
                s = ops.index_update(s, ops.index[-1],
                                     2 * (2 * y[-2] - y[-3] - y[-1]))

        # Compute spline coefficients by solving the system
        if k == 1:
            assert n_data > 1, "Not enough input points for linear spline."
            self._b = y[1:] - y[:-1]
        else:
            self._b = np.linalg.solve(A, s)

        # Save for evaluations
        self.k = k
        self.x = x
        self.y = y
        self.n_data = n_data

    @functools.partial(jit, static_argnums=(0,))
    def __call__(self, x):
        """Jitted evaluation of the spline.

        Notes
        -----
        No extrapolation is currently implemented. Input values
        should lie within the bounds x[0] <= x <= x[-1].

        """
        if self.k == 1:
            t, h, a, b = self._compute_coeffs(x)
            result = a + b * t

        if self.k == 2:
            t, h, a, b, c = self._compute_coeffs(x)
            result = a + b * t + c * np.power(t, 2)

        if self.k == 3:
            t, h, a, b, c, d = self._compute_coeffs(x)
            result = a + b * t + c * np.power(t, 2) + d * np.power(t, 3)

        return result

    @functools.partial(jit, static_argnums=(0,))
    def _compute_coeffs(self, x):
        """Compute spline coeffs and parameter t for a given x.

        Notes
        -----
        An implicit extrapolation occurs here if x is outside of
        the original domain of fixed interpolation points. If
        x is less than the left-most point, f[0] is used for the
        evaluation; similarly for x beyond the right-most point.

        """
        # Determine the interval that x lies in
        ind = np.digitize(x, self.x) - 1
        # Include the right endpoint in spline piece C[m-1]
        ind = np.clip(ind, 0, self.n_data - 2)


        h = self.x[ind + 1] - self.x[ind]
        t = (x - self.x[ind]) / h
        a = self.y[ind]
        b = self._b[ind]
        b1 = self._b[ind + 1]

        if self.k == 1:
            result = (t, h, a, b)

        if self.k == 2:
            c = 0.5 * (b1 - b)
            result = (t, h, a, b, c)

        if self.k == 3:
            c = 3 * (self.y[ind + 1] - self.y[ind]) - 2 * b - b1
            d = 2 * (self.y[ind] - self.y[ind + 1]) + b + b1
            result = (t, h, a, b, c, d)

        return result

    @functools.partial(jit, static_argnums=(0, 2,))
    def derivative(self, x, n):
        """Jitted analytic nth derivative of the spline.

        The spline has derivatives up to its order k.

        """
        assert n in range(self.k + 1), "Invalid n."

        if n == 0:
            result = self.__call__(x)
        else:
            # Linear
            if self.k == 1:
                t, h, a, b = self._compute_coeffs(x)
                result = b / h

            # Quadratic
            if self.k == 2:
                t, h, a, b, c = self._compute_coeffs(x)
                if n == 1:
                    result = (b + 2 * c * t) / h
                if n == 2:
                    result = 2 * c / np.power(h, 2)

            # Cubic
            if self.k == 3:
                t, h, a, b, c, d = self._compute_coeffs(x)
                if n == 1:
                    result = (b + 2 * c * t + 3 * d * np.power(t, 2)) / h
                if n == 2:
                    result = (2 * c + 6 * d * t) / np.power(h, 2)
                if n == 3:
                    result = 6 * d / np.power(h, 3)

        return result

    @functools.partial(jit, static_argnums=(0, 2,))
    @functools.partial(vmap, in_axes=(None, 0, None))
    def autodiff(self, x, n):
        """Jitted autodiff nth derivative of the spline.

        The spline has derivatives up to its order k.

        """
        assert n in range(self.k + 1), "Invalid n."

        if n == 0:
            return self.__call__(x)
        else:
            if n == 1:
                return grad(self.__call__)(x)
            if n == 2:
                return grad(grad(self.__call__))(x)
            if n == 3:
                return grad(grad(grad(self.__call__)))(x)

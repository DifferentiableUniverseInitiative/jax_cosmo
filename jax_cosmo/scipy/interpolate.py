# This module contains some missing ops from jax
import functools
import jax.numpy as np
from jax.numpy import zeros, ones, array, concatenate
from jax import jit
from jax import vmap
from jax import grad

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
    def __init__(self, x, y, k, endpoints="not-a-knot"):
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
          f[i](x) = a[i] + b[i](x - x[i]) + c[i](x - x[i])^2 + d[i](x - x[i])^3,
          i = 0, ..., n-1,
        where d = 0 for `k` = 2, and c = d = 0 for `k` = 1.

        The unknown coefficients (a, b, c, d) define a symmetric, diagonal
        linear system of equations, Az = s, where z = b for `k` = 1 and `k` = 2,
        and z = c for `k` = 3. In each case, the coefficients defining each
        spline piece can be expressed in terms of only z[i], z[i+1],
        y[i], and y[i+1]. The coefficients are solved for using
        `np.linalg.solve` when `k` = 2 and `k` = 3.

        """
        # Verify inputs
        k = int(k)
        assert k in (1, 2, 3), "Order k must be in {1, 2, 3}."
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        assert len(x) == len(y), "Input arrays must be the same length."
        assert x.ndim == 1 and y.ndim == 1, "Input arrays must be 1D."
        n_data = len(x)

        # Difference vectors
        h = np.diff(x)  # x[i+1] - x[i] for i=0,...,n-1
        p = np.diff(y)  # y[i+1] - y[i]

        # Build the linear system of equations depending on k
        # (No matrix necessary for k=1)
        if k == 1:
            assert n_data > 1, "Not enough input points for linear spline."
            self._b = p / h

        if k == 2:
            assert n_data > 2, "Not enough input points for quadratic spline."

            # Construct the bi-diagonal matrix A
            A = np.diag(ones(n_data)) + np.diag(ones(n_data - 1), k=-1)

            # Build the RHS vector s
            s = concatenate((array([0.]), 2 * p / h))

            # Compute spline coefficients by solving the system
            self._b = np.linalg.solve(A, s)

        if k == 3:
            assert n_data > 3, "Not enough input points for cubic spline."

            if endpoints not in ("natural", "not-a-knot"):
                print("Warning : endpoints not recognized. Using natural.")
                endpoints = "natural"

            # Special values for the first and last equations
            zero = array([0.])
            one = array([1.])
            A00 = one if endpoints == "natural" else array([h[1]])
            A01 = zero if endpoints == "natural" else array([-(h[0] + h[1])])
            A02 = zero if endpoints == "natural" else array([h[0]])
            ANN = one if endpoints == "natural" else array([h[-2]])
            AN1 = -one if endpoints == "natural" else array([-(h[-2] + h[-1])])  # A[N, N-1]
            AN2 = zero if endpoints == "natural" else array([h[-1]])  # A[N, N-2]

            # Construct the tri-diagonal matrix A
            A = np.diag(concatenate((A00, 2 * (h[:-1] + h[1:]), ANN)))
            upper_diag1 = np.diag(concatenate((A01, h[1:])), k=1)
            upper_diag2 = np.diag(concatenate((A02, zeros(n_data - 3))), k=2)
            lower_diag1 = np.diag(concatenate((h[:-1], AN1)), k=-1)
            lower_diag2 = np.diag(concatenate((zeros(n_data - 3), AN2)), k=-2)
            A += upper_diag1 + upper_diag2 + lower_diag1 + lower_diag2

            # Construct RHS vector s
            center = 3 * (p[1:] / h[1:] - p[:-1] / h[:-1])
            s = concatenate((zero, center, zero))

            # Compute spline coefficients by solving the system
            self._c = np.linalg.solve(A, s)

        # Save for evaluations
        self.k = k
        self.x = x
        self._h = h
        self.y = y
        self.n_data = n_data

    @functools.partial(jit, static_argnums=(0,))
    def __call__(self, x):
        """Jitted evaluation of the spline.

        Notes
        -----
        Values are extrapolated if x is outside of the original domain
        of knots. If x is less than the left-most knot, the spline piece
        f[0] is used for the evaluation; similarly for x beyond the
        right-most point.

        """
        if self.k == 1:
            t, a, b = self._compute_coeffs(x)
            result = a + b * t

        if self.k == 2:
            t, a, b, c = self._compute_coeffs(x)
            result = a + b * t + c * np.power(t, 2)

        if self.k == 3:
            t, a, b, c, d = self._compute_coeffs(x)
            result = a + b * t + c * np.power(t, 2) + d * np.power(t, 3)

        return result

    @functools.partial(jit, static_argnums=(0,))
    def _compute_coeffs(self, x):
        """Compute the spline coefficients for a given x."""
        # Determine the interval that x lies in
        ind = np.digitize(x, self.x) - 1
        # Include the right endpoint in spline piece C[m-1]
        ind = np.clip(ind, 0, self.n_data - 2)

        t = x - self.x[ind]
        a = self.y[ind]

        if self.k == 1:
            result = (t, a, self._b[ind])

        if self.k == 2:
            # Necessities
            h = self._h[ind]
            a1 = self.y[ind + 1]
            b = self._b[ind]

            # Remaining coefficient
            c = (a1 - a) / np.power(h, 2) - b / h

            result = (t, a, b, c)

        if self.k == 3:
            # Necessities
            h = self._h[ind]
            a1 = self.y[ind + 1]
            c = self._c[ind]
            c1 = self._c[ind + 1]

            # Remaining coefficients of the spline
            b = (a1 - a) / h - (2 * c + c1) * h / 3.
            d = (c1 - c) / (3 * h)

            result = (t, a, b, c, d)

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
                t, a, b = self._compute_coeffs(x)
                result = b

            # Quadratic
            if self.k == 2:
                t, a, b, c = self._compute_coeffs(x)
                if n == 1:
                    result = b + 2 * c * t
                if n == 2:
                    result = 2 * c

            # Cubic
            if self.k == 3:
                t, a, b, c, d = self._compute_coeffs(x)
                if n == 1:
                    result = b + 2 * c * t + 3 * d * np.power(t, 2)
                if n == 2:
                    result = 2 * c + 6 * d * t
                if n == 3:
                    result = 6 * d

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

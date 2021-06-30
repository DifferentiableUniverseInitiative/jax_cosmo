from functools import partial

import jax
import jax.numpy as np
from jax import jit
from jax import vmap

__all__ = ["romb", "simps", "TrapezoidalQuad", "ClenshawCurtisQuad"]


# Romberg quadratures for numeric integration.
#
# Written by Scott M. Ransom <ransom@cfa.harvard.edu>
# last revision: 14 Nov 98
#
# Cosmetic changes by Konrad Hinsen <hinsen@cnrs-orleans.fr>
# last revision: 1999-7-21
#
# Adapted to scipy by Travis Oliphant <oliphant.travis@ieee.org>
# last revision: Dec 2001

#
# JEC 30-June-2021
# introduce Quadrature base class with two inherited classes
# o TrapezoidalQuad   :  Trapezoidal quadrature
# o ClenshawCurtisQuad:  Clenshaw-Curtis quadrature
# 
#


def _difftrap1(function, interval):
    """
    Perform part of the trapezoidal rule to integrate a function.
    Assume that we had called difftrap with all lower powers-of-2
    starting with 1.  Calling difftrap only returns the summation
    of the new ordinates.  It does _not_ multiply by the width
    of the trapezoids.  This must be performed by the caller.
        'function' is the function to evaluate (must accept vector arguments).
        'interval' is a sequence with lower and upper limits
                   of integration.
        'numtraps' is the number of trapezoids to use (must be a
                   power-of-2).
    """
    return 0.5 * (function(interval[0]) + function(interval[1]))


def _difftrapn(function, interval, numtraps):
    """
    Perform part of the trapezoidal rule to integrate a function.
    Assume that we had called difftrap with all lower powers-of-2
    starting with 1.  Calling difftrap only returns the summation
    of the new ordinates.  It does _not_ multiply by the width
    of the trapezoids.  This must be performed by the caller.
        'function' is the function to evaluate (must accept vector arguments).
        'interval' is a sequence with lower and upper limits
                   of integration.
        'numtraps' is the number of trapezoids to use (must be a
                   power-of-2).
    """
    numtosum = numtraps // 2
    h = (1.0 * interval[1] - 1.0 * interval[0]) / numtosum
    lox = interval[0] + 0.5 * h
    points = lox + h * np.arange(0, numtosum)
    s = np.sum(function(points))
    return s


def _romberg_diff(b, c, k):
    """
    Compute the differences for the Romberg quadrature corrections.
    See Forman Acton's "Real Computing Made Real," p 143.
    """
    tmp = 4.0 ** k
    return (tmp * c - b) / (tmp - 1.0)


def romb(function, a, b, args=(), divmax=6, return_error=False):
    """
    Romberg integration of a callable function or method.
    Returns the integral of `function` (a function of one variable)
    over the interval (`a`, `b`).
    If `show` is 1, the triangular array of the intermediate results
    will be printed.  If `vec_func` is True (default is False), then
    `function` is assumed to support vector arguments.
    Parameters
    ----------
    function : callable
        Function to be integrated.
    a : float
        Lower limit of integration.
    b : float
        Upper limit of integration.
    Returns
    -------
    results  : float
        Result of the integration.
    Other Parameters
    ----------------
    args : tuple, optional
        Extra arguments to pass to function. Each element of `args` will
        be passed as a single argument to `func`. Default is to pass no
        extra arguments.
    divmax : int, optional
        Maximum order of extrapolation. Default is 10.
    See Also
    --------
    fixed_quad : Fixed-order Gaussian quadrature.
    quad : Adaptive quadrature using QUADPACK.
    dblquad : Double integrals.
    tplquad : Triple integrals.
    romb : Integrators for sampled data.
    simps : Integrators for sampled data.
    cumtrapz : Cumulative integration for sampled data.
    ode : ODE integrator.
    odeint : ODE integrator.
    References
    ----------
    .. [1] 'Romberg's method' http://en.wikipedia.org/wiki/Romberg%27s_method
    Examples
    --------
    Integrate a gaussian from 0 to 1 and compare to the error function.
    >>> from scipy import integrate
    >>> from scipy.special import erf
    >>> gaussian = lambda x: 1/np.sqrt(np.pi) * np.exp(-x**2)
    >>> result = integrate.romberg(gaussian, 0, 1, show=True)
    Romberg integration of <function vfunc at ...> from [0, 1]
    ::
       Steps  StepSize  Results
           1  1.000000  0.385872
           2  0.500000  0.412631  0.421551
           4  0.250000  0.419184  0.421368  0.421356
           8  0.125000  0.420810  0.421352  0.421350  0.421350
          16  0.062500  0.421215  0.421350  0.421350  0.421350  0.421350
          32  0.031250  0.421317  0.421350  0.421350  0.421350  0.421350  0.421350
    The final result is 0.421350396475 after 33 function evaluations.
    >>> print("%g %g" % (2*result, erf(1)))
    0.842701 0.842701
    """
    vfunc = jit(lambda x: function(x, *args))

    n = 1
    interval = [a, b]
    intrange = b - a
    ordsum = _difftrap1(vfunc, interval)
    result = intrange * ordsum
    state = np.repeat(np.atleast_1d(result), divmax + 1, axis=-1)
    err = np.inf

    def scan_fn(carry, y):
        x, k = carry
        x = _romberg_diff(y, x, k + 1)
        return (x, k + 1), x

    for i in range(1, divmax + 1):
        n = 2 ** i
        ordsum = ordsum + _difftrapn(vfunc, interval, n)

        x = intrange * ordsum / n
        _, new_state = jax.lax.scan(scan_fn, (x, 0), state[:-1])

        new_state = np.concatenate([np.atleast_1d(x), new_state])

        err = np.abs(state[i - 1] - new_state[i])
        state = new_state

    if return_error:
        return state[i], err
    else:
        return state[i]


def simps(f, a, b, N=128):
    """Approximate the integral of f(x) from a to b by Simpson's rule.

    Simpson's rule approximates the integral \int_a^b f(x) dx by the sum:
    (dx/3) \sum_{k=1}^{N/2} (f(x_{2i-2} + 4f(x_{2i-1}) + f(x_{2i}))
    where x_i = a + i*dx and dx = (b - a)/N.

    Parameters
    ----------
    f : function
        Vectorized function of a single variable
    a , b : numbers
        Interval of integration [a,b]
    N : (even) integer
        Number of subintervals of [a,b]

    Returns
    -------
    float
        Approximation of the integral of f(x) from a to b using
        Simpson's rule with N subintervals of equal length.

    Examples
    --------
    >>> simps(lambda x : 3*x**2,0,1,10)
    1.0

    Notes:
    ------
    Stolen from: https://www.math.ubc.ca/~pwalls/math-python/integration/simpsons-rule/
    """
    if N % 2 == 1:
        raise ValueError("N must be an even integer.")
    dx = (b - a) / N
    x = np.linspace(a, b, N + 1)
    y = f(x)
    
    #print("JEC:simps) y shape:",y.shape)
    
    S = dx / 3 * np.sum(y[0:-1:2] + 4 * y[1::2] + y[2::2], axis=0)
    return S


##
# JEC 30-June-2021
##
class Quadrature():
    """
        Base class for quadratures providing node lcations, the weights and error weights
    """
    def __init__(self,order=5):
        self.order = int(order)
        self.absc, self.absw, self.errw = self.ComputeAbsWeights()
        
    def rescaleAbsWeights(self, xInmin=-1.0, xInmax=1.0, xOutmin=0.0, xOutmax=1.0):
        """
            Translate nodes,weights for [xInmin,xInmax] integral to [xOutmin,xOutmax] 
        """
        deltaXIn = xInmax-xInmin
        deltaXOut= xOutmax-xOutmin
        scale = deltaXOut/deltaXIn
        self.absw *= scale
        tmp = np.array([((xi-xInmin)*xOutmax
                         -(xi-xInmax)*xOutmin)/deltaXIn for xi in self.absc])
        self.absc=tmp

    def computeIntegral(self, func,bounds, return_error=False):
        """
            One step integral computation of \int_a^b f(x) dx \approx  \sum_i w_i f(x_i)
            a = bounds[0]
            b = bounds[1]
            f(x) = func
        """
        a = bounds[0]
        b = bounds[1]
        d = b-a
        xi = a + self.absc * d
        fi = func(xi)
        integ = d*np.sum(np.dot(self.absw,fi))
        if return_error:
            return {'val':integ,
                    'err':d*np.abs(np.sum(np.dot(self.errw,fi))),
                    'nodes':xi}
        else:
            integ = integ[...,np.newaxis]
            return integ

    
class TrapezoidalQuad(Quadrature):
    """
        Simple Trapezoidale quadrature
        nb. The order is transformed into a odd number just to get 
        an error based on a sub-sampling of the quadrature
    """
    def __init__(self,order=5):
        # 2n-1 quad
        Quadrature.__init__(self,int(2*order-1))

    def ComputeAbsWeights(self):
        x,wx = self.absweights(self.order)
        nsub = (self.order+1)/2
        xSub, wSub = self.absweights(nsub)
        errw = wx
        errw=errw.at[::2].add(-wSub)
        return x,wx,errw
    
    
    def absweights(self, n):
        h = 1./(n-1)
        x = np.arange(n,dtype=np.float32)
        w = np.ones_like(x)
        x *= h
        w *= h
        w = w.at[0].mul(0.5)
        w = w.at[-1].mul(0.5)
        return x, w

class ClenshawCurtisQuad(Quadrature):
    """
        Clenshaw-Curtis quadrature nodes and weights computed by FFT. 
        nb. The order is transformed into a odd number just to get 
        an error based on a sub-sampling of the quadrature
    """
    def __init__(self,order=5):
        # 2n-1 quad
        Quadrature.__init__(self,int(2*order-1))
        self.rescaleAbsWeights()  # rescale [-1,1] to [0,1]
    
    def ComputeAbsWeights(self):
        x,wx = self.absweights(self.order)
        nsub = (self.order+1)//2
        xSub, wSub = self.absweights(nsub)
        errw = wx
        errw=errw.at[::2].add(-wSub)
        return x,wx,errw
  
    
    def absweights(self,n):
        degree = n

        points = -np.cos((np.pi * np.arange(n)) / (n - 1))

        if n == 2:
            weights = np.array([1.0, 1.0])
            return points, weights
            
        n -= 1
        N = np.arange(1, n, 2)
        length = len(N)
        m = n - length
        v0 = np.concatenate([2.0 / N / (N - 2), np.array([1.0 / N[-1]]), np.zeros(m)])
        v2 = -v0[:-1] - v0[:0:-1]
        g0 = -np.ones(n)
        g0 = g0.at[length].add(n)
        g0 = g0.at[m].add(n)
        g = g0 / (n ** 2 - 1 + (n % 2))

        w = np.fft.ihfft(v2 + g)
        ###assert max(w.imag) < 1.0e-15
        w = w.real

        if n % 2 == 1:
            weights = np.concatenate([w, w[::-1]])
        else:
            weights = np.concatenate([w, w[len(w) - 2 :: -1]])
            
        #return
        return points, weights


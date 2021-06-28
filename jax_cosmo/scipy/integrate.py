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
# JEC Code 25-June-2021
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
        xi = a + self.absc * d   # xi is a DeviceArray
        fi = func(xi) #fi = np.array([func(x) for x in xi])
        integ = d*np.sum(np.dot(self.absw,fi))
        if return_error:
            return {'val':integ,
                    'err':d*np.abs(np.sum(np.dot(self.errw,fi))),
                    'nodes':xi}
        else:
            integ = integ[...,np.newaxis]
            return integ

    def _NOTUSED_globalAdapStrat(self, func,bounds, tol=1.e-6, MaxIter=1000, MinIter=1):
        """
            Global adaptative strategy to compute int_a^b f(x) dx
            a = bounds[0]
            b = bounds[1]
            f(x) = func
            tol: error < tol*integral
            nbre of iterations is [MinIter, MinIter]
            The idea is to split an interval [ai,bi] if its error is the largest one and if the
            global error is larger the user tolerance 
        """
        curBounds=bounds
        curValues = self.computeIntegral(func,bounds)
        integral  = curValues['val']
        error     = curValues['err']
        curResult = [curBounds,[integral,error]]
        theResult = [curResult]
                
        nIter=0
        while ( ((error >= tol*np.abs(integral)) and (nIter<MaxIter)) or (nIter<MinIter)):
            nIter += 1
            curBounds =  theResult[-1][0] #last
            xMiddle = (curBounds[0]+curBounds[1])*0.5
            # [x0, mid]
            lowerB = [curBounds[0],xMiddle]
            valLow = self.computeIntegral(func,lowerB)
            resLow = [lowerB, [valLow['val'],valLow['err']]]
            # [mid, x1]
            upperB = [xMiddle,curBounds[1]]
            valUp  = self.computeIntegral(func,upperB)
            resUp  = [upperB, [valUp['val'],valUp['err']]]
            #
            theResult.pop() #remove laast element
            theResult.append(resLow)
            theResult.append(resUp)
            #sort: the last element [[a,b],[val,err]] has the largest "err"
            theResult=sorted(theResult,key=lambda x:x[1][1])
            #Collect the differente pieces
            tmp = np.array(theResult)
            integral = np.sum(tmp[:,1,0])
            error    = np.sum(tmp[:,1,1])
        return {'val':integral,'err':error}
    
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
        errw = wx                           # np.copy(wx)
        errw=errw.at[::2].add(-wSub)             # errw[::2] -= wSub
        return x,wx,errw
    
    
    def absweights(self, n):
        h = 1./(n-1)
        x = np.arange(n,dtype=np.float32)
        w = np.ones_like(x)
        x *= h
        w *= h
        w = w.at[0].mul(0.5) #    w[0] *= 0.5
        w = w.at[-1].mul(0.5) #  w[-1]*= 0.5
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
        errw = wx                                # np.copy(wx)
        errw=errw.at[::2].add(-wSub)             # errw[::2] -= wSub
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
        g0 = g0.at[length].add(n)     # g0[length] += n
        g0 = g0.at[m].add(n)          # g0[m] += n
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

"""

#hard coded 99pts Clenshaw-Curtis quadrature
def ccquad(func,a,b):
    absc = jax.numpy.array([0.00000000e+00, 2.56891900e-04, 1.02730362e-03,
             2.31044353e-03, 4.10499309e-03, 6.40910829e-03,
             9.22042150e-03, 1.25360439e-02, 1.63525685e-02,
             2.06660735e-02, 2.54721265e-02, 3.07657890e-02,
             3.65416213e-02, 4.27936885e-02, 4.95155660e-02,
             5.67003468e-02, 6.43406479e-02, 7.24286185e-02,
             8.09559476e-02, 8.99138727e-02, 9.92931891e-02,
             1.09084259e-01, 1.19277021e-01, 1.29861001e-01,
             1.40825325e-01, 1.52158725e-01, 1.63849555e-01,
             1.75885802e-01, 1.88255099e-01, 2.00944735e-01,
             2.13941670e-01, 2.27232549e-01, 2.40803716e-01,
             2.54641224e-01, 2.68730855e-01, 2.83058130e-01,
             2.97608328e-01, 3.12366498e-01, 3.27317473e-01,
             3.42445891e-01, 3.57736207e-01, 3.73172708e-01,
             3.88739533e-01, 4.04420686e-01, 4.20200052e-01,
             4.36061419e-01, 4.51988487e-01, 4.67964890e-01,
             4.83974211e-01, 5.00000000e-01, 5.16025789e-01,
             5.32035110e-01, 5.48011513e-01, 5.63938581e-01,
             5.79799948e-01, 5.95579314e-01, 6.11260467e-01,
             6.26827292e-01, 6.42263793e-01, 6.57554109e-01,
             6.72682527e-01, 6.87633502e-01, 7.02391672e-01,
             7.16941870e-01, 7.31269145e-01, 7.45358776e-01,
             7.59196284e-01, 7.72767451e-01, 7.86058330e-01,
             7.99055265e-01, 8.11744901e-01, 8.24114198e-01,
             8.36150445e-01, 8.47841275e-01, 8.59174675e-01,
             8.70138999e-01, 8.80722979e-01, 8.90915741e-01,
             9.00706811e-01, 9.10086127e-01, 9.19044052e-01,
             9.27571382e-01, 9.35659352e-01, 9.43299653e-01,
             9.50484434e-01, 9.57206312e-01, 9.63458379e-01,
             9.69234211e-01, 9.74527874e-01, 9.79333927e-01,
             9.83647432e-01, 9.87463956e-01, 9.90779578e-01,
             9.93590892e-01, 9.95895007e-01, 9.97689556e-01,
             9.98972696e-01, 9.99743108e-01, 1.00000000e+00])

    absw=jax.numpy.array([5.20670624e-05, 5.01572524e-04, 1.03121876e-03,
             1.53700278e-03, 2.05092396e-03, 2.55733783e-03,
             3.06456773e-03, 3.56625644e-03, 4.06604219e-03,
             4.56029596e-03, 5.05093870e-03, 5.53551538e-03,
             6.01513272e-03, 6.48795208e-03, 6.95463479e-03,
             7.41370982e-03, 7.86557262e-03, 8.30899248e-03,
             8.74419731e-03, 9.17012525e-03, 9.58689541e-03,
             9.99357180e-03, 1.03902023e-02, 1.07759498e-02,
             1.11508161e-02, 1.15140450e-02, 1.18656105e-02,
             1.22048250e-02, 1.25316479e-02, 1.28454517e-02,
             1.31461910e-02, 1.34332928e-02, 1.37067144e-02,
             1.39659329e-02, 1.42109146e-02, 1.44411834e-02,
             1.46567197e-02, 1.48570915e-02, 1.50422976e-02,
             1.52119482e-02, 1.53660638e-02, 1.55042954e-02,
             1.56266879e-02, 1.57329318e-02, 1.58230990e-02,
             1.58969179e-02, 1.59544898e-02, 1.59955798e-02,
             1.60203204e-02, 1.60285123e-02, 1.60203204e-02,
             1.59955798e-02, 1.59544898e-02, 1.58969179e-02,
             1.58230990e-02, 1.57329318e-02, 1.56266879e-02,
             1.55042954e-02, 1.53660638e-02, 1.52119482e-02,
             1.50422976e-02, 1.48570915e-02, 1.46567197e-02,
             1.44411834e-02, 1.42109146e-02, 1.39659329e-02,
             1.37067144e-02, 1.34332928e-02, 1.31461910e-02,
             1.28454517e-02, 1.25316479e-02, 1.22048250e-02,
             1.18656105e-02, 1.15140450e-02, 1.11508161e-02,
             1.07759498e-02, 1.03902023e-02, 9.99357180e-03,
             9.58689541e-03, 9.17012525e-03, 8.74419731e-03,
             8.30899248e-03, 7.86557262e-03, 7.41370982e-03,
             6.95463479e-03, 6.48795208e-03, 6.01513272e-03,
             5.53551538e-03, 5.05093870e-03, 4.56029596e-03,
             4.06604219e-03, 3.56625644e-03, 3.06456773e-03,
             2.55733783e-03, 2.05092396e-03, 1.53700278e-03,
             1.03121876e-03, 5.01572524e-04, 5.20670624e-05])
    
    d = b-a
    xi = a + absc * d   # xi is a DeviceArray
    fi = func(xi) #fi = np.array([func(x) for x in xi])
    #print("JEC:ccquad) fi shape:",fi.shape)
    integ = d*np.sum(np.dot(absw,fi))
    integ = integ[...,np.newaxis]
    return integ
    
"""
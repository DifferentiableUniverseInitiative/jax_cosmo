# This module contains the various kernels for various tracers
import jax.numpy as np
from jax.experimental.ode import odeint

def get_lensing_efficiency_kernel(cosmo, dndz):
  """
  Computes the lensing kernel, returns a function of chi
  cosmo is the cosmology object
  dndz is a function that returns the dndz(z)
  returns a function that computes what we want
  """
  def kernel(a):
    def integrand(y, t, a):
      return dndz(1./t - t) * cosmo.g(a, t) # probably not enough, careful about change ofvar
    y0 = 0.
    y = odeint(integrand, y0, np.array([cosmo._amin, a]), a)
    return y[1]
  return kernel

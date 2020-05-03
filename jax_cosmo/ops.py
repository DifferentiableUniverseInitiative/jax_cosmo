# This module contains some missing ops from jax
from jax import vmap
import functools
import jax.numpy as np

@functools.partial(vmap,in_axes=(0, None, None))
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
  a1 = (fp[ind+1] - fp[ind])/(xp[ind+1] - xp[ind])
  a2 = (fp[ind] - fp[ind-1])/(xp[ind] - xp[ind-1])
  a = 0.5*(a1+a2)
  b = fp[ind] - a*xp[ind]
  return a*x + b

# This module contains some missing ops from jax
from jax import vmap
import functools
import jax.numpy as np

__all__ = ["interp"]

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
  # Figure out if we are on the right or the left of nearest
  s = np.sign(np.clip(x, xp[1], xp[-2]) - xi).astype(np.int64)
  a = (fp[ind + np.copysign(1,s)] - fp[ind])/(xp[ind+ np.copysign(1,s)] - xp[ind])
  b = fp[ind] - a*xp[ind]
  return a*x + b

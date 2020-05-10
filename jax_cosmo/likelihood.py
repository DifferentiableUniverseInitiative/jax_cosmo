# This module implements a few likelihoods useful
import jax.numpy as np
from jax_cosmo.angular_cl import gaussian_cl_covariance

def gaussian_log_likelihood(data, mu, C,
                            constant_cov=True,
                            inverse_method='inverse'):
  """
  Computes the likelihood for some cl
  """
  # Computes residuals
  r  = mu - data

  # TODO: check what is the fastest and works the best between cholesky+solve
  # and just inversion
  if inverse_method == 'inverse':
    y = np.linalg.inv(C) @ r
  elif inverse_method == 'cholesky':
    raise NotImplementedError
  else:
    raise NotImplementedError

  if constant_cov:
    return -0.5 * ( r.T @ y )
  else:
    _, logdet = np.linalg.slogdet(C)
    return -0.5 * ( r.T @ y ) - 0.5 * logdet

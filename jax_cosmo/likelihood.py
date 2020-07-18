# This module implements a few likelihoods useful
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
import jax.scipy as sp

from jax_cosmo.angular_cl import gaussian_cl_covariance


def gaussian_log_likelihood(data, mu, C, constant_cov=True, inverse_method="inverse"):
    """
    Computes the likelihood for some cl
    """
    # Computes residuals
    r = mu - data

    # TODO: check what is the fastest and works the best between cholesky+solve
    # and just inversion
    if inverse_method == "inverse":
        y = np.dot(np.linalg.inv(C), r)
    elif inverse_method == "cholesky":
        y = sp.linalg.cho_solve(sp.linalg.cho_factor(C, lower=True), r)
    else:
        raise NotImplementedError

    if constant_cov:
        return -0.5 * r.dot(y)
    else:
        _, logdet = np.linalg.slogdet(C)
        return -0.5 * r.dot(y) - 0.5 * logdet

# This module implements a few likelihoods useful
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np
import jax.scipy as sp

import jax_cosmo.sparse as sparse
from jax_cosmo.angular_cl import gaussian_cl_covariance


def gaussian_log_likelihood(data, mu, C, constant_cov=True, inverse_method="inverse"):
    """
    Computes the likelihood for some cl

    If the covariance C is sparse (according to :meth:`jax_cosmo.sparse.is_sparse`)
    use sparse inverse and determinant algorithms (and ignore ``inverse_method``).
    """
    # Computes residuals
    r = mu - data

    if sparse.is_sparse(C):
        r = r.reshape(-1, 1)
        rT_Cinv_r = sparse.dot(r.T, sparse.inv(C), r)[0, 0]
    else:
        # TODO: check what is the fastest and works the best between cholesky+solve
        # and just inversion
        if inverse_method == "inverse":
            y = np.dot(np.linalg.inv(C), r)
        elif inverse_method == "cholesky":
            y = sp.linalg.cho_solve(sp.linalg.cho_factor(C, lower=True), r)
        else:
            raise NotImplementedError
        rT_Cinv_r = r.dot(y)

    if constant_cov:
        return -0.5 * rT_Cinv_r
    else:
        if sparse.is_sparse(C):
            _, logdet = sparse.slogdet(C)
        else:
            _, logdet = np.linalg.slogdet(C)
        return -0.5 * (rT_Cinv_r - logdet)

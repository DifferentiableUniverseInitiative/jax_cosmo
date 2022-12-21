# This module implements a few useful likelihoods
import jax.numpy as np
import jax.scipy as sp

import jax_cosmo.sparse as sparse
from jax_cosmo.angular_cl import gaussian_cl_covariance


def gaussian_log_likelihood(data, mu, C, include_logdet=True, inverse_method="inverse"):
    """
    Computes the log likelihood for a given data vector under a multivariate
    Gaussian distribution.

    If the covariance C is sparse (according to :meth:`jax_cosmo.sparse.is_sparse`)
    use sparse inverse and determinant algorithms (and ignore ``inverse_method``).

    Parameters
    ----------
    data: array_like
        Data vector, with shape [N].

    mu: array_like, 1d
        Mean of the Gaussian likelihood, with shape [N].

    C: array_like or sparse matrix
        Covariance of Gaussian likelihood with shape [N,N]

    include_logdet: boolean
        Whether to include the log determinant of the covariance matrix in the
        likelihood. Can be set to False if the covariance is constant, to skip this
        costly operation (default: True)

    inverse_method: string
        Methods for computing the precision matrix. Either "inverse", "cholesky".
        Note that this option is ignored when the covariance is sparse. (default: "inverse")
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

    if not include_logdet:
        return -0.5 * rT_Cinv_r
    else:
        if sparse.is_sparse(C):
            _, logdet = sparse.slogdet(C)
        else:
            _, logdet = np.linalg.slogdet(C)
        return -0.5 * (rT_Cinv_r - logdet)

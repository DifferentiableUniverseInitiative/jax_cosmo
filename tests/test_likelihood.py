import jax.numpy as jnp
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from jax_cosmo import Planck15
from jax_cosmo import probes
from jax_cosmo.angular_cl import gaussian_cl_covariance_and_mean
from jax_cosmo.bias import constant_linear_bias
from jax_cosmo.likelihood import gaussian_log_likelihood
from jax_cosmo.redshift import smail_nz
from jax_cosmo.sparse import to_dense


def test_gaussian_log_likelihood():
    n_ell = 5
    ell = jnp.logspace(1, 3, n_ell)
    nz1 = smail_nz(1.0, 2.0, 1.0)
    nz2 = smail_nz(1.0, 2.0, 0.5)
    n_cls = 3
    P = [probes.NumberCounts([nz1, nz2], constant_linear_bias(1.0))]
    cosmo = Planck15()
    mu, cov_sparse = gaussian_cl_covariance_and_mean(cosmo, ell, P, sparse=True)
    cov_dense = to_dense(cov_sparse)
    data = 1.1 * mu
    for include_logdet in (True, False):
        loglike_sparse = gaussian_log_likelihood(
            data, mu, cov_sparse, include_logdet=include_logdet
        )
        for method in "inverse", "cholesky":
            loglike_dense = gaussian_log_likelihood(
                data,
                mu,
                cov_dense,
                include_logdet=include_logdet,
                inverse_method=method,
            )
            assert_allclose(loglike_sparse, loglike_dense, rtol=1e-6)

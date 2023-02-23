import numpy as np
from jax_cosmo.redshift import histogram_nz
from numpy.testing import assert_allclose


def test_histogram_nz():
    # samples to generate numpy histogram
    data = np.array(
        [
            0.2,
            0.2,
            1.0,
            1.0,
            1.0,
            1.5,
            2.0,
            2.0,
            3.0,
            4.0,
            4.0,
            4.0,
            4.0,
            5.0,
            5.0,
            5.0,
            5.0,
            6.0,
            6.0,
            7.0,
        ]
    )

    # numpy histogram
    bin_edges = np.array([0.0, 0.5, 0.9, 1.1, 2.0, 3.0, 4.0, 4.5, 5.0, 7.0])
    hist = np.histogram(data, bin_edges)[0]
    area_under_curve_numpy = np.sum(hist * np.diff(bin_edges))

    # jax redshift distribution histogram
    nz_hist = histogram_nz(hist, bin_edges)
    dz = 1e-3
    z_space = np.arange(0.0 - dz, 7.0 + dz, dz)
    func = nz_hist.pz_fn(z_space)
    area_under_curve_jax_redshift = np.trapz(func, z_space)

    # check area under each histogram is equal
    assert_allclose(area_under_curve_numpy, area_under_curve_jax_redshift, rtol=1e-10)

    # check normalisation correct
    normalised_nz = nz_hist(z_space)
    assert_allclose(1.0, np.trapz(normalised_nz, z_space), rtol=1e-10)

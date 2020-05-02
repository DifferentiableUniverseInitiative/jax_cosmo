# This module defines a few default cosmologies
from jax_cosmo.core import Cosmology

# Planck 2015 paper XII Table 4 final column (best fit)
Planck15 = Cosmology(
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=1.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1,
    wa=0
)

# This module defines a few default cosmologies
from functools import partial

from jax_cosmo.core import Cosmology

# To add new cosmologies, we just set the parameters to some default values using `partial`

# Planck 2015 paper XII Table 4 final column (post mean)
Planck15 = partial(Cosmology,
    # Omega_m = 0.3089,
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=0.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1.0,
    wa=0.0,)

# Planck 2018 paper VI Table 2 final column (post mean)
Planck18 = partial(Cosmology,
    # Omega_m = 0.3111,
    Omega_c=0.2607,
    Omega_b=0.0490,
    Omega_k=0.0,
    h=0.6766,
    n_s=0.9665,
    sigma8=0.8102,
    w0=-1.0,
    wa=0.0,)

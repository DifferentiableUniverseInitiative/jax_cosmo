# This module defines a few default cosmologies
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import partial

from jax_cosmo.core import Cosmology
from jax_cosmo.power import halofit

# To add new cosmologies, we just set the parameters to some default values using
# partial

# Planck 2015 paper XII Table 4 final column (best fit)
Planck15 = partial(
    Cosmology,
    Omega_c=0.2589,
    Omega_b=0.04860,
    Omega_k=0.0,
    h=0.6774,
    n_s=0.9667,
    sigma8=0.8159,
    w0=-1.0,
    wa=0.0,
)

# Shortcuts for the different halofit implementations
halofit_smith2003 = partial(halofit, prescription="smith2003")
halofit_takahashi2012 = partial(halofit, prescription="takahashi2012")

# Cosmology in JAX
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution

try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    # package is not installed
    pass

import jax_cosmo.angular_cl as cl
import jax_cosmo.background as background
import jax_cosmo.bias as bias
import jax_cosmo.likelihood as likelihood
import jax_cosmo.power as power
import jax_cosmo.probes as probes
import jax_cosmo.redshift as redshift
import jax_cosmo.transfer as transfer
from jax_cosmo.core import *
from jax_cosmo.parameters import *

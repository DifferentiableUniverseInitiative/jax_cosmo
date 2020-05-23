# This defines a few utility functions
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def z2a(z):
    """ converts from redshift to scale factor """
    return 1.0 / (1.0 + z)


def a2z(a):
    """ converts from scale factor to  redshift """
    return 1.0 / a - 1.0

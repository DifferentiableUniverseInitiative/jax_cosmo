# This defines a few utility functions
def z2a(z):
    """converts from redshift to scale factor"""
    return 1.0 / (1.0 + z)


def a2z(a):
    """converts from scale factor to  redshift"""
    return 1.0 / a - 1.0

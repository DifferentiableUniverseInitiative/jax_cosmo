from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import jax.numpy as np


class container(object):
    """
    Generic structure to trace a parameterized  function

    Paramters for the object, i.e. things that need to be traced for autodiff
    are stored as a list in self.params
    Configuration arguments, i.e. static things that do not need to be traced
    are stored as a dictionary in self.config This is for things like flags or
    type of PS or things like that.
    """

    def __init__(self, *args, **kwargs):
        self.params = args
        self.config = kwargs

    def __repr__(self):
        return str(self.params)

    # Operations for flattening/unflattening representation
    def tree_flatten(self):
        children = self.params
        aux_data = self.config
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)

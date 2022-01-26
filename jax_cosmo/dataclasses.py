from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass

from jax.tree_util import register_pytree_node


def pytree_dataclass(cls, aux_fields=None, **kwargs):
    """Register python dataclasses as custom pytree nodes.

    Parameters
    ----------
    cls : type
        Class to be registered, not a python dataclass yet.
    aux_fields : str or Sequence[str], optional
        Fields to be treated as pytree aux_data; unrecognized ones are ignored.
    kwargs
        Keyword arguments to be passed to python dataclass decorator.

    Returns
    -------
    cls : type
        Registered dataclass.

    Raises
    ------
    TypeError
        If cls is already a python dataclass.

    .. _Augmented dataclass for JAX pytree:
        https://gist.github.com/odashi/813810a5bc06724ea3643456f8d3942d

    """
    if is_dataclass(cls):
        raise TypeError("cls cannot already be a dataclass")
    cls = dataclass(cls, **kwargs)

    if aux_fields is None:
        aux_fields = []
    elif isinstance(aux_fields, str):
        aux_fields = [aux_fields]
    akeys = [field.name for field in fields(cls) if field.name in aux_fields]
    ckeys = [field.name for field in fields(cls) if field.name not in aux_fields]

    def tree_flatten(obj):
        children = [getattr(obj, key) for key in ckeys]
        aux_data = [getattr(obj, key) for key in akeys]
        return children, aux_data

    def tree_unflatten(aux_data, children):
        return cls(**dict(zip(ckeys, children)), **dict(zip(akeys, aux_data)))

    register_pytree_node(cls, tree_flatten, tree_unflatten)

    return cls

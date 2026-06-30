"""Host-side caching for JAX functions via ``pure_callback``.

The ``@caching`` decorator keeps computed tables on the Python (host) side
using an LRU ``OrderedDict``, so no JAX tracers are ever stored — eliminating
``UnexpectedTracerError`` when the same ``Cosmology`` object is reused across
nested JAX transformations (JIT → scan → while_loop → cond).

Forward evaluation goes through ``jax.pure_callback``; the backward pass
re-invokes the underlying function and differentiates with ``jax.vjp``.

Set the environment variable ``JAX_COSMO_DEACTIVATE_CACHE`` to a truthy value
(``1``/``true``/``yes``/``on``) before importing ``jax_cosmo`` to disable the
host-side cache entirely: ``@caching`` then becomes a no-op and the decorated
ODE solve runs fully in-graph (XLA-compiled, no ``pure_callback``).
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from functools import wraps

import jax
import jax.numpy as jnp
from jax import tree_util

# When truthy, ``caching`` is a no-op and decorated functions run in-graph.
# Read once at import time, so set it before ``import jax_cosmo``.
DEACTIVATE_CACHE = os.environ.get("JAX_COSMO_DEACTIVATE_CACHE", "").strip().lower() in (
    "1",
    "true",
    "yes",
    "on",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_hashable(val):
    """Convert *val* to a hashable representation for use as a cache key."""
    if isinstance(val, (int, float, str, bool, type(None))):
        return val
    if isinstance(val, (tuple, list)):
        return tuple(make_hashable(v) for v in val)
    if isinstance(val, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in val.items()))
    # JAX arrays — use raw bytes for hashing
    if hasattr(val, "tobytes"):
        return val.tobytes()
    # JAX pytree (e.g. Cosmology) — flatten and hash leaves
    try:
        leaves, treedef = tree_util.tree_flatten(val)
        return (str(treedef), tuple(make_hashable(l) for l in leaves))
    except Exception:
        return id(val)


def get_byte_size(obj):
    """Approximate byte size of a JAX pytree (sum of leaf array nbytes)."""
    try:
        leaves = tree_util.tree_leaves(obj)
        total = 0
        for l in leaves:
            if hasattr(l, "nbytes"):
                total += l.nbytes
            else:
                total += sys.getsizeof(l)
        return total
    except Exception:
        return sys.getsizeof(obj)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------


def caching(arg_name, max_entries=None, max_bytes=None):
    """Decorator that caches a function's return value on the host side.

    Parameters
    ----------
    arg_name : str
        Name of the keyword/positional argument used as the cache key
        (typically ``"cosmo"``).
    max_entries : int | None
        Maximum number of cache entries (LRU eviction).
    max_bytes : int | None
        Maximum total byte size of cached values (LRU eviction).
    """

    def decorator(fn):
        if DEACTIVATE_CACHE:
            # In-graph behaviour: no host cache, the ODE solve is compiled by XLA.
            return fn

        cache: OrderedDict = OrderedDict()
        cache_bytes: list[int] = [0]  # mutable counter in a list

        def _evict_if_needed(new_size):
            if max_entries is not None:
                while len(cache) >= max_entries:
                    _, evicted = cache.popitem(last=False)
                    cache_bytes[0] -= get_byte_size(evicted)
            if max_bytes is not None:
                while cache_bytes[0] + new_size > max_bytes and cache:
                    _, evicted = cache.popitem(last=False)
                    cache_bytes[0] -= get_byte_size(evicted)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            import inspect

            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            key_arg = bound.arguments[arg_name]

            # Build a hashable key from the target argument + remaining args
            all_args = dict(bound.arguments)
            all_args.pop(arg_name)
            cache_key = (make_hashable(key_arg), make_hashable(all_args))

            # --- forward via pure_callback -----------------------------------
            def _compute_concrete(*flat_key_arg):
                """Executed on the host with concrete values."""
                key_arg_concrete = tree_util.tree_unflatten(treedef, flat_key_arg)
                # Rebuild full kwargs
                call_kwargs = dict(all_args)
                call_kwargs[arg_name] = key_arg_concrete
                result = fn(**call_kwargs)
                return result

            flat_key_arg, treedef = tree_util.tree_flatten(key_arg)

            if cache_key in cache:
                cache.move_to_end(cache_key)
                result_cached = cache[cache_key]
            else:
                # Compute with concrete values (works outside JIT too)
                call_kwargs = dict(all_args)
                call_kwargs[arg_name] = key_arg
                result_cached = fn(**call_kwargs)

                entry_size = get_byte_size(result_cached)
                _evict_if_needed(entry_size)
                cache[cache_key] = result_cached
                cache_bytes[0] += entry_size

            # Build the result structure for pure_callback
            result_leaves, result_treedef = tree_util.tree_flatten(result_cached)
            result_shapes = [
                jax.ShapeDtypeStruct(l.shape, l.dtype) if hasattr(l, "shape") else l
                for l in result_leaves
            ]

            def _callback(*flat_key_arg):
                key_arg_cb = tree_util.tree_unflatten(treedef, flat_key_arg)
                cb_key = (make_hashable(key_arg_cb), make_hashable(all_args))
                if cb_key in cache:
                    cache.move_to_end(cb_key)
                    res = cache[cb_key]
                else:
                    call_kw = dict(all_args)
                    call_kw[arg_name] = key_arg_cb
                    res = fn(**call_kw)
                    entry_sz = get_byte_size(res)
                    _evict_if_needed(entry_sz)
                    cache[cb_key] = res
                    cache_bytes[0] += entry_sz
                return tree_util.tree_leaves(res)

            result_shape_list = [
                jax.ShapeDtypeStruct(l.shape, l.dtype) if hasattr(l, "shape") else l
                for l in result_leaves
            ]

            # Use custom_vjp to support differentiation
            @jax.custom_vjp
            def _cached_call(*flat_args):
                out_flat = jax.pure_callback(
                    _callback, result_shape_list, *flat_args, vmap_method="sequential"
                )
                return tuple(out_flat)

            def _cached_call_fwd(*flat_args):
                out = _cached_call(*flat_args)
                return out, flat_args

            def _cached_call_bwd(res, g):
                flat_args = res

                # Re-run the original function and differentiate through it
                def _differentiable(*fa):
                    key_arg_diff = tree_util.tree_unflatten(treedef, fa)
                    call_kw = dict(all_args)
                    call_kw[arg_name] = key_arg_diff
                    result = fn(**call_kw)
                    return tuple(tree_util.tree_leaves(result))

                _, vjp_fn = jax.vjp(_differentiable, *flat_args)
                return vjp_fn(g)

            _cached_call.defvjp(_cached_call_fwd, _cached_call_bwd)

            out_flat = _cached_call(*flat_key_arg)
            return tree_util.tree_unflatten(result_treedef, out_flat)

        # Expose cache for testing/debugging
        wrapper.cache = cache
        wrapper.clear_cache = lambda: (cache.clear(), cache_bytes.__setitem__(0, 0))

        return wrapper

    return decorator

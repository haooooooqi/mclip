from typing import Tuple, Any
import tree as nest

import jax
import jax.numpy as jnp

from optax._src import base
from optax._src.wrappers import MaskedState


# ---------------------------------------------------------
# exclude wd
# ---------------------------------------------------------
def filter_bias_and_norm(path: Tuple[Any], val: jnp.ndarray):
    """Filter to exclude biases and normalizations weights."""
    del val
    if path[-1] == "bias" or path[-1] == 'scale':
        return False
    return True


def filter_posembed(path: Tuple[Any], val: jnp.ndarray):
    """Filter to exclude pos emb."""
    del val
    name = '.'.join(path)
    if 'pos_embedding' in name:
        return False
    return True


# ---------------------------------------------------------
# freeze parameters
# ---------------------------------------------------------
def filter_by_keywords(path: Tuple[Any], val: jnp.ndarray, keywords: Tuple[Any]):
    """Filter given a list of keywords"""
    del val
    name = '.'.join(path)
    for kw in keywords:
        if kw in name:
            return False
    return True


# ---------------------------------------------------------
# the entrance function:
# ---------------------------------------------------------
def filter_parameters(params, filter_fn):
    """Filter the params based on filter_fn."""
    params_to_filter = nest.map_structure_with_path(filter_fn, params)
    return params_to_filter


# ------------------------------------------
# Masked optimizer
# ------------------------------------------
def masked(
    inner: base.GradientTransformation,
    mask: Union[base.PyTree, Callable[[base.Params], base.PyTree]]
) -> base.GradientTransformation:

  def mask_pytree(pytree, mask_tree):
    return jax.tree_map(lambda m, p: p if m else None, mask_tree, pytree)

  def init_fn(params):
    mask_tree = mask(params) if callable(mask) else mask
    masked_params = mask_pytree(params, mask_tree)
    return MaskedState(inner_state=inner.init(masked_params))

  def update_fn(updates, state, params=None):
    mask_tree = mask(updates) if callable(mask) else mask
    masked_updates = mask_pytree(updates, mask_tree)
    masked_params = None if params is None else mask_pytree(params, mask_tree)

    new_masked_updates, new_inner_state = inner.update(
        masked_updates, state.inner_state, masked_params)

    new_updates = jax.tree_map(
        lambda m, new_u: new_u if m else 0.,
        mask_tree, new_masked_updates)  # this takes zero
    return new_updates, MaskedState(inner_state=new_inner_state)

  return base.GradientTransformation(init_fn, update_fn)
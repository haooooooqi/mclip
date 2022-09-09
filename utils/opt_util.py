from typing import Tuple, Any, Union, Callable, NamedTuple
import tree

import jax
import jax.numpy as jnp

import chex
import flax

from optax._src import base
from optax._src import numerics
from optax._src.alias import ScalarOrSchedule
from optax._src.wrappers import MaskedState

freeze = flax.core.frozen_dict.freeze
unfreeze = flax.core.frozen_dict.unfreeze


# ---------------------------------------------------------
# exclude wd
# ---------------------------------------------------------
def filter_bias_and_norm(path: Tuple[Any], val: jnp.ndarray):
    """Filter to exclude biases and normalizations weights."""
    del val
    if path[-1] == "bias" or path[-1] == 'scale':
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
    # https://tree.readthedocs.io/en/latest/api.html
    params_to_filter = tree.map_structure_with_path(filter_fn, params)
    return params_to_filter


# ------------------------------------------
# Masked optimizer
# ------------------------------------------
def masked(
    inner: base.GradientTransformation,
    mask: Union[base.PyTree, Callable[[base.Params], base.PyTree]]
) -> base.GradientTransformation:

  def mask_pytree(pytree, mask_tree):
    # given a mask_tree, only returns parameters that are true
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
        mask_tree, new_masked_updates)
    return new_updates, MaskedState(inner_state=new_inner_state)

  return base.GradientTransformation(init_fn, update_fn)


# ------------------------------------------
# Masked optimizer + Momentum update
# ------------------------------------------
class MaskedMomentumState(NamedTuple):
  """Maintains inner transform state for masked transformations."""
  count: chex.Array
  inner_state: Any


def masked_with_momentum(
    inner: base.GradientTransformation,
    ema_momentum: ScalarOrSchedule,
    mask: Union[base.PyTree, Callable[[base.Params], base.PyTree]]
) -> base.GradientTransformation:

  def mask_pytree(pytree, mask_tree):
    # given a mask_tree, only returns parameters that are true
    return jax.tree_map(lambda m, p: p if m else None, mask_tree, pytree)

  def init_fn(params):
    mask_tree = mask(params) if callable(mask) else mask
    masked_params = mask_pytree(params, mask_tree)
    return MaskedMomentumState(count=jnp.zeros([], jnp.int32), inner_state=inner.init(masked_params))

  def update_fn(updates, state, params=None):
    mask_tree = mask(updates) if callable(mask) else mask
    masked_updates = mask_pytree(updates, mask_tree)
    masked_params = None if params is None else mask_pytree(params, mask_tree)

    new_masked_updates, new_inner_state = inner.update(
        masked_updates, state.inner_state, masked_params)

    new_updates = jax.tree_map(
        lambda m, new_u: new_u if m else 0.,
        mask_tree, new_masked_updates)

    if params is not None and 'Source' in params and 'Target' in params:
        mmt = ema_momentum(state.count) if callable(ema_momentum) else ema_momentum
        momentum_updates = momentum_delta(params['Source'], params['Target'], mmt)
        # hack, directly update target with source
        new_updates = unfreeze(new_updates)
        new_updates['Target'] = momentum_updates
        new_updates = freeze(new_updates)

    count_inc = numerics.safe_int32_increment(state.count)

    return new_updates, MaskedMomentumState(count=count_inc, inner_state=new_inner_state)

  return base.GradientTransformation(init_fn, update_fn)


# to output: mmt * t + (1 - mmt) * s = t + (1 - mmt) * (s - t)
# delta: (1 - mmt) * (s - t)
def momentum_delta(updates, params, mmt):
  """Compute the exponential moving average of the first moment."""
  return jax.tree_map(lambda x, y: (y - x) * (1 - mmt), params, updates)


# ------------------------------------------
# momentum schedule
# ------------------------------------------
def cosine_increase_schedule(
    init_value: float,
    steps: int,
) -> base.Schedule:
  """Returns a function which implements cosine rate increase."""
  if not steps > 0:
    raise ValueError('The cosine_decay_schedule requires positive steps!')

  def schedule(count):
    count = jnp.minimum(count, steps)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * count / steps))
    value = 1.0 - cosine_decay * (1.0 - init_value)
    return value

  return schedule

from typing import Any, Callable, NamedTuple, Optional, Union

import chex
import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import combine
from optax._src import numerics
from optax._src import utils

from optax._src import transform
from optax._src.transform import update_moment, bias_correction, ScaleByAdamState


# ------------------------------------------
# Momentum update as an optimizer
# ------------------------------------------
def momentum_update(
    momentum: float
) -> base.GradientTransformation:

  return _momentum_update(momentum)

  def init_fn(params):
    # do nothing
    return

  def update_fn(updates, state, params=None):
    del state
    delta = momentum_delta(updates, params, momentum)
    return delta, None

  return base.GradientTransformation(init_fn, update_fn)


# updates: s
# params: t
# to output: tau * t + (1 - tau) * s = t + (1 - tau) * (s - t)
# delta: (1 - tau) * (s - t)
def momentum_delta(updates, params, tau):
  """Compute the exponential moving average of the first moment."""
  return jax.tree_map(lambda x, y: (y - x) * (1 - tau), params, updates)

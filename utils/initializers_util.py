from typing import Any, Callable

from jax import random

import jax.numpy as jnp
from jax import dtypes

DType = Any

def constant(value, dtype: DType = jnp.float_):

  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.full(shape, value, dtype=dtype)

  return init


def patch_kernel(dtype: DType = jnp.float_):
  """
  ViT patch embedding initializer:
  As patch_embed is implemented as Conv, we view its 4D params as 2D
  """
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    h, w, c, n = shape
    fan_in = h * w * c
    fan_out = n
    denominator = (fan_in + fan_out) / 2
    variance = jnp.array(1. / denominator, dtype=dtype)
    return random.uniform(key, shape, dtype, -1) * jnp.sqrt(3 * variance)

  return init


def normal_l2(dtype: DType = jnp.float_):

  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    x = random.normal(key, shape, dtype)
    l2norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1.e-12)
    return x / l2norm

  return init

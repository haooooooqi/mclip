from typing import Any, Callable

import jax.numpy as jnp
from jax import dtypes

DType = Any
def constant(value, dtype: DType = jnp.float_) -> Callable:
  """Builds an initializer that returns arrays full of a constant ``value``.

  Args:
    value: the constant value with which to fill the initializer.
    dtype: optional; the initializer's default dtype.

  >>> import jax, jax.numpy as jnp
  >>> initializer = jax.nn.initializers.constant(-7)
  >>> initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)
  DeviceArray([[-7., -7., -7.],
               [-7., -7., -7.]], dtype=float32)
  """
  def init(key, shape, dtype=dtype):
    dtype = dtypes.canonicalize_dtype(dtype)
    return jnp.full(shape, value, dtype=dtype)
  return init
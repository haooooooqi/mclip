import jax.numpy as jnp

import numpy as np

# build the mask:
# mask: mask for the attention weights. This should be broadcastable to the
#   shape `[batch..., num_heads, q_length, kv_length]`.
#   This can be used for incorporating causal masks.
#   Attention weights are masked out if their corresponding mask value
#   is `False`.

def get_causal_mask(inputs):
  _, L, _ = inputs.shape
  mask = jnp.tril(jnp.ones(shape=(L, L), dtype=jnp.bool_)) # make a lower triangle
  mask = jnp.reshape(mask, (1, 1, L, L))
  return mask


def get_row_mask(inputs):
  """
  example: h=3, w=3, mask=
    [[1 0 0 1 0 0 1 0 0]
    [1 1 0 1 1 0 1 1 0]
    [1 1 1 1 1 1 1 1 1]
    [1 0 0 1 0 0 1 0 0]
    [1 1 0 1 1 0 1 1 0]
    [1 1 1 1 1 1 1 1 1]
    [1 0 0 1 0 0 1 0 0]
    [1 1 0 1 1 0 1 1 0]
    [1 1 1 1 1 1 1 1 1]]
  """
  _, L, _ = inputs.shape

  h = w = int(L**.5)
  assert h * w == L  # no cls token for now

  xs, ys = np.meshgrid(range(w), range(h))
  xys = np.concatenate([xs.reshape(-1, 1), ys.reshape(-1, 1)], axis=-1)
  mask = [[(k[0] <= q[0]) for k in xys] for q in xys]

  mask = jnp.array(mask, dtype=jnp.bool_)
  mask = jnp.reshape(mask, (1, 1, L, L))
  return mask

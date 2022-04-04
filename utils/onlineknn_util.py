import functools
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

import flax.linen as nn


class OnlineKNN(nn.Module):
  """Online kNN Monitor
  """
  knn: Any

  @nn.compact
  def __call__(self, x):
    from IPython import embed; embed();
    if (0 == 0): raise NotImplementedError
    # create the queue
    # K = self.knn.queue_size
    # queue = self.variable('knn_vars', 'queue',
    #                         lambda s: jnp.zeros(s, jnp.float32),
    #                         (K, x.shape[-1]))

    return
import functools
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

import flax.linen as nn


class OnlineKNN(nn.Module):
  """Online kNN Monitor during training.
  """
  knn: Any

  @nn.compact
  def __call__(self, features, labels, train):
    K = self.knn.queue_size
    D = features.shape[-1]

    # create the queue
    queue_features = self.variable('knn_vars', 'queue_features', lambda s: jnp.zeros(s, jnp.float32), (K, D))
    queue_labels = self.variable('knn_vars', 'queue_labels', lambda s: jnp.zeros(s, jnp.int32), (K,))
    queue_ptr = self.variable('knn_vars', 'queue_ptr', lambda s: jnp.zeros(s, jnp.int32), ())

    queue_features = queue_features.value
    queue_labels = queue_labels.value
    queue_ptr = queue_ptr.value

    if not train:  # we only monitor the training set.
      return

    # first, compute_knn_accuracy
    self.compute_knn_accuracy(features, labels, queue_features, queue_labels)

    return

  def compute_knn_accuracy(self, features, labels, queue_features, queue_labels):
    from IPython import embed; embed();
    if (0 == 0): raise NotImplementedError

    # [N, C] * [K, C] => [N, K]
    sim_matrix = jnp.einsum('nc,kc->nk', features, queue_features)

    # => [N, k] for top-k
    sim_weight, sim_indices = jax.lax.top_k(sim_matrix, k=self.knn.num_knns)

    # turn into scores: [N, k]
    sim_weight = tf.math.exp(sim_weight / flags.knn_t)

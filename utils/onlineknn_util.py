import functools
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import flax.linen as nn

import t5x.layers

from utils import initializers_util


class OnlineKNN(nn.Module):
  """Online kNN Monitor during training.
  """
  knn: Any

  @nn.compact
  def __call__(self, features, labels, train):
    N = self.knn.batch_size
    K = self.knn.queue_size // N
    E = features.shape[-1]

    # create the queue
    queue_features = t5x.layers.variable_with_axes(
        'knn_vars',
        'queue_features',
        initializers_util.normal_l2(),
        jax.random.PRNGKey(0),
        (K, N, E),
        jnp.float32,
        axes=('_null0', '_null1', '_null2'))

    queue_labels = t5x.layers.variable_with_axes(
        'knn_vars',
        'queue_labels',
        lambda s: jnp.zeros(s, jnp.int32),
        (K, N),
        axes=('_null0', '_null1'))

    queue_ptr = t5x.layers.variable_with_axes(
        'knn_vars',
        'queue_ptr',
        lambda s: jnp.zeros(s, jnp.int32),
        (1,),
        axes=('_null0',))

    if not train:  # we only monitor the training set.
      return None

    # compute knn accuracy
    knn_accuracy = self.compute_knn_accuracy(features, labels, queue_features, queue_labels, N, K)

    # update queue with the current batch
    self.update_queue(features, labels, queue_features, queue_labels, queue_ptr, K)

    return knn_accuracy

  def compute_knn_accuracy(self, features, labels, queue_features, queue_labels, N, K):
    # [B, E] * [K, N, E] => [B, K, N]
    sim_matrix = jnp.einsum('be,kne->bkn', features, queue_features.value)

    # [B, K, N] => [B, KxN]
    sim_matrix = jnp.reshape(sim_matrix, (N, self.knn.queue_size))

    # => [B, t] for top-k
    sim_weight, sim_indices = jax.lax.top_k(sim_matrix, k=self.knn.num_knns)

    # turn into scores: [N, t]
    sim_weight = jnp.exp(sim_weight / self.knn.temperature)

    # [B, t] => [B, t, KxN]
    sim_indices = jax.nn.one_hot(sim_indices, self.knn.queue_size)

    # [KxN] * [B, t, KxN] => [B, t]
    sim_labels = jnp.einsum('K,btK->bt',
                            jnp.reshape(queue_labels.value, (self.knn.queue_size,)),
                            sim_indices)

    # compute scores, [B, t, C]
    one_hot_labels = jax.nn.one_hot(sim_labels,
                                    self.knn.num_classes,
                                    dtype=sim_weight.dtype,
                                    axis=-1)

    # [B, t, C]
    pred_scores = one_hot_labels * jnp.expand_dims(sim_weight, -1)
    # [B, C]
    pred_scores = jnp.sum(pred_scores, axis=1)
    # [B,]
    pred_labels = jnp.argmax(pred_scores, axis=-1)

    accuracy = jnp.mean(pred_labels == labels)

    return accuracy

  def update_queue(self, features, labels, queue_features, queue_labels, queue_ptr, K):
    ptr = queue_ptr.value[0]

    queue_features.value = queue_features.value.at[ptr].set(features)
    queue_labels.value = queue_labels.value.at[ptr].set(labels)

    queue_ptr.value = queue_ptr.value.at[0].set((ptr + 1) % K)

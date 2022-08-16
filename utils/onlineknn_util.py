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
    K = self.knn.queue_size
    D = features.shape[-1]

    # create the queue
    queue_features = t5x.layers.variable_with_axes(
        'knn_vars',
        'queue_features',
        initializers_util.normal_l2,
        (K, D),
        jnp.float32,
        axes=('queue', 'feature'))

    queue_labels = t5x.layers.variable_with_axes(
        'knn_vars',
        'queue_labels',
        nn.initializers.zeros,
        (K,),
        jnp.int32,
        axes=('queue'))

    queue_ptr = t5x.layers.variable_with_axes(
        'knn_vars',
        'queue_ptr',
        nn.initializers.zeros,
        (),
        jnp.int32)

    if not train:  # we only monitor the training set.
      return None

    # compute knn accuracy
    knn_accuracy = self.compute_knn_accuracy(features, labels, queue_features, queue_labels)

    # update queue with the current batch
    self.update_queue(features, labels, queue_features, queue_labels, queue_ptr)

    return knn_accuracy

  def compute_knn_accuracy(self, features, labels, queue_features, queue_labels):
    # [N, E] * [K, E] => [N, K]
    sim_matrix = jnp.einsum('ne,ke->nk', features, queue_features.value)

    # => [N, t] for top-k
    sim_weight, sim_indices = jax.lax.top_k(sim_matrix, k=self.knn.num_knns)

    # turn into scores: [N, t]
    sim_weight = jnp.exp(sim_weight / self.knn.temperature)

    # [N, t] => [N, t, K]
    sim_indices = jax.nn.one_hot(sim_indices, queue_labels.value.shape[1])

    # [K] * [N, t, K] => [N, t]
    sim_labels = jnp.einsum('k,ntk->nt', queue_labels.value, sim_indices)

    # compute scores, [N, t, C]
    one_hot_labels = jax.nn.one_hot(sim_labels,
                                    self.knn.num_classes,
                                    dtype=sim_weight.dtype,
                                    axis=-1)

    # [N, t, C]
    pred_scores = one_hot_labels * jnp.expand_dims(sim_weight, -1)
    # [N, C]
    pred_scores = jnp.sum(pred_scores, axis=1)
    # [N,]
    pred_labels = jnp.argmax(pred_scores, axis=-1)

    accuracy = jnp.mean(pred_labels == labels)

    return accuracy

  def update_queue(self, features, labels, queue_features, queue_labels, queue_ptr):
    # assume it is from a single batch
    N = features.shape[0]

    assert self.knn.queue_size % N == 0
    inds = jnp.arange(N) + queue_ptr.value

    queue_features.value = queue_features.value.at[inds].set(features)
    queue_labels.value = queue_labels.value.at[inds].set(labels)

    queue_ptr.value = (queue_ptr.value + N) % self.knn.queue_size

import functools
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

import flax.linen as nn


# queue is not batched; ids is batched
gather = jax.vmap(lambda queue, ids: queue[ids], in_axes=(None, 0), out_axes=0)

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

    # compute knn accuracy
    knn_accuracy = self.compute_knn_accuracy(features, labels, queue_features, queue_labels)

    # update queue with the current batch
    self.update_queue(features, labels, queue_features, queue_labels, queue_ptr)

    return

  def compute_knn_accuracy(self, features, labels, queue_features, queue_labels):
    # [N, C] * [K, C] => [N, K]
    sim_matrix = jnp.einsum('nc,kc->nk', features, queue_features)

    # => [N, k] for top-k
    sim_weight, sim_indices = jax.lax.top_k(sim_matrix, k=self.knn.num_knns)

    # turn into scores: [N, k]
    sim_weight = jnp.exp(sim_weight / self.knn.temperature)

    # get labels: [N, k]
    sim_labels = gather(queue_labels, sim_indices)

    # compute scores
    one_hot_labels = jax.nn.one_hot(sim_labels, self.knn.num_classes, dtype=sim_weight.dtype, axis=-1)  # [N, k, CLS]
    pred_scores = one_hot_labels * jnp.expand_dims(sim_weight, -1)  # [N, k, CLS]
    pred_scores = jnp.sum(pred_scores, axis=1)  # [N, CLS]

    pred_labels = jnp.argmax(pred_scores, axis=-1)  # [N,]

    accuracy = jnp.mean(pred_labels == labels)
    return accuracy
  
  def update_queue(self, features, labels, queue_features, queue_labels, queue_ptr):
    features_all = jax.lax.all_gather(features, axis_name='batch')
    labels_all = jax.lax.all_gather(labels, axis_name='batch')

    features_all = jnp.reshape(features_all, [-1, features_all.shape[-1]])  # [MN, C]
    labels_all = jnp.reshape(labels_all, [-1])  # [MN,]

    from IPython import embed; embed();
    if (0 == 0): raise NotImplementedError

    batch_size = features_all.shape[0]
    new_queue_ptr = queue_ptr + batch_size
    queue_features.at[queue_ptr:new_queue_ptr].set(features_all)
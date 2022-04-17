import functools
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

import flax.linen as nn

from utils import dist_util


# queue is not batched; ids is batched
gather = jax.vmap(lambda queue, ids: queue[ids], in_axes=(None, 0), out_axes=0)


# initialize queue
def queue_features_init(shape):
  x = jax.random.normal(jax.random.PRNGKey(0), shape)
  l2norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1.e-12)
  x /= l2norm
  return x


class OnlineKNN(nn.Module):
  """Online kNN Monitor during training.
  """
  knn: Any

  @nn.compact
  def __call__(self, features, labels, train):
    K = self.knn.queue_size
    D = features.shape[-1]

    # create the queue
    queue_features = self.variable('knn_vars', 'queue_features', queue_features_init, (K, D))
    queue_labels = self.variable('knn_vars', 'queue_labels', lambda s: jnp.zeros(s, jnp.int32), (K,))
    queue_ptr = self.variable('knn_vars', 'queue_ptr', lambda s: jnp.zeros(s, jnp.int32), ())

    if not train:  # we only monitor the training set.
      return None

    # compute knn accuracy
    num_samples = int(self.knn.eval_sampling * features.shape[0])
    knn_accuracy = self.compute_knn_accuracy(features[:num_samples], labels[:num_samples], queue_features, queue_labels)

    # update queue with the current batch
    self.update_queue(features, labels, queue_features, queue_labels, queue_ptr)

    return knn_accuracy

  def compute_knn_accuracy(self, features, labels, queue_features, queue_labels):
    # [N, C] * [K, C] => [N, K]
    sim_matrix = jnp.einsum('nc,kc->nk', features, queue_features.value)

    # => [N, k] for top-k
    sim_weight, sim_indices = jax.lax.top_k(sim_matrix, k=self.knn.num_knns)

    # turn into scores: [N, k]
    sim_weight = jnp.exp(sim_weight / self.knn.temperature)

    # get labels: [N, k]
    sim_labels = gather(queue_labels.value, sim_indices)

    # compute scores
    one_hot_labels = jax.nn.one_hot(sim_labels, self.knn.num_classes, dtype=sim_weight.dtype, axis=-1)  # [N, k, CLS]
    pred_scores = one_hot_labels * jnp.expand_dims(sim_weight, -1)  # [N, k, CLS]
    pred_scores = jnp.sum(pred_scores, axis=1)  # [N, CLS]

    pred_labels = jnp.argmax(pred_scores, axis=-1)  # [N,]

    accuracy = jnp.mean(pred_labels == labels)
    return accuracy
  
  def update_queue(self, features, labels, queue_features, queue_labels, queue_ptr):
    features_all = dist_util.all_gather(features, axis_name='batch')
    labels_all = dist_util.all_gather(labels, axis_name='batch')

    # features_all = jax.lax.all_gather(features, axis_name='batch')
    # labels_all = jax.lax.all_gather(labels, axis_name='batch')

    features_all = jnp.reshape(features_all, [-1, features_all.shape[-1]])  # [MN, C]
    labels_all = jnp.reshape(labels_all, [-1,])  # [MN,]

    batch_size = features_all.shape[0]
    assert self.knn.queue_size % batch_size == 0
    inds = jnp.arange(batch_size) + queue_ptr.value

    queue_features.value = queue_features.value.at[inds].set(features_all)
    queue_labels.value = queue_labels.value.at[inds].set(labels_all)
    queue_ptr.value = (queue_ptr.value + batch_size) % self.knn.queue_size
    return 
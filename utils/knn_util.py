
from absl import logging

import time, datetime

import jax
import jax.numpy as jnp
from jax import lax


def gather(x, ids):
  return x[ids]
vmapped_gather = jax.jit(jax.vmap(gather, in_axes=(0, 0), out_axes=0))


def apply_knn(state, p_encode_step, eval_iter, knn_train_iter, dataset_builder, config):
  tic = time.time()

  batch_size = config.batch_size

  steps_per_eval = dataset_builder.info.splits['validation'].num_examples // batch_size
  steps_per_train = dataset_builder.info.splits['train'].num_examples // batch_size

  # extract val features
  val_features = []
  val_labels = []
  for i in range(steps_per_eval):
    eval_batch = next(eval_iter)
    features, labels = p_encode_step(state, eval_batch)  # all-gathered
    features = features[0]  # [batch_size, dim,]
    labels = labels[0]  # [batch_size,]

    if i == 0:
        logging.info('features.shape: {}'.format(features.shape))

    val_features.append(features)
    val_labels.append(labels)

  val_features = jnp.concatenate(val_features, axis=0)
  val_labels = jnp.concatenate(val_labels, axis=0)
  logging.info('val_features.shape: {}'.format(val_features.shape))


#   for _ in range(steps_per_eval):
  train_batch = next(knn_train_iter)

  from IPython import embed; embed();
  if (0 == 0): raise NotImplementedError

  train_features, train_labels = p_encode_step(state, train_batch)  # all-gathered
  train_features = train_features[0]  # [batch_size, dim,]
  train_labels = train_labels[0]  # [batch_size, dim,]


  CLS = dataset_builder.info.features['label'].num_classes

  N = val_features.shape[0]  # N: # val samples (queries)

  # cached the top-k neighbors for each query: [N, k]  
  k_knns = config.knn.num_knns
  sim_matrix_cached = -jnp.ones((N, k_knns), dtype=val_features.dtype)  # [N, k]
  sim_labels_cached = jnp.zeros((N, k_knns), dtype=train_labels.dtype)  # [N, k]


  # [N, C] * [K, C] => [N, K], K: # train samples in this batch (keys)
  sim_matrix = jnp.einsum('nc,kc->nk', val_features, train_features)
  sim_matrix = jnp.concatenate([sim_matrix, sim_matrix_cached], axis=1)  # [N, K + k]
  
  ex_train_labels = jnp.expand_dims(train_labels, axis=0)
  ex_train_labels = jnp.repeat(ex_train_labels, N, axis=0)
  sim_labels = jnp.concatenate([ex_train_labels, sim_labels_cached], axis=1)  # [N, K + k]

  # => [N, k] for top-k
  sim_weight, sim_indices = jax.lax.top_k(sim_matrix, k=k_knns)

  # update the cache
  sim_matrix_cached = sim_weight  # update the cache

  # update the labels
  sim_labels = vmapped_gather(sim_labels, sim_indices)  # [N, k]
  sim_labels_cached = sim_labels


  toc = time.time() - tic
  logging.info('kNN time: {}'.format(str(datetime.timedelta(seconds=int(toc)))))

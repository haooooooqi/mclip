
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
  logging.info('Start kNN.')

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


  # cached the top-k neighbors for each query: [N, k]  
  N = val_features.shape[0]  # N: # val samples (queries)
  k_knns = config.knn.num_knns
  sim_matrix_cached = -jnp.ones((N, k_knns), dtype=val_features.dtype)  # [N, k]
  sim_labels_cached = jnp.zeros((N, k_knns), dtype=val_labels.dtype)  # [N, k]

  steps_per_train = 100
  for i in range(steps_per_train):
    train_batch = next(knn_train_iter)

    train_features, train_labels = p_encode_step(state, train_batch)  # all-gathered
    train_features = train_features[0]  # [batch_size, dim,]
    train_labels = train_labels[0]  # [batch_size, dim,]

    # Wait until computations are done before exiting
    # jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    sim_matrix_cached, sim_labels_cached = update_knn(
        val_features, train_features, train_labels, sim_matrix_cached, sim_labels_cached)
    if (i % 10 == 0):
        logging.info('Updating train kNN: {} steps.'.format(i))

  toc = time.time() - tic
  logging.info('kNN time: {}'.format(str(datetime.timedelta(seconds=int(toc)))))
  from IPython import embed; embed();
  if (0 == 0): raise NotImplementedError

  # finalize kNN metrics
  CLS = dataset_builder.info.features['label'].num_classes


  # To reduce memory, we process it by batching
  split_batch_size = 256
  sim_matrix_cached = jnp.reshape(sim_matrix_cached, [N // split_batch_size, split_batch_size, k_knns])
  sim_labels_cached = jnp.reshape(sim_labels_cached, [N // split_batch_size, split_batch_size, k_knns])
  val_labels = jnp.reshape(val_labels, [N // split_batch_size, split_batch_size,])
  
  total_accuracy = 0.
  for i in range(sim_matrix_cached.shape[0]):
    sim_weight = sim_matrix_cached[i]  # [B, k]
    sim_labels = sim_labels_cached[i]  # [B, k]
    gt_labels = val_labels[i]  # [B,]

    sim_weight = jnp.exp(sim_weight / config.knn.temperature)  # [B, k]
    one_hot_labels = jax.nn.one_hot(sim_labels, CLS, dtype=sim_weight.dtype, axis=-1)  # [B, k, CLS]

    pred_scores = one_hot_labels * jnp.expand_dims(sim_weight, -1)  # [B, k, CLS]
    pred_scores = jnp.sum(pred_scores, axis=1)  # [B, CLS]

    pred_labels = jnp.argmax(pred_scores, axis=-1)
    accuracy = jnp.mean(pred_labels == gt_labels)
    total_accuracy += accuracy

@jax.jit
def update_knn(val_features, train_features, train_labels, sim_matrix_cached, sim_labels_cached):
  k_knns = sim_matrix_cached.shape[1]

  # [N, C] * [K, C] => [N, K], K: # train samples in this batch (keys)
  sim_matrix = jnp.einsum('nc,kc->nk', val_features, train_features)
  sim_matrix = jnp.concatenate([sim_matrix, sim_matrix_cached], axis=1)  # [N, K + k]
  
  ex_train_labels = jnp.expand_dims(train_labels, axis=0)
  ex_train_labels = jnp.repeat(ex_train_labels, val_features.shape[0], axis=0)
  sim_labels = jnp.concatenate([ex_train_labels, sim_labels_cached], axis=1)  # [N, K + k]

  # => [N, k] for top-k
  sim_weight, sim_indices = jax.lax.top_k(sim_matrix, k=k_knns)

  # update the cache
  sim_matrix_cached = sim_weight  # update the cache

  # update the labels
  sim_labels = vmapped_gather(sim_labels, sim_indices)  # [N, k]
  sim_labels_cached = sim_labels

  return sim_matrix_cached, sim_labels_cached

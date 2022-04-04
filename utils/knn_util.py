
from absl import logging

import time, datetime

import jax
import jax.numpy as jnp
from jax import lax


def apply_knn(state, p_encode_step, eval_iter, dataset_builder, config):
  tic = time.time()

  batch_size = config.batch_size

  steps_per_eval = dataset_builder.info.splits['validation'].num_examples // batch_size
  steps_per_epoch = dataset_builder.info.splits['train'].num_examples // batch_size

  from IPython import embed; embed();
  if (0 == 0): raise NotImplementedError
  # extract val features
  val_features = []
  val_labels = []
  for _ in range(steps_per_eval):
    eval_batch = next(eval_iter)
    features, labels = p_encode_step(state, eval_batch)  # all-gathered
    features = features[0, :, :, :]  # [hosts, batch_per_device, dim,]
    labels = labels[0, :, :]  # [hosts, batch_per_device,]
    features = jnp.reshape(features, (-1, features.shape[-1]))
    labels = jnp.reshape(labels, (-1,))

    val_features.append(features)
    val_labels.append(labels)

  val_features = jnp.concatenate(val_features, axis=0)
  val_labels = jnp.concatenate(val_labels, axis=0)
  logging.info('val_features.shape: {}'.format(val_features.shape))
    
  toc = time.time() - tic
  logging.info('kNN time: {}'.format(str(datetime.timedelta(seconds=int(toc)))))

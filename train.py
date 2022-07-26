# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ImageNet example.

This script trains a ViT on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import functools
import time, datetime
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax import optim
from flax import struct
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import input_pipeline
import models_mae

from utils import summary_util as summary_util  # must be after 'from clu import metric_writers'
from utils import opt_util
# from utils import mix_util
from utils import adamw_util
from utils.transform_util import MEAN_RGB, STDDEV_RGB
from utils import torchloader_util

import torch

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

import jax.profiler

import numpy as np
import os
import random as _random


def initialized(key, image_size, model, init_backend='tpu'):
  init_batch_size = 16
  input_shape = (init_batch_size, 3, image_size, image_size, 3)
  # TODO{kaiming}: load a real batch
  init_batch = {'image': jax.random.normal(jax.random.PRNGKey(0), input_shape, dtype=model.dtype),
    'label': jnp.zeros((init_batch_size,), jnp.int32)}

  def init(*args):
    return model.init(*args, train=False)
  init = jax.jit(init, backend=init_backend)
  logging.info('Initializing params...')
  variables = init(
    {'params': key, 'dropout': random.PRNGKey(0)},  # kaiming: random masking needs the 'dropout' key
    init_batch)
  logging.info('Initializing params done.')
  return variables


def cross_entropy_loss(logits, labels_one_hot):
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels, labels_one_hot):
  loss = cross_entropy_loss(logits, labels_one_hot)
  accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  metrics = lax.pmean(metrics, axis_name='batch')
  return metrics


def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=0., end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch)
  cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  cosine_fn = optax.cosine_decay_schedule(
      init_value=base_learning_rate,
      decay_steps=cosine_epochs * steps_per_epoch,
      alpha=config.min_abs_lr / base_learning_rate)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  return schedule_fn


def train_step(state, batch, learning_rate_fn, config):
  """Perform a single training step."""
  _, new_rng = jax.random.split(state.rng)
  # Bind the rng key to the device id (which is unique across hosts)
  # Note: This is only used for multi-host training (i.e. multiple computers
  # each with multiple accelerators).
  dropout_rng = jax.random.fold_in(state.rng, jax.lax.axis_index('batch'))
  def loss_fn(params):
    """loss function used for training."""
    mutable = [k for k in state.variables]
    outcome = state.apply_fn(
        {'params': params, **state.variables},
        inputs=batch,
        mutable=mutable,
        rngs=dict(dropout=dropout_rng),
        train=True)
    (loss, pred, knn_accuracy, artifacts), new_variables = outcome
    return loss, (new_variables, loss, knn_accuracy, artifacts)

  step = state.step
  lr = learning_rate_fn(step)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = lax.pmean(grads, axis_name='batch')

  new_variables, loss, knn_accuracy, artifacts = aux[1]

  metrics = {'loss': loss, 'learning_rate': lr, 'knn_accuracy': knn_accuracy}
  metrics = {**metrics, **artifacts}
  metrics = lax.pmean(metrics, axis_name='batch')

  # ----------------------------------------------------------------------------
  # original
  # new_state = state.apply_gradients(grads=grads, variables=new_variables, rng=new_rng)
  # if new_state.ema is not None:
  #   new_ema = new_state.ema.update(flax.core.FrozenDict({'params': new_state.params, **new_variables}))
  #   new_state = new_state.replace(ema=new_ema)
  # ----------------------------------------------------------------------------

  # ----------------------------------------------------------------------------
  # modified impl.
  updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
  new_params = optax.apply_updates(state.params, updates)

  if config.ema:
    _, new_ema_state = state.ema_tx.update(
      updates=flax.core.frozen_dict.FrozenDict({'params': new_params, **new_variables}),
      state=state.ema_state)
  else:
    new_ema_state = None
  
  # new_ema = state.ema.update(flax.core.FrozenDict({'params': new_params, **new_variables})) if state.ema is not None else None
  new_state = state.replace(
    step=state.step + 1,
    params=new_params,
    opt_state=new_opt_state,
    variables=new_variables,
    rng=new_rng,
    ema_state=new_ema_state
  )
  # ----------------------------------------------------------------------------

  return new_state, metrics


def eval_step(state, batch):
  variables = {'params': state.params, **state.variables}

  dropout_rng = jax.random.fold_in(state.rng, jax.lax.axis_index('batch'))  # kaiming: eval rng should not matter?
  outcome = state.apply_fn(variables, batch, train=False, mutable=False, rngs=dict(dropout=dropout_rng))
  loss, imgs_vis, _, _ = outcome

  metrics = {'test_loss': loss}
  metrics = lax.pmean(metrics, axis_name='batch')
  metrics['imgs_vis'] = imgs_vis

  return metrics


def prepare_pt_data(xs):
  """Convert a input batch from PyTorch Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x.numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


class TrainState(train_state.TrainState):
  rng: Any
  variables: flax.core.FrozenDict[str, Any]
  ema_tx: optax.GradientTransformation = struct.field(pytree_node=False)
  ema_state: optax.EmaState


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=10)


def profile_memory(workdir):
  jax.profiler.save_device_memory_profile("/tmp/memory.prof")
  if jax.process_index() == 0:
    os.system('gsutil cp /tmp/memory.prof {}'.format(workdir))


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  if 'batch_stats' not in state.variables:
    return state
  else:
    new_variables, batch_stats = state.variables.pop('batch_stats')
    batch_stats = cross_replica_mean(batch_stats)
    return state.replace(variables=flax.core.FrozenDict({'batch_stats': batch_stats, **new_variables}))


def create_train_state(rng, config: ml_collections.ConfigDict,
                       model, image_size, learning_rate_fn):
  """Create initial training state."""

  # split rng for init and for state
  rng_init, rng_state = jax.random.split(rng)

  variables = initialized(rng_init, image_size, model, config.init_backend)
  variables_states, params = variables.pop('params')

  # optional: rescale
  if config.rescale_init:
    rescales = opt_util.filter_parameters(params, opt_util.layer_rescale)
    params = jax.tree_util.tree_multimap(lambda x, y: x * y, rescales, params)

  # stds = jax.tree_util.tree_map(lambda x: (x.shape, np.array(x).std()), params)
  # logging.info('std: {}'.format(stds))

  # optional: exclude some wd
  if config.exclude_wd:
    mask = jax.tree_util.tree_multimap(lambda x, y: bool(x and y), 
      opt_util.filter_parameters(params, opt_util.filter_bias_and_norm),
      opt_util.filter_parameters(params, opt_util.filter_posembed)  # Note: we must exclude posembed wd in adamw
    )
  else:
    mask = None
  # logging.info('Apply weight decay: {}'.format(mask))

  # tx = getattr(optax, config.opt_type)  # optax.adamw
  tx = getattr(adamw_util, config.opt_type)  # optax.adamw
  tx = tx(learning_rate=learning_rate_fn, **config.opt, mask=mask, mu_dtype=getattr(jnp, config.opt_mu_dtype))
  tx = optax.GradientTransformation(init=jax.jit(tx.init, backend=config.init_backend), update=tx.update)  # put to cpu
  if config.ema:
    ema_tx = optax.ema(decay=config.ema_decay, debias=False)
    ema_state = ema_tx.init(flax.core.frozen_dict.FrozenDict({'params': params, **variables_states}))
  else:
    ema_tx = None
    ema_state = None
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      rng=rng_state,
      variables=variables_states,
      ema_tx=ema_tx,
      ema_state=ema_state)
  return state


def seed_worker(worker_id, global_seed, offset_seed=0):
    # worker_seed = torch.initial_seed() % 2**32 + jax.process_index() + offset_seed
    worker_seed = (global_seed + worker_id + jax.process_index() + offset_seed) % 2**32
    np.random.seed(worker_seed)
    _random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    logging.info('worker_id: {}, worker_seed: {}; offset_seed {}'.format(worker_id, worker_seed, offset_seed))


def set_seed_torch(seed):
  rng_torch = torch.Generator()
  rng_torch.manual_seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  _random.seed(seed)
  return rng_torch


def rebuild_data_loader_train(dataset_train, sampler_train, local_batch_size, config, offset_seed):
  rng_torch = torch.Generator()
  rng_torch.manual_seed(offset_seed)
  data_loader_train = torch.utils.data.DataLoader(
    dataset_train, sampler=sampler_train,
    batch_size=local_batch_size,
    num_workers=config.torchload.num_workers,
    pin_memory=True,
    drop_last=True,
    generator=rng_torch,
    worker_init_fn=functools.partial(seed_worker, offset_seed=offset_seed, global_seed=config.seed_pt),
    persistent_workers=True,
    timeout=60.,
  )
  return data_loader_train


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
  # ------------------------------------
  # Set random seed
  # ------------------------------------
  rng_torch = set_seed_torch(config.seed_pt)
  tf.random.set_seed(config.seed_tf + jax.process_index())
  rng = random.PRNGKey(config.seed_jax)  # used to be 0
  # ------------------------------------

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)

  rng = random.PRNGKey(config.seed_jax)  # used to be 0

  image_size = 224

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()

  dataset_val = torchloader_util.build_dataset(is_train=False, data_dir=config.torchload.data_dir, aug=config.aug)
  dataset_train = torchloader_util.build_dataset(is_train=True, data_dir=config.torchload.data_dir, aug=config.aug)

  sampler_train = torch.utils.data.DistributedSampler(
    dataset_train,
    num_replicas=jax.process_count(),
    rank=jax.process_index(),
    shuffle=True,
    seed=config.seed_pt,
  )
  sampler_val = torch.utils.data.DistributedSampler(
    dataset_val,
    num_replicas=jax.process_count(),
    rank=jax.process_index(),
    shuffle=True,  # shuffle for visualization
  )

  # steps_per_epoch = len(data_loader_train)
  # assert steps_per_epoch == len(dataset_train) // config.batch_size
  steps_per_epoch = len(dataset_train) // config.batch_size

  abs_learning_rate = config.learning_rate * config.batch_size / 256.

  model = models_mae.VisionTransformer(num_classes=-1, **config.model)  # num_classes not used

  learning_rate_fn = create_learning_rate_fn(config, abs_learning_rate, steps_per_epoch)

  state = create_train_state(rng, config, model, image_size, learning_rate_fn)
  state = restore_checkpoint(state, workdir if config.resume_dir == '' else config.resume_dir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)

  data_loader_train = rebuild_data_loader_train(dataset_train, sampler_train, local_batch_size, config, offset_seed=step_offset)
  # data_loader_train = torch.utils.data.DataLoader(
  #   dataset_train, sampler=sampler_train,
  #   batch_size=local_batch_size,
  #   num_workers=config.torchload.num_workers,
  #   pin_memory=True,
  #   drop_last=True,
  #   generator=rng_torch,
  #   worker_init_fn=functools.partial(seed_worker, offset_seed=step_offset),
  #   persistent_workers=True,
  #   timeout=60.,
  # )
  data_loader_val = torch.utils.data.DataLoader(
    dataset_val, sampler=sampler_val,
    batch_size=local_batch_size,
    # num_workers=config.torchload.num_workers,
    pin_memory=True,
    drop_last=True,
    # persistent_workers=True,
    # timeout=60.,
  )


  # --------------------------------------------------------------------------------
  # debug
  batch = next(iter(data_loader_val))
  # --------------------------------------------------------------------------------

  # --------------------------------------------------------------------------------
  # up til now, state.params are for one device
  # image = jnp.ones([3, 224, 224, 3])
  # label = jnp.ones([3,], dtype=jnp.int32)
  # mutable_keys = [k for k in state.variables]
  # outcome = state.apply_fn(
  #     {'params': state.params,
  #      **state.variables,
  #     },
  #     rngs=dict(dropout=state.rng),
  #     inputs={'image': image, 'label': label},
  #     mutable=mutable_keys,
  #     train=True)
  # (loss, pred), new_variables = outcome
  # num_params = np.sum([np.prod(p.shape) for p in jax.tree_leaves(state.opt_state[0].nu)])
  # num_params = np.sum([np.prod(p.shape) for p in jax.tree_leaves(state.params)])
  # num_params_mem = num_params * 4 / 1024 / 1024
  # --------------------------------------------------------------------------------  
  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn, config=config),
      axis_name='batch',
      donate_argnums=(0,) if config.donate else ()
      )
  p_eval_step = jax.pmap(
      eval_step,
      axis_name='batch')

  train_metrics = []
  hooks = []
  # if jax.process_index() == 0:
  #   hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')

  epoch_offset = (step_offset + 1) // steps_per_epoch
  step = epoch_offset * steps_per_epoch
  assert step == int(state.step[0])  # sanity when loading

  for epoch in range(epoch_offset, int(config.num_epochs)):
    data_loader_train.sampler.set_epoch(epoch)  # reset random seed

    # ------------------------------------------------------------
    # train one epoch
    # ------------------------------------------------------------
    for i, batch in enumerate(data_loader_train):
      batch = parse_batch(batch)
      state, metrics = p_train_step(state, batch)

      epoch_1000x = int(step * config.batch_size / 1281167 * 1000)  # normalize to IN1K epoch anyway

      if epoch == epoch_offset and i == 0:
        logging.info('Initial compilation completed.')
        start_time = time.time()  # log the time after compilation

      if config.get('log_every_steps'):
        train_metrics.append(metrics)
        if (step + 1) % config.log_every_steps == 0:        
          # Wait until computations are done before exiting
          jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
          train_metrics = common_utils.get_metrics(train_metrics)
          summary = {
              f'train_{k}': float(v)
              for k, v in jax.tree_map(lambda x: x.mean(), train_metrics).items()
          }
          summary['steps_per_second'] = config.log_every_steps / (
              time.time() - train_metrics_last_t)

          # to make it consistent with PyTorch log
          summary['loss'] = summary['train_loss']  # add extra name
          summary['lr'] = summary.pop('train_learning_rate')  # rename
          summary['knn_accuracy'] = summary.pop('train_knn_accuracy')  # rename
          # summary['class_acc'] = summary.pop('train_accuracy')  # this is [0, 1]
          summary['step_tensorboard'] = epoch_1000x  # step for tensorboard

          writer.write_scalars(step + 1, summary)
          train_metrics = []
          train_metrics_last_t = time.time()

      step += 1

    # ------------------------------------------------------------
    # finished one epoch: eval
    # ------------------------------------------------------------
    if ((epoch + 1) % config.vis_every_epochs == 0 or epoch == epoch_offset) and config.model.visualize:
      data_loader_val.sampler.set_epoch(epoch)
      eval_batch = next(iter(data_loader_val))
      eval_batch = parse_batch(eval_batch)
      metrics = p_eval_step(state, eval_batch)

      imgs_vis = metrics.pop('imgs_vis')[0]  # keep the master device
      imgs_vis = imgs_vis * jnp.asarray(STDDEV_RGB) + jnp.asarray(MEAN_RGB)
      imgs_vis = jnp.uint8(jnp.clip(imgs_vis, 0, 255.))
      writer.write_images(step=epoch_1000x, images=dict(imgs_vis=imgs_vis))

      summary = jax.tree_map(lambda x: x.mean(), metrics)
      values = [f"{k}: {v:.6f}" for k, v in sorted(summary.items())]
      logging.info('eval epoch: %d, %s', epoch, ', '.join(values))

    # ------------------------------------------------------------
    # finished one epoch: save
    # ------------------------------------------------------------
    if (epoch + 1) % config.save_every_epochs == 0 or epoch + 1 == int(config.num_epochs):
      state = sync_batch_stats(state)
      save_checkpoint(state, workdir)

      # rebuild the data loader for reproducibility (TODO: verify)
      data_loader_train = rebuild_data_loader_train(dataset_train, sampler_train, local_batch_size, config, offset_seed=step)
      assert step == int(state.step[0])

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=int(total_time)))
  logging.info('Elapsed time: {}'.format(total_time_str))

  if config.profile_memory:
    profile_memory(workdir)

  return state


def parse_batch(batch):
  images, labels = batch
  # images = images.permute([0, 2, 3, 1])  # nchw -> nhwc
  images = images.permute([0, 1, 3, 4, 2])  # nvchw -> nvhwc
  batch = {'image': images, 'label': labels}
  batch = prepare_pt_data(batch)  # to (local_devices, device_batch_size, height, width, 3)
  return batch

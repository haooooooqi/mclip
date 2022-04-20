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
# from flax.training import train_state
import utils.train_state as train_state
import utils.adamw_util as adamw_util
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import input_pipeline
import models_vit

from utils import summary_util as summary_util  # must be after 'from clu import metric_writers'
from utils import opt_util
from utils import mix_util
from utils import checkpoint_util
from utils import lrd_util
from utils import torchvision_util
from utils import torchloader_util

import jax.profiler

import numpy as np
import os
import math

import torch
import torch.utils.data
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

NUM_CLASSES = 1000


def create_model(*, model_cls, half_precision, **kwargs):
  assert not half_precision
  # platform = jax.local_devices()[0].platform
  # if half_precision:
  #   if platform == 'tpu':
  #     model_dtype = jnp.bfloat16
  #   else:
  #     model_dtype = jnp.float16
  # else:
  #   model_dtype = jnp.float32
  return model_cls(num_classes=NUM_CLASSES, **kwargs)


def initialized(key, image_size, model, init_backend='tpu'):
  input_shape = (1, image_size, image_size, 3)
  def init(*args):
    return model.init(*args, train=False)
  init = jax.jit(init, backend=init_backend)
  logging.info('Initializing params...')
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
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


def compute_eval_metrics(logits, labels, labels_one_hot):
  """kaiming: we do not average here (to support the reminder batch)
  """
  loss = optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot)
  accuracy = (jnp.argmax(logits, -1) == labels)
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'label': labels
  }
  metrics = lax.all_gather(metrics, axis_name='batch')
  metrics = jax.tree_map(lambda x: jnp.reshape(x, [-1,]), metrics)
  return metrics


def create_learning_rate_fn(
    config: ml_collections.ConfigDict,
    base_learning_rate: float,
    steps_per_epoch: int):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=config.warmup_abs_lr, end_value=base_learning_rate,
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
        inputs=batch['image'],
        mutable=mutable,
        rngs=dict(dropout=dropout_rng),
        train=True)
    logits, new_variables = outcome

    loss = cross_entropy_loss(logits, batch['label_one_hot'])
    return loss, (new_variables, logits)

  step = state.step
  dynamic_scale = state.dynamic_scale
  lr = learning_rate_fn(step)

  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
        loss_fn, has_aux=True, axis_name='batch')
    dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')
  new_variables, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'], batch['label_one_hot'])
  metrics['learning_rate'] = lr

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

  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
    new_state = new_state.replace(
        opt_state=jax.tree_map(
            functools.partial(jnp.where, is_fin),
            new_state.opt_state,
            state.opt_state),
        params=jax.tree_map(
            functools.partial(jnp.where, is_fin),
            new_state.params,
            state.params))
    metrics['scale'] = dynamic_scale.scale

  return new_state, metrics


def eval_step(state, batch, ema_eval=False):
  variables = {'params': state.params, **state.variables}
  logits = state.apply_fn(variables, batch['image'], train=False, mutable=False)
  metrics = compute_eval_metrics(logits, batch['label'], batch['label_one_hot'])
  metrics['test_acc1'] = metrics.pop('accuracy') * 100  # rename
  metrics['perf/test_acc1'] = metrics['test_acc1']  # for comparing with pytorch
  metrics['test_loss'] = metrics.pop('loss')  # rename

  # if ema_eval:
  #   logits = state.apply_fn(state.ema_state.ema, batch['image'], train=False, mutable=False)
  #   metrics_ema = compute_metrics(logits, batch['label'], batch['label_one_hot'])
  #   metrics['test_acc1_ema'] = metrics_ema.pop('accuracy') * 100  # rename

  return metrics


def prepare_tf_data(xs, batch_size):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    if x.shape[0] != batch_size:
      pads = -np.ones((batch_size - x.shape[0],) + x.shape[1:], dtype=x.dtype)
      x = np.concatenate([x, pads], axis=0)

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def prepare_pt_data(xs, batch_size):
  """Convert a input batch from PyTorch Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x.numpy()  # pylint: disable=protected-access

    if x.shape[0] != batch_size:
      pads = -np.ones((batch_size - x.shape[0],) + x.shape[1:], dtype=x.dtype)
      x = np.concatenate([x, pads], axis=0)

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
                      cache, aug=None):
  ds = input_pipeline.create_split(
      dataset_builder, batch_size, image_size=image_size, dtype=dtype,
      train=train, cache=cache, aug=aug)

  if aug and (aug.mix.mixup or aug.mix.cutmix) and (not aug.mix.torchvision):
    apply_mix = functools.partial(mix_util.apply_mix, cfg=aug.mix)
    ds = map(apply_mix, ds)
  elif aug and (aug.mix.mixup or aug.mix.cutmix) and (aug.mix.torchvision):
    num_classes = dataset_builder.info.features['label'].num_classes
    ds = map(torchvision_util.get_torchvision_map_mix_fn(aug, num_classes), ds)

  # ------------------------------------------------
  # x = next(iter(ds))
  # raise NotImplementedError
  # ------------------------------------------------

  ds = map(functools.partial(prepare_tf_data, batch_size=batch_size), ds)
  it = jax_utils.prefetch_to_device(ds, 2)
  return it


class TrainState(train_state.TrainState):
  rng: Any
  variables: flax.core.FrozenDict[str, Any]
  # dynamic_scale: flax.optim.DynamicScale
  dynamic_scale: Any
  ema_tx: optax.GradientTransformation = struct.field(pytree_node=False)
  ema_state: optax.EmaState


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


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
  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if config.half_precision and platform == 'gpu':
    dynamic_scale = optim.DynamicScale()
  else:
    dynamic_scale = None

  # split rng for init and for state
  rng_init, rng_state = jax.random.split(rng)

  variables = initialized(rng_init, image_size, model, config.init_backend)
  variables_states, params = variables.pop('params')

  # optional: rescale
  if config.rescale_init:
    rescales = opt_util.filter_parameters(params, opt_util.layer_rescale)
    params = jax.tree_util.tree_map(lambda x, y: x * y, rescales, params)

  if config.rescale_head_init != 1.:
    params = flax.core.frozen_dict.unfreeze(params)
    params['head']['kernel'] *= config.rescale_head_init
    params = flax.core.frozen_dict.freeze(params)

  # stds = jax.tree_util.tree_map(lambda x: np.array(x).std(), params)
  # logging.info('std: {}'.format(stds))

  # optional: exclude some wd
  if config.exclude_wd:
    mask = jax.tree_util.tree_map(lambda x, y: bool(x and y), 
      opt_util.filter_parameters(params, opt_util.filter_bias_and_norm),
      opt_util.filter_parameters(params, opt_util.filter_cls_and_posembed)
    )
  else:
    mask = None
  # logging.info('Apply weight decay: {}'.format(mask))

  # tx = getattr(optax, config.opt_type)  # optax.adamw
  tx = getattr(adamw_util, config.opt_type)  # optax.adamw
  tx = tx(learning_rate=learning_rate_fn, **config.opt, mask=mask, mu_dtype=getattr(jnp, config.opt_mu_dtype))

  if config.learning_rate_decay < 1.:
    lrd_func = lrd_util.lrd_func(config.model.transformer.num_layers, config.learning_rate_decay)
    lrd = lrd_util.filter_parameters(params, lrd_func)
    # logging.info('Apply lrd: {}'.format(lrd))
    tx = optax._src.combine.chain(tx, lrd_util.scale_by_lrd(lrd))

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
      dynamic_scale=dynamic_scale,
      ema_tx=ema_tx,
      ema_state=ema_state)
  return state


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)

  rng = random.PRNGKey(config.seed_jax)  # used to be 0

  image_size = 224

  if config.batch_size % jax.device_count() > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = config.batch_size // jax.process_count()

  platform = jax.local_devices()[0].platform

  if config.half_precision:
    if platform == 'tpu':
      input_dtype = tf.bfloat16
    else:
      input_dtype = tf.float16
  else:
    input_dtype = tf.float32

  dataset_val = torchloader_util.build_dataset(is_train=False, data_dir=config.torchload.data_dir, aug=config.aug)
  dataset_train = torchloader_util.build_dataset(is_train=True, data_dir=config.torchload.data_dir, aug=config.aug)

  sampler_train = torch.utils.data.DistributedSampler(dataset_train, num_replicas=jax.process_count(), rank=jax.process_index(), shuffle=True)
  sampler_val = torch.utils.data.DistributedSampler(dataset_val, num_replicas=jax.process_count(), rank=jax.process_index(), shuffle=False)


  collate_fn = functools.partial(torchloader_util.collate_and_reshape_fn, batch_size=local_batch_size)
  data_loader_train = torch.utils.data.DataLoader(
      dataset_train, sampler=sampler_train,
      batch_size=local_batch_size,
      num_workers=0,  # config.torchload.num_workers,
      pin_memory=True,
      drop_last=True,
      collate_fn=collate_fn,
  )
  data_loader_val = torch.utils.data.DataLoader(
      dataset_val, sampler=sampler_val,
      batch_size=local_batch_size,
      num_workers=config.torchload.num_workers,
      pin_memory=True,
      drop_last=False,
      collate_fn=collate_fn,
  )

  num_classes = len(dataset_train.classes)

  steps_per_epoch = len(data_loader_train)
  assert steps_per_epoch == len(dataset_train) // config.batch_size

  mixup_fn = torchloader_util.get_mixup_fn(config.aug)
  
  # dataset_builder = tfds.builder(config.dataset)
  # train_iter = create_input_iter(
  #     dataset_builder, local_batch_size, image_size, input_dtype, train=True,
  #     cache=config.cache, aug=config.aug)
  # eval_iter = create_input_iter(
  #     dataset_builder, local_batch_size, image_size, input_dtype, train=False,
  #     cache=config.cache)

  # steps_per_epoch = (
  #     dataset_builder.info.splits['train'].num_examples // config.batch_size
  # )

  # if config.num_train_steps == -1:
  #   num_steps = int(steps_per_epoch * config.num_epochs)
  # else:
  #   num_steps = config.num_train_steps

  # if config.steps_per_eval == -1:
  #   num_validation_examples = dataset_builder.info.splits[
  #       'validation'].num_examples
  #   num_validation_examples_split = math.ceil(num_validation_examples / jax.process_count())
  #   steps_per_eval = math.ceil(num_validation_examples_split / local_batch_size)
  # else:
  #   steps_per_eval = config.steps_per_eval

  # steps_per_checkpoint = int(steps_per_epoch * config.save_every_epochs)

  abs_learning_rate = config.learning_rate * config.batch_size / 256.

  # model_cls = getattr(models, config.model)
  model_cls = models_vit.VisionTransformer
  model = create_model(
      model_cls=model_cls, half_precision=config.half_precision, **config.model)

  learning_rate_fn = create_learning_rate_fn(
      config, abs_learning_rate, steps_per_epoch)

  state = create_train_state(rng, config, model, image_size, learning_rate_fn)

  if config.resume_dir != '':
    state = restore_checkpoint(state, config.resume_dir)
  elif config.pretrain_dir != '':
    logging.info('Loading from pre-training:')
    state = checkpoint_util.load_from_pretrain(state, config.pretrain_dir)

    # stds = jax.tree_util.tree_map(lambda x: np.array(x).std(), state.params)
    # logging.info('std: {}'.format(stds))
  
  # try to restore
  state = restore_checkpoint(state, workdir)

  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)

  # --------------------------------------------------------------------------------
  # up til now, state.params are for one device
  # image = jnp.ones([2, 224, 224, 3])
  # label = jnp.ones([2,], dtype=jnp.int32)
  # mutable_keys = [k for k in state.variables]
  # outcome = state.apply_fn(
  #     {'params': state.params,
  #      **state.variables,
  #     },
  #     rngs=dict(dropout=state.rng),
  #     inputs=image,
  #     mutable=mutable_keys,
  #     train=True)
  # logits, new_variables = outcome
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
      functools.partial(eval_step, ema_eval=(config.ema and config.ema_eval)),
      axis_name='batch')

  train_metrics = []
  hooks = []
  # if jax.process_index() == 0:
  #   hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]

  logging.info('Work dir: {}'.format(workdir))
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')

  if config.eval_only:
    # run eval only and return
    logging.info('Evaluating...')
    run_eval(state, p_eval_step, data_loader_val, local_batch_size, -1, num_classes)
    return

  epoch_offset = (step_offset + 1) // steps_per_epoch
  step = epoch_offset * steps_per_epoch
  assert step == int(state.step[0])  # sanity when loading

  best_acc = 0.
  for epoch in range(epoch_offset, int(config.num_epochs)):
    data_loader_train.sampler.set_epoch(epoch)  # reset random seed
    
    # ------------------------------------------------------------
    # train one epoch
    # ------------------------------------------------------------
    for i, batch in enumerate(data_loader_train):
      break
      # images, labels, labels_one_hot = batch
      
      # if mixup_fn:
      #   images, labels_one_hot = mixup_fn(images, labels)

      # batch = {'image': images, 'label': labels, 'label_one_hot': labels_one_hot}
      # batch = prepare_pt_data(batch, local_batch_size)

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
          summary['class_acc'] = summary.pop('train_accuracy')  # this is [0, 1]
          summary['step_tensorboard'] = epoch_1000x  # step for tensorboard

          writer.write_scalars(step + 1, summary)
          train_metrics = []
          train_metrics_last_t = time.time()

      step += 1  
    # ------------------------------------------------------------
    # finished one epoch: eval
    # ------------------------------------------------------------
    if True:
      summary = run_eval(state, p_eval_step, data_loader_val, local_batch_size, epoch, num_classes)
      best_acc = max(best_acc, summary['test_acc1'])

      # to make it consistent with PyTorch log
      summary['step_tensorboard'] = epoch  # step for tensorboard (no need to minus 1)

      writer.write_scalars(step + 1, summary)
      writer.flush()

    # ------------------------------------------------------------
    # finished one epoch: eval
    # ------------------------------------------------------------
    if (epoch + 1) % config.save_every_epochs == 0 or epoch + 1 == int(config.num_epochs):
      state = sync_batch_stats(state)
      save_checkpoint(state, workdir)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
  total_time = time.time() - start_time
  total_time_str = str(datetime.timedelta(seconds=int(total_time)))
  logging.info('Elapsed time: {}'.format(total_time_str))
  logging.info('Best accuracy: {}'.format(best_acc))

  if config.profile_memory:
    profile_memory(workdir)

  return state


def run_eval(state, p_eval_step, data_loader_val, local_batch_size, epoch, num_classes=1000):
  eval_metrics = []
  # sync batch statistics across replicas
  state = sync_batch_stats(state)
  tic = time.time()
  for i, batch in enumerate(data_loader_val):
    # images, labels, labels_one_hot = batch
    # batch = {'image': images, 'label': labels, 'label_one_hot': labels_one_hot}
    # batch = prepare_pt_data(batch, local_batch_size)

    metrics = p_eval_step(state, batch)
    eval_metrics.append(metrics)
    logging.info('{} / {}'.format(i, len(data_loader_val)))

  eval_metrics = jax.tree_map(lambda x: x[0], eval_metrics)
  eval_metrics = jax.device_get(eval_metrics)
  eval_metrics = jax.tree_map(lambda *args: np.concatenate(args), *eval_metrics)

  valid = np.where(eval_metrics['label'] >= 0)  # remove padded patch
  eval_metrics.pop('label')
  eval_metrics = jax.tree_util.tree_map(lambda x: x[valid], eval_metrics)

  toc = time.time() - tic
  logging.info('Eval time: {}, {} steps, {} samples'.format(
    str(datetime.timedelta(seconds=int(toc))),
    len(data_loader_val),
    len(eval_metrics['test_acc1'])))

  summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
  values = [f"{k}: {v:.6f}" for k, v in sorted(summary.items())]
  logging.info('eval epoch: %d, %s', epoch, ', '.join(values))
  return summary
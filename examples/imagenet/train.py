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

This script trains a ResNet-50 on the ImageNet dataset.
The data is loaded using tensorflow_datasets.
"""

import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax import optim
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
import models_vit

from utils import summary_util as summary_util  # must be after 'from clu import metric_writers'

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


def initialized(key, image_size, model):
  input_shape = (1, image_size, image_size, 3)
  @jax.jit
  def init(*args):
    return model.init(*args, train=False)
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))

  batch_stats = variables['batch_stats'] if 'batch_stats' in variables else flax.core.frozen_dict.FrozenDict()

  return variables['params'], batch_stats


def cross_entropy_loss(logits, labels):
  one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
  return jnp.mean(xentropy)


def compute_metrics(logits, labels):
  loss = cross_entropy_loss(logits, labels)
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
      decay_steps=cosine_epochs * steps_per_epoch)
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, cosine_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  return schedule_fn


def train_step(state, batch, learning_rate_fn):
  """Perform a single training step."""
  _, new_rng = jax.random.split(state.rng)
  # Bind the rng key to the device id (which is unique across hosts)
  # Note: This is only used for multi-host training (i.e. multiple computers
  # each with multiple accelerators).
  dropout_rng = jax.random.fold_in(state.rng, jax.lax.axis_index('batch'))
  def loss_fn(params):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        inputs=batch['image'],
        mutable=['batch_stats'],
        rngs=dict(dropout=dropout_rng),
        train=True)
    loss = cross_entropy_loss(logits, batch['label'])
    # weight_penalty_params = jax.tree_leaves(params)
    # weight_decay = 0.0001
    # weight_l2 = sum([jnp.sum(x ** 2)
    #                  for x in weight_penalty_params
    #                  if x.ndim > 1])
    # weight_penalty = weight_decay * 0.5 * weight_l2
    # loss = loss + weight_penalty
    return loss, (new_model_state, logits)

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
  new_model_state, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'])
  metrics['learning_rate'] = lr

  new_state = state.apply_gradients(
      grads=grads, batch_stats=new_model_state['batch_stats'], rng=new_rng)
  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
    new_state = new_state.replace(
        opt_state=jax.tree_multimap(
            functools.partial(jnp.where, is_fin),
            new_state.opt_state,
            state.opt_state),
        params=jax.tree_multimap(
            functools.partial(jnp.where, is_fin),
            new_state.params,
            state.params))
    metrics['scale'] = dynamic_scale.scale

  return new_state, metrics


def eval_step(state, batch):
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  logits = state.apply_fn(
      variables, batch['image'], train=False, mutable=False)
  return compute_metrics(logits, batch['label'])


def prepare_tf_data(xs):
  """Convert a input batch from tf Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    return x.reshape((local_device_count, -1) + x.shape[1:])

  return jax.tree_map(_prepare, xs)


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
                      cache):
  ds = input_pipeline.create_split(
      dataset_builder, batch_size, image_size=image_size, dtype=dtype,
      train=train, cache=cache)
  it = map(prepare_tf_data, ds)
  it = jax_utils.prefetch_to_device(it, 2)
  return it


class TrainState(train_state.TrainState):
  rng: Any
  batch_stats: Any
  dynamic_scale: flax.optim.DynamicScale


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  if jax.process_index() == 0:
    # get train state from the first replica
    state = jax.device_get(jax.tree_map(lambda x: x[0], state))
    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=3)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  if len(state.batch_stats) == 0:
    return state
  else:
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


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

  params, batch_stats = initialized(rng_init, image_size, model)

  tx = getattr(optax, config.opt_type)  # optax.adamw
  tx = tx(learning_rate=learning_rate_fn, **config.opt)
  state = TrainState.create(
      apply_fn=model.apply,
      params=params,
      tx=tx,
      rng=rng_state,
      batch_stats=batch_stats,
      dynamic_scale=dynamic_scale)
  return state


# from clu.metric_writers.summary_writer import SummaryWriter
# def summary_writer_write_scalars(self, step: int, scalars):
#   from IPython import embed; embed();
#   if (0 == 0): raise NotImplementedError
#   with self._summary_writer.as_default():
#     for key, value in scalars.items():
#       tf.summary.scalar(key, value, step=step)
# SummaryWriter.write_scalars = summary_writer_write_scalars


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

  rng = random.PRNGKey(0)

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

  dataset_builder = tfds.builder(config.dataset)
  train_iter = create_input_iter(
      dataset_builder, local_batch_size, image_size, input_dtype, train=True,
      cache=config.cache)
  eval_iter = create_input_iter(
      dataset_builder, local_batch_size, image_size, input_dtype, train=False,
      cache=config.cache)

  steps_per_epoch = (
      dataset_builder.info.splits['train'].num_examples // config.batch_size
  )

  if config.num_train_steps == -1:
    num_steps = int(steps_per_epoch * config.num_epochs)
  else:
    num_steps = config.num_train_steps

  if config.steps_per_eval == -1:
    num_validation_examples = dataset_builder.info.splits[
        'validation'].num_examples
    steps_per_eval = num_validation_examples // config.batch_size
  else:
    steps_per_eval = config.steps_per_eval

  steps_per_checkpoint = steps_per_epoch * 10

  abs_learning_rate = config.learning_rate * config.batch_size / 256.

  # model_cls = getattr(models, config.model)
  model_cls = models_vit.VisionTransformer
  model = create_model(
      model_cls=model_cls, half_precision=config.half_precision, **config.model)

  learning_rate_fn = create_learning_rate_fn(
      config, abs_learning_rate, steps_per_epoch)

  state = create_train_state(rng, config, model, image_size, learning_rate_fn)
  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)

  # --------------------------------------------------------------------------------
  # up til now, state.params are for one device
  # image = jnp.ones([2, 224, 224, 3])
  # label = jnp.ones([2,], dtype=jnp.int32)
  # logits, new_model_state = state.apply_fn(
  #     {'params': state.params,
  #      'batch_stats': state.batch_stats,
  #     },
  #     rngs=dict(dropout=state.rng),
  #     inputs=image,
  #     mutable=['batch_stats'],
  #     train=True)
  # --------------------------------------------------------------------------------
  state = jax_utils.replicate(state)

  p_train_step = jax.pmap(
      functools.partial(train_step, learning_rate_fn=learning_rate_fn),
      axis_name='batch')
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  train_metrics = []
  hooks = []
  if jax.process_index() == 0:
    hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')

  for step, batch in zip(range(step_offset, num_steps), train_iter):
    state, metrics = p_train_step(state, batch)
    for h in hooks:
      h(step)
    if step == step_offset:
      logging.info('Initial compilation completed.')

    epoch_1000x = int(step * config.batch_size / 1281167 * 1000)  # normalize to IN1K epoch anyway

    if config.get('log_every_steps'):
      train_metrics.append(metrics)
      if (step + 1) % config.log_every_steps == 0:
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

    if (step + 1) % steps_per_epoch == 0:
      epoch = step // steps_per_epoch
      eval_metrics = []

      # sync batch statistics across replicas
      state = sync_batch_stats(state)
      for _ in range(steps_per_eval):
        eval_batch = next(eval_iter)
        metrics = p_eval_step(state, eval_batch)
        eval_metrics.append(metrics)
      eval_metrics = common_utils.get_metrics(eval_metrics)
      summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
      logging.info('eval epoch: %d, loss: %.4f, accuracy: %.2f',
                   epoch, summary['loss'], summary['accuracy'] * 100)

      # to make it consistent with PyTorch log
      summary['test_acc1'] = summary.pop('accuracy') * 100  # rename
      summary['test_loss'] = summary.pop('loss')  # rename
      summary['step_tensorboard'] = epoch - 1  # step for tensorboard (-1: the end of first epoch is epoch 0)

      writer.write_scalars(step + 1, summary)
      writer.flush()
    if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
      state = sync_batch_stats(state)
      save_checkpoint(state, workdir)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

  return state

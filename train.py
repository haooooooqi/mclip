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

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import functools
import time, datetime
from typing import Any

from absl import logging
import torch
from torch import nn
import math

from absl import logging
from clu import metric_writers
import flax
from flax import jax_utils
from flax.training import common_utils
from flax.training import train_state
import jax
import jax.numpy as jnp
from jax import random
try:
    from jax.interpreters.sharded_jit import PartitionSpec
except ImportError:
    from jax.interpreters.pxla import PartitionSpec
import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import input_pipeline
import input_pipeline_laion
import input_pipeline_flickr
import models_mae

from utils import summary_util as summary_util  # must be after 'from clu import metric_writers'
from utils import checkpoint_util as ckp
from utils import torchloader_util
from utils import logging_util
from utils.torchloader_util import MEAN_RGB, STDDEV_RGB

from t5x.train_state_initializer import create_train_state
import t5x.partitioning
import t5x.rng
import t5x.model_info
import t5x.checkpoints

import jax.profiler

import numpy as np
import os
import random as _random

import torch
import torch.utils.data


def create_imagenet_input_iter(local_batch_size, data_layout, image_size, dtype, train, cache, seed=0, aug=None, dataset=""):
  if dataset == "":
    dataset = "imagenet2012:5.*.*"
  dataset_builder = tfds.builder(dataset)
  ds = input_pipeline.create_split(
      dataset_builder, local_batch_size, data_layout, image_size=image_size, dtype=dtype,
      train=train, cache=cache, seed=seed, aug=aug,)

  # ds = ds.apply(tf.data.experimental.ignore_errors(log_warning=False))
  ds = map(functools.partial(prepare_tf_data, batch_size=local_batch_size), ds)
  return ds

def create_laion_input_iter(local_batch_size, data_layout, image_size, dtype, train,
                      cache, seed=0, cfg=None, from_tags=None):
  ds = input_pipeline_laion.create_split(
      local_batch_size, data_layout, image_size=image_size, dtype=dtype,
      train=train, cache=cache, seed=seed, cfg=cfg, from_tags=from_tags)

  # ------------------------------------------------
  # x = next(iter(ds))
  # ------------------------------------------------

  ds = map(functools.partial(prepare_tf_data, batch_size=local_batch_size), ds)
  return ds


def create_flickr_input_iter(local_batch_size, data_layout, image_size, dtype, train,
                      cache, seed=0, cfg=None,):
  ds = input_pipeline_flickr.create_split(
      local_batch_size, data_layout, image_size=image_size, dtype=dtype,
      train=train, cache=cache, seed=seed, cfg=cfg,)

  # ------------------------------------------------
  # from IPython import embed; embed();
  # if (0 == 0): raise NotImplementedError
  # x = next(iter(ds))
  # ------------------------------------------------

  ds = map(functools.partial(prepare_tf_data, batch_size=local_batch_size), ds)
  return ds


def build_dataloaders(config, partitioner):

  batch_size = config.batch_size

  data_layout = partitioner.get_data_layout(batch_size)
  shard_id = data_layout.shard_id
  num_shards = data_layout.num_shards

  if batch_size % num_shards > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = batch_size // num_shards

  # ----------------------------------------
  logging_util.verbose_on()
  logging_util.sync_and_delay()
  logging.info("shard_id: {}".format(shard_id))
  logging_util.verbose_off()
  # ----------------------------------------

  image_size = config.image_size
  input_dtype = tf.float32

  # ImageNet tags
  from vocab.class_names import CLASS_NAMES
  from vocab.class_names import TEMPLATES
  templates = TEMPLATES[config.dataset]
  CLASS_NAMES = CLASS_NAMES[config.dataset]
  num_classes = len(CLASS_NAMES)

  while len(templates) * len(CLASS_NAMES) % 8 != 0:
    templates = templates * 2

  tags = []
  for c in CLASS_NAMES:
    for t in templates:
      if callable(t):
        tags.append(t(c))
      else:
        tags.append(
          t.format(c)
        )

  data_loader_tags = create_laion_input_iter(
      8,  # local_batch_size=8
      data_layout,
      image_size,
      input_dtype,
      train=False,
      cache=False, # config.cache,
      seed=config.seed_tf,
      cfg=config,
      from_tags=tags)

  data_loader_train = create_laion_input_iter(
      local_batch_size,
      data_layout,
      image_size,
      input_dtype,
      train=True,
      cache=False, # config.cache,
      seed=config.seed_tf,
      cfg=config)

  # val set is imagenet
  data_loader_val = create_imagenet_input_iter(
      local_batch_size,
      data_layout,
      image_size,
      input_dtype,
      train=False,
      cache=config.cache,
      seed=config.seed_tf,
      aug=config.aug,
      dataset=config.dataset,
  )
  # data_loader_val = None

  return data_loader_train, data_loader_val, data_loader_tags, num_classes


def build_dataloaders_retrieval(config, partitioner):

  batch_size = config.batch_size

  data_layout = partitioner.get_data_layout(batch_size)
  shard_id = data_layout.shard_id
  num_shards = data_layout.num_shards

  if batch_size % num_shards > 0:
    raise ValueError('Batch size must be divisible by the number of devices')
  local_batch_size = batch_size // num_shards

  # ----------------------------------------
  logging_util.verbose_on()
  logging_util.sync_and_delay()
  logging.info("shard_id: {}".format(shard_id))
  logging_util.verbose_off()
  # ----------------------------------------

  image_size = config.image_size
  input_dtype = tf.float32
  data_loader_train = create_flickr_input_iter(
      local_batch_size,
      data_layout,
      image_size,
      input_dtype,
      train=False,
      cache=False, # config.cache,
      seed=config.seed_tf,
      cfg=config)

  # val set is imagenet
  data_loader_val = create_flickr_input_iter(
      local_batch_size,
      data_layout,
      image_size,
      input_dtype,
      train=False,
      cache=config.cache,
      seed=config.seed_tf,
      cfg=config)

  return data_loader_train, data_loader_val


def print_sanity_check(batch, shard_id):
  """A sanity check when model partitions > 8 and data must be shared across nodes
  """
  logging_util.sync_and_delay(delay=shard_id * 0.5)
  logging_util.verbose_on()
  str = '{}'.format(batch['label'])
  str = (str + ' ' * 60)[:60] + '...'
  logging.info('shard: {}, label: {}'.format(shard_id, str))

  logging_util.sync_and_delay(delay=shard_id * 0.5)
  str = '{}'.format(np.array(batch['image'][:, 0, 0, 0]))
  str = (str + ' ' * 60)[:60] + '...'
  logging.info('shard: {}, image: {}'.format(shard_id, str))
  logging_util.verbose_off()
  return


def train_step(state, batch, model, rng):
  """Perform a single training step."""
  dropout_rng = jax.random.fold_in(rng, state.step)

  def loss_fn(params):
    """loss function used for training."""
    mutable = [k for k in state.flax_mutables]
    outcome = model.apply(
        {'params': params, **state.flax_mutables},
        inputs=batch,
        mutable=mutable,
        rngs=dict(dropout=dropout_rng),
        train=True)
    (loss, _, artifacts), new_mutables = outcome
    return loss, (new_mutables, artifacts)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)

  new_mutables, artifacts = aux[1]

  # metrics = {'loss': loss}
  metrics = {**artifacts}

  # only for metric logging
  lr = state._optimizer.optimizer_def.metric_learning_rate_fn(state.step)
  metrics['learning_rate'] = lr

  new_state = state.apply_gradient(
    grads,
    learning_rate=None,  # TODO: not used in adamw
    flax_mutables=new_mutables)
  return new_state, metrics

def eval_step(state, batch, encoded_tags, model, rng):
  variables = {'params': state.params, **state.flax_mutables}

  dropout_rng = jax.random.fold_in(rng, state.step)

  outcome = model.apply(variables, batch, train=False, mutable=False, rngs=dict(dropout=dropout_rng), encode_txt=False)
  loss, _, artifacts = outcome
  z_img = artifacts['z_img']

  labels = batch['label']

  z_txt = encoded_tags
  logits = jnp.einsum('nc,mc->nm', z_img, z_txt)

  # --------
  # dev: for maxout multiple templates
  # logits = logits.reshape([logits.shape[0], 1000, -1])
  # logits = jnp.max(logits, axis=-1)
  # --------

  pred_labels = jnp.argmax(logits, -1)
  accuracy = jnp.float32(pred_labels == labels)
  metrics = {'test_acc1': accuracy, 'label': labels}
  metrics = jax.tree_map(lambda x: jnp.reshape(x, [-1,]), metrics)
  return metrics


def eval_tags_step(state, batch, model, rng):
  variables = {'params': state.params, **state.flax_mutables}

  dropout_rng = jax.random.fold_in(rng, state.step)

  outcome = model.apply(variables, batch, train=False, mutable=False, rngs=dict(dropout=dropout_rng), encode_img=False)
  loss, _, artifacts = outcome
  z_txt = artifacts['z_txt']

  # metrics = {'test_loss': loss, 'imgs_vis': imgs_vis}
  return z_txt


def prepare_tf_data(xs, batch_size):
  """Convert a input batch from PyTorch Tensors to numpy arrays."""
  local_device_count = jax.local_device_count()
  def _prepare(x):
    # Use _numpy() for zero-copy conversion between TF and NumPy.
    x = x._numpy()  # pylint: disable=protected-access

    if x.shape[0] != batch_size:
      logging.info(f" =========  batch_size {batch_size} x.shape {x.shape}")
      pads = -np.ones((batch_size - x.shape[0],) + x.shape[1:], dtype=x.dtype)
      x = np.concatenate([x, pads], axis=0)

    # reshape (host_batch_size, height, width, 3) to
    # (local_devices, device_batch_size, height, width, 3)
    # return x.reshape((local_device_count, -1) + x.shape[1:])
    return x.reshape((-1,) + x.shape[1:])  # do not reshape into (local_devices, -1, ...)

  return jax.tree_map(_prepare, xs)


def profile_memory(workdir):
  jax.profiler.save_device_memory_profile("/tmp/memory.prof")
  if jax.process_index() == 0:
    logging.info('Saving memory.prof...')
    os.system('cd ~; gsutil cp /tmp/memory.prof {}'.format(workdir))
    logging.info('Saved memory.prof.')


def seed_worker(worker_id, shard_id):
    # worker_seed = torch.initial_seed() % 2**32 + shard_id
    worker_seed = worker_id + shard_id * 10000
    np.random.seed(worker_seed)
    _random.seed(worker_seed)

    # logging_util.verbose_on()
    # logging.info('worker_id: {}, shard_id: {}, worker_seed: {}'.format(worker_id, shard_id, worker_seed))
    # logging_util.verbose_off()


def set_seed_torch(seed):
  rng_torch = torch.Generator()
  rng_torch.manual_seed(seed)
  torch.manual_seed(seed)
  np.random.seed(seed)
  _random.seed(seed)
  return rng_torch


def train_and_evaluate(config: ml_collections.ConfigDict,
                       workdir: str) -> train_state.TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """
  # ------------------------------------
  # Set random seeds
  # ------------------------------------
  # rng_torch = set_seed_torch(config.seed_pt)
  tf.random.set_seed(config.seed_tf + jax.process_index())

  t5x.rng.set_hardware_rng_ops()
  rng = random.PRNGKey(config.seed_jax)
  # ------------------------------------

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0)

  image_size = 224  # TODO: move to config and model

  # ------------------------------------
  # Create partitioner
  # ------------------------------------
  partitioner = t5x.partitioning.PjitPartitioner(**config.partitioning)
  partitioner._logical_axis_rules += (('_null0', None),)
  partitioner._logical_axis_rules += (('_null1', None),)
  partitioner._logical_axis_rules += (('_null2', None),)
  partitioner._logical_axis_rules += (('classes', None),)

  # ------------------------------------
  # Create data loader
  # ------------------------------------
  data_loader_train, data_loader_val, data_loader_tags, num_classes = build_dataloaders(config, partitioner)  # we do not use data_loader_val
  batched_tags = [d for d in data_loader_tags]  # 1000x80 or 1000x7

  steps_per_epoch = config.samples_per_epoch // config.batch_size  # for lr schedule

  # ------------------------------------
  # Create model
  # ------------------------------------
  model = models_mae.ImageTextFeatLearner(config=config.model)
  # model = models_mae.ImageLearner(config=config.model)

  p_init_fn, state_axes, state_shape = create_train_state(
    config, model, steps_per_epoch, partitioner, init_batch=next(data_loader_train))
  rng_init, rng = jax.random.split(rng)

  t5x.model_info.log_model_info(None, state_shape, partitioner)

  # # ------------------------------------
  # # Create checkpointer
  # # ------------------------------------
  checkpointer = t5x.checkpoints.Checkpointer(
    train_state=state_shape,
    partitioner=partitioner,
    checkpoints_dir=workdir,
    keep=None,  # TODO: move to config
  )

  if config.resume_dir != '':
    state = ckp.restore_checkpoint(checkpointer, path=config.resume_dir)
  elif config.pretrain_dir != '':
    raise NotImplementedError
  else:
    logging.info('Initializing train_state...')
    state = p_init_fn(rng_init)
    logging.info('Initializing train_state done.')
    # stds = jax.tree_util.tree_map(lambda x: (x.shape, np.array(x).std()), state.params)
    # logging.info('std: {}'.format(stds))

    if config.get("model_img_feat_pretrain_dir", "") != '':
      mode = config.get("model_img_feat_pretrain_flax", "clip")
      if mode == "clip":
        # clip and openclip pretrain
        params_old = state.params
        params_ref = flax.training.checkpoints.restore_checkpoint(
            config.model_img_feat_pretrain_dir, target=None)
        print("img_feat params:", params_old["img_encoder"].keys())
        print("img_feat transformer params:", params_old["img_encoder"]["Transformer"].keys())
        print("pretrain transformer params:", params_ref["Transformer"].keys())

        print("img_feat transformer params:", params_old["img_encoder"].keys())
        print("pretrain transformer params:", params_ref.keys())
        # print(params_ref["embedding"])
        logging.info(params_ref["embedding"]["kernel"].sum())
        for key in params_ref.keys():
          logging.info(type(params_ref[key]))
          if key in params_old["img_encoder"]:
            logging.info("init from pretrain:", key)
          else:
            logging.info("pop pre-train", key)
        params_ref = {"img_encoder": params_ref}
        params_new = flax.core.frozen_dict.freeze({
            **params_ref,
            **params_old.pop("img_encoder")[0],
        })
        print("new params transformer params:", params_new["img_encoder"]["Transformer"].keys())
        params_new = partitioner.move_params_to_devices(params_new, state_axes.params)
        logging.info(params_new["img_encoder"]["embedding"]["kernel"].sum())
        state = state.replace_params(params_new)
      elif mode == "clip_text":
        # clip and openclip pretrain
        params_old = state.params
        params_ref_img_and_txt = flax.training.checkpoints.restore_checkpoint(
            config.model_img_feat_pretrain_dir, target=None)
        logging.info(params_ref_img_and_txt.keys())
        params_ref_img = params_ref_img_and_txt["img_encoder"]
        params_ref_txt = params_ref_img_and_txt["txt_encoder"]
        params_ref_img = {"img_encoder": params_ref_img}
        params_ref_txt = {"txt_encoder": params_ref_txt}
        logging.info(params_old.keys())

        params_new = flax.core.frozen_dict.freeze({
            **params_ref_img,
            **params_ref_txt,
            **params_old.pop("img_encoder")[0].pop("txt_encoder")[0],
        })
        logging.info(params_new.keys())
        logging.info(state_axes.params.keys())
        params_new = partitioner.move_params_to_devices(params_new, state_axes.params)
        state = state.replace_params(params_new)
      elif mode == "fairclip":
        # fairclip pretrain
        path = config.model_img_feat_pretrain_dir
        step = t5x.checkpoints.latest_step(path)
        path_chkpt = path if step is None else t5x.checkpoints.get_checkpoint_dir(path, step)
        state = checkpointer.restore(
          path=path_chkpt, fallback_state=state.state_dict(),
          state_transformation_fns=[remove_optimizer_state, remove_prefix]
        )
      elif mode == "hqclip":
        # fairclip pretrain
        path = config.model_img_feat_pretrain_dir
        step = t5x.checkpoints.latest_step(path)
        path_chkpt = path if step is None else t5x.checkpoints.get_checkpoint_dir(path, step)
        state = checkpointer.restore(
          path=path_chkpt, fallback_state=state.state_dict(),
          state_transformation_fns=[remove_optimizer_state, functools.partial(remove_prefix, hq=True)]
        )
      elif mode == "fairclip_text":
        # fairclip pretrain load text and img_feat
        path = config.model_img_feat_pretrain_dir
        step = t5x.checkpoints.latest_step(path)
        path_chkpt = path if step is None else t5x.checkpoints.get_checkpoint_dir(path, step)
        state = checkpointer.restore(
          path=path_chkpt, fallback_state=state.state_dict(),
          state_transformation_fns=[remove_optimizer_state, rename_prefix, ]
        )
      elif mode == "fairclip_text_interp":
        # fairclip pretrain load text and img_feat
        path = config.model_img_feat_pretrain_dir
        step = t5x.checkpoints.latest_step(path)
        path_chkpt = path if step is None else t5x.checkpoints.get_checkpoint_dir(path, step)
        state = checkpointer.restore(
          path=path_chkpt, fallback_state=state.state_dict(),
          state_transformation_fns=[
            remove_optimizer_state,
            rename_prefix,
          ]
        )
        logging.info(f"{dir(state)}")

        params = flax.core.frozen_dict.unfreeze(state.params)

        for key in params.keys():
          logging.info(type(params[key]))

        dim = params['img_encoder']['posembed_encoder']['pos_embedding'].shape
        params['img_encoder']['posembed_encoder']['pos_embedding'] = interpolate_pos_embed(
          params['img_encoder']['posembed_encoder']['pos_embedding'],
          [
            dim[0],
            int((config.image_size // (224 // (dim[1] - 1) ** 0.5)) ** 2 + 1),
            dim[2],
          ],
        )
        params_new = flax.core.frozen_dict.freeze({
            **params,
        })
        params_new = partitioner.move_params_to_devices(params_new, state_axes.params)
        state = state.replace_params(params_new)
      else:
        raise NotImplementedError

  t5x.model_info.log_state_info(state)

  # ------------------------------------------
  # for debugging with real tensors
  # batch = next(iter(data_loader_train))
  # mutable = [k for k in state.flax_mutables]
  # outcome = model.apply(
  #     {'params': state.params, **state.flax_mutables},
  #     inputs=batch,
  #     mutable=mutable,
  #     rngs=dict(dropout=rng),
  #     train=True)
  # # use the following to add checkpoints
  # import jaxlib
  # if isinstance(x, jnp.DeviceArray):
  #   pass
  # ------------------------------------------

  # --------------------------------------------------------
  # logging.info('Saving debug checkpoint: {}'.format(workdir))
  # checkpointer.save(state)
  # --------------------------------------------------------

  # step_offset > 0 if restarting from checkpoint
  # step_offset = int(state.step)
  # logging.info('step_offset: {}'.format(step_offset))

  # ------------------------------------------
  # build eval_tags_step
  eval_step_fn = functools.partial(eval_tags_step, model=model, rng=rng)  # (state, batch) -> metrics
  eval_axes = PartitionSpec('data', None,)
  partitioned_eval_tags_step = partitioner.partition(
        eval_step_fn,
        in_axis_resources=(state_axes, partitioner.data_partition_spec),
        out_axis_resources=eval_axes)


  # to create partitioned train_step
  train_step_fn = functools.partial(train_step, model=model, rng=rng)  # (state, batch) -> (state, metrics)
  partitioned_train_step = partitioner.partition(
        train_step_fn,
        in_axis_resources=(state_axes, partitioner.data_partition_spec),
        out_axis_resources=(state_axes, None),
        donate_argnums=(0,))

  eval_step_fn = functools.partial(eval_step, model=model, rng=rng)  # (state, batch) -> metrics
  eval_axes = None
  partitioned_eval_step = partitioner.partition(
        eval_step_fn,
        in_axis_resources=(state_axes, partitioner.data_partition_spec, None),
        out_axis_resources=eval_axes)

  # ------------------------------------------
  # debug
  # batch = next(iter(data_loader_train))
  # logging.info('To run partitioned_eval_step:')
  # outcome = train_step(state, batch, rng)
  # logging.info(jax.tree_map(lambda x: x.shape, outcome))
  # ------------------------------------------

  logging.info('Evaluating...')
  # run_eval(state, partitioned_eval_step, data_loader_val, epoch=-1)
  logging.info('Eval only...')
  summary = run_eval(
    state,
    batched_tags,
    partitioned_eval_tags_step,
    data_loader_val,
    partitioned_eval_step,
    config,
    num_classes=num_classes,
  )
  values = [f"{k}: {v:.6f}" for k, v in sorted(summary.items())]
  logging.info('eval: %s', ', '.join(values))
  return


def compute_encoded_tags(
  state,
  batched_tags,
  partitioned_eval_tags_step,
  num_classes,
):
  # Encoding tags: no data-parallism across nodes
  logging.info('Encoding tags...')
  encoded_tags = []
  for i, tags_batch in enumerate(batched_tags):
    z_txt = partitioned_eval_tags_step(state, tags_batch)
    encoded_tags.append(z_txt)
    if i % 100 == 0:
      logging.info('{} / {}'.format(i, len(batched_tags)))
  encoded_tags = jnp.concatenate(encoded_tags, axis=0)  # type: DeviceArray

  # ----------------
  # average multiple templates
  logging.info(f"NUM_CLASS {num_classes} encoded_tags {encoded_tags.shape}")
  encoded_tags = encoded_tags.reshape([num_classes, -1, encoded_tags.shape[-1]])  # [1000, 7, 512]
  encoded_tags = encoded_tags.mean(axis=1)
  encoded_tags /= jnp.linalg.norm(encoded_tags, axis=-1, keepdims=True) + 1e-8
  assert encoded_tags.shape[0] == num_classes
  # ----------------

  logging.info('Encoding tags done: {}'.format(encoded_tags.shape))
  return encoded_tags


def interpolate_pos_embed(param, target_shape):
    return_np = False
    npatch_wanted = target_shape[1] - 1
    npatch_actual = param.shape[1] - 1

    if npatch_wanted == npatch_actual:
        return param

    logging.info(f"interpolating positional embedding from {param.shape} to {target_shape}")
    logging.info(f"{param}")
    logging.info(f"{type(param)}")
    logging.info(f"{dir(param)}")
    assert int(math.sqrt(npatch_wanted)) * int(math.sqrt(npatch_wanted)) == npatch_wanted, (
        "wanted patch size should correspond to a perfect square otherwise there is no way to bicubic interpolate"
    )
    assert int(math.sqrt(npatch_actual)) * int(math.sqrt(npatch_actual)) == npatch_actual, (
        "actual patch size should correspond to a perfect square otherwise there is no way to bicubic interpolate"
    )
    if not isinstance(param, torch.Tensor):
        return_np = True
        param = torch.tensor(param.__array__())

    class_emb = param[:, 0]
    pos_emb = param[:, 1:]
    dim = param.shape[-1]
    pos_emb = pos_emb.reshape(1, int(math.sqrt(npatch_actual)), int(math.sqrt(npatch_actual)), dim).permute(0, 3, 1, 2)
    pos_emb = nn.functional.interpolate(
        pos_emb, size=(int(math.sqrt(npatch_wanted)), int(math.sqrt(npatch_wanted))), mode="bicubic",
    )
    pos_emb = pos_emb.permute(0, 2, 3, 1).view(1, -1, dim)
    output = torch.cat((class_emb.unsqueeze(0), pos_emb), dim=1)
    if return_np:
        output = output.numpy()
    return output


def adjust_resolution(optimizer_state, size=224):
  if size == 224:
      return optimizer_state
  logging.info(f"{optimizer_state}")
  logging.info(f"{optimizer_state['target']['img_encoder']['posembed_encoder']['pos_embedding']}")

  target_shape = [
    dim[0],
    int((size // (224 // (dim[1] - 1) ** 0.5)) ** 2 + 1),
    dim[2],
  ]

  optimizer_state = flax.core.frozen_dict.unfreeze(optimizer_state)
  logging.info(f"optimizer_state['target']['img_encoder']['posembed_encoder']['pos_embedding'].shape {optimizer_state['target']['img_encoder']['posembed_encoder']['pos_embedding'].shape}")

  optimizer_state['target']['img_encoder']['posembed_encoder']['pos_embedding'] = interpolate_pos_embed(
    optimizer_state['target']['img_encoder']['posembed_encoder']['pos_embedding'],
    optimizer_state['target']['img_encoder']['posembed_encoder']['pos_embedding'].shape,
  )

  optimizer_state = flax.core.frozen_dict.freeze(optimizer_state)

  return ckpt_optimizer_state


def remove_optimizer_state(ckpt_optimizer_state, optimizer_state):
    logging.info("pop state")
    ckpt_optimizer_state.pop("state")
    return ckpt_optimizer_state


def remove_prefix(ckpt_optimizer_state, optimizer_state, hq_img=False, hq_txt=False):
  print(ckpt_optimizer_state['target'].keys(), ckpt_optimizer_state['target']['img_encoder']['Transformer'].keys())
  if "txt_encoder" in ckpt_optimizer_state['target']:
    logging.info("pop txt_encoder")
    ckpt_optimizer_state['target'].pop('txt_encoder')
  ckpt_optimizer_state['target']['img_feat'] = ckpt_optimizer_state['target']['img_encoder']
  logging.info("pop img_encoder")
  ckpt_optimizer_state['target'].pop('img_encoder')
  logging.info("pop img_mlp1")
  if hq_img:
    ckpt_optimizer_state['target']['img_feat']['Transformer']['encoder_proj'] = ckpt_optimizer_state['target']['img_proj']['img_mlp1']
    ckpt_optimizer_state['target'].pop('img_proj')
  else:
    ckpt_optimizer_state['target']['img_feat']['Transformer']['encoder_proj'] = ckpt_optimizer_state['target']['img_mlp1']
    ckpt_optimizer_state['target'].pop('img_mlp1')

  if hq_txt:
    assert False
    ckpt_optimizer_state['target']['txt_proj']['txt_mlp1'] = ckpt_optimizer_state['target']['txt_mlp1']

  if "encoder_norm_pre" in ckpt_optimizer_state["target"]["img_feat"]['Transformer']:
    ckpt_optimizer_state["target"]["img_feat"]['Transformer']["encoder_prenorm"] = ckpt_optimizer_state["target"]["img_feat"]['Transformer']["encoder_norm_pre"]
    ckpt_optimizer_state["target"]["img_feat"]['Transformer'].pop("encoder_norm_pre")
    logging.info("rename encoder_norm_pre to encoder_prenorm")
  return ckpt_optimizer_state


def rename_prefix(ckpt_optimizer_state, optimizer_state):
  ckpt_optimizer_state['target']['img_encoder'] = ckpt_optimizer_state['target']['img_encoder']
  logging.info("rename img_encoder to img_feat")
  return ckpt_optimizer_state



# the implemention for pjit
def gather_by_einsum(x, ids):
  """kaiming: vmap + gather is slow with pjit; use einsum instead
  Args:
    x: [N, L, ...]
    ids: [N, K]
  """
  mat = jax.nn.one_hot(ids, x.shape[0])  # [N, K, L]
  x = jnp.einsum('l...,kl->k...', x, mat)
  return x


def run_eval(
  state,
  batched_tags,
  partitioned_eval_tags_step,
  data_loader_val,
  partitioned_eval_step,
  config,
  num_classes,
):
  tic = time.time()
  encoded_tags = compute_encoded_tags(state, batched_tags, partitioned_eval_tags_step, num_classes=num_classes)

  steps_per_eval = math.ceil(50000 / config.batch_size)
  eval_metrics = []
  for i in range(steps_per_eval):
    eval_batch = next(data_loader_val)
    metrics = partitioned_eval_step(state, eval_batch, encoded_tags)
    eval_metrics.append(metrics)
    if i % 10 == 0:
      logging.info('{} / {}, shape: {}'.format(i, steps_per_eval, eval_batch['image'].shape))

  eval_metrics = jax.device_get(eval_metrics)
  eval_metrics = jax.tree_map(lambda *args: np.concatenate(args), *eval_metrics)

  valid = np.where(eval_metrics['label'] >= 0)  # remove padded patch
  eval_metrics.pop('label')
  eval_metrics = jax.tree_util.tree_map(lambda x: x[valid], eval_metrics)

  toc = time.time() - tic
  logging.info('Eval time: {}, {} steps, {} samples'.format(
    str(datetime.timedelta(seconds=int(toc))),
    steps_per_eval,
    len(eval_metrics['test_acc1'])))

  summary = jax.tree_map(lambda x: x.mean(), eval_metrics)
  return summary

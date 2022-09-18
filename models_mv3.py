# Copyright 2021 Google LLC.
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

import functools
from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as random

import flax.linen as nn
from flax.linen.partitioning import remat
import optax

import t5x.layers

from utils import posembed_util
from utils import initializers_util
from utils import onlineknn_util

from models_mclr import AddPositionEmbs, gather_by_einsum, Encoder

# init hacks
INIT_VER = 'mae_jax_v2'

fixed_gaussian_init = nn.initializers.normal(stddev=0.02)
if INIT_VER == 'mae_jax_v2':
  clstoken_init = fixed_gaussian_init
  masktoken_init = fixed_gaussian_init
  posemb_init = fixed_gaussian_init  # not used if sincos

  patch_kernel_init = initializers_util.patch_kernel()
  patch_bias_init = nn.initializers.zeros  # different from PyTorch?

  # TF/PyTorch: qkv is [D, 3*D], fan_in + fan_out = 4*D.
  # JAX: q, k, v each is [D, D], fan_in + fan_out = 2*D. So we compensate by scale=0.5
  qkv_kernel_init = functools.partial(nn.initializers.variance_scaling, 0.5, "fan_avg", "uniform")()
  out_kernel_init = nn.initializers.xavier_uniform()

  mlp_kernel_init = nn.initializers.xavier_uniform()
  mlp_bias_init = nn.initializers.zeros

else:
  raise NotImplementedError


class VisionTransformer(nn.Module):
  """VisionTransformer."""
  sincos: bool
  patches: Any
  transformer: Any
  image_size: int
  hidden_size: int
  proj_layers: int
  proj_dim_hidden: int
  proj_dim_out: int
  classifier: str = 'token' # not used
  dtype: Any = jnp.float32

  def setup(self):
    self.conv_0 = t5x.layers.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding',
        kernel_init=patch_kernel_init,
        bias_init=patch_bias_init,
        kernel_axes=('_null0', '_null1', '_null2', 'embed'),
    )

    self.pos_embed = AddPositionEmbs(
        sincos=self.sincos,
        use_cls_token=False,
        img_shape=(self.image_size // self.patches.size[0],
                  self.image_size // self.patches.size[1],
                  self.hidden_size),
        name='posembed_encoder',
    )

    self.encoder = Encoder(name='Transformer',
                          hidden_size=self.hidden_size,
                          **self.transformer,
                          prefix='encoder')

    flip = False
    projector = []
    for i in range(self.proj_layers):
      kernel_axes = ('mlp', 'embed') if flip else ('embed', 'mlp')
      if i < self.proj_layers - 1:
        dim = self.proj_dim_hidden
        post_act = True
      else:
        dim = self.proj_dim_out
        post_act = False

      projector.append(t5x.layers.Dense(
          features=dim,
          dtype=self.dtype,
          kernel_init=mlp_kernel_init,
          kernel_axes=kernel_axes,
          use_bias=False,
          name='proj{}'.format(i),
      ))
      if post_act:
        projector.append(t5x.layers.TrainOnlyBatchNorm(
          dtype=self.dtype,
          axes=(kernel_axes[-1],),
          name='proj_norm{}'.format(i),
        ))
        projector.append(nn.relu)
      else:
        projector.append(t5x.layers.TrainOnlyBatchNorm(
          use_bias=False,
          use_scale=False,
          dtype=self.dtype,
          axes=(kernel_axes[-1],),
          name='proj_norm{}'.format(i),
        ))

      flip = not flip

    self.projector = nn.Sequential(projector)

  def __call__(self, inputs, train):
    x = self.conv_0(inputs)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    x = self.pos_embed(x)

    # apply the encoder
    x = self.encoder(x, train=train)

    # just do global average pooling
    p = jnp.mean(x, axis=1)

    # apply projector
    p = self.projector(p)

    return x, p


class SiameseLearner(nn.Module):
  """SiameseLearner."""

  encoder: Any
  image_size: int
  temp: float
  pred_layers: int
  pred_dim_hidden: int
  loss_type: str = 'info-nce'
  visualize: bool = False
  knn: Any = None
  dtype: Any = jnp.float32

  def setup(self):
    # source
    self.encoder.name = 'Source' # hack to change name
    self.source_encoder = VisionTransformer(image_size=self.image_size, **self.encoder)

    # target
    self.encoder.name = 'Target'
    self.target_encoder = VisionTransformer(image_size=self.image_size, **self.encoder)

    # knn
    if self.knn.on:
      self.online_knn = onlineknn_util.OnlineKNN(knn=self.knn)

      self.knn_norm = None
      if self.knn.postnorm == 'SBN0':
        self.knn_norm = t5x.layers.TrainOnlyBatchNorm(use_bias=False, use_scale=False,
                                dtype=self.dtype, axes=('embed',),
                                name='knn_postnorm')
      elif self.knn.postnorm != 'None':
        raise NotImplementedError

    # predictor
    flip = False if self.encoder.proj_layers % 2 == 0 else True
    predictor = []
    for i in range(self.pred_layers):
      kernel_axes = ('mlp', 'embed') if flip else ('embed', 'mlp')
      if i < self.pred_layers - 1:
        dim = self.pred_dim_hidden
        post_act = True
      else:
        dim = self.encoder.proj_dim_out
        post_act = False

      predictor.append(t5x.layers.Dense(
          features=dim,
          dtype=self.dtype,
          kernel_init=mlp_kernel_init,
          kernel_axes=kernel_axes,
          use_bias=False,
          name='pred{}'.format(i),
      ))
      if post_act:
        predictor.append(t5x.layers.TrainOnlyBatchNorm(
          dtype=self.dtype,
          axes=(kernel_axes[-1],),
          name='pred_norm{}'.format(i),
        ))
        predictor.append(nn.relu)
      else:
        predictor.append(t5x.layers.TrainOnlyBatchNorm(
          use_bias=False,
          use_scale=False,
          dtype=self.dtype,
          axes=(kernel_axes[-1],),
          name='pred_norm{}'.format(i),
        ))

      flip = not flip
    self.predictor = nn.Sequential(predictor)

  def apply_knn(self, x, labels, train):
    if not self.knn.on:
      return

    # => [N, E]
    if self.knn.pool == 'gap':
      x = jnp.mean(x, axis=1)
    else:
      raise NotImplementedError

    if self.knn_norm is not None:
      x = self.knn_norm(x)

    if self.knn.l2norm:
      l2norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1.e-12)
      x /= l2norm

    knn_accuracy = self.online_knn(x, labels, train=train)

    return knn_accuracy

  def compute_loss(self, source, target):
    # l2 normalization
    source /= jnp.sqrt(jnp.sum(source**2, axis=-1, keepdims=True) + 1.e-12)
    target /= jnp.sqrt(jnp.sum(target**2, axis=-1, keepdims=True) + 1.e-12)

    # split them
    s0, s1 = jnp.split(source, 2, axis=0)
    t0, t1 = jnp.split(target, 2, axis=0)

    if self.loss_type == 'cos':
      return self.cosine(s0, t1) + self.cosine(s1, t0)
    elif self.loss_type == 'info-nce':
      return self.info_nce(s0, t1) + self.info_nce(s1, t0)
    else:
      raise NotImplementedError

  def cosine(self, source, target):
    logits = jnp.einsum('nc,nc->n', source, target)
    return -logits.mean()

  def info_nce(self, source, target):
    # inter-image contrast
    logits = jnp.einsum('nc,mc->nm', source, target) / self.temp
    labels = jnp.arange(source.shape[0], dtype=jnp.int32)
    xent = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    return xent.mean() * self.temp

  def __call__(self, inputs, *, train, update=True):
    imgs = inputs['image']
    labels = inputs['label']

    assert len(imgs.shape) == 5 and imgs.shape[1] == 2
    # split the images
    imgs0 = imgs[:, 0, :, :, :]
    imgs1 = imgs[:, 1, :, :, :]
    imgs = jnp.concatenate([imgs0, imgs1], axis=0)

    # the augmentations are shared, just w/ or w/o masking
    _, p0 = self.source_encoder(imgs, train)
    x1, p1 = self.target_encoder(imgs, train)

    # optionally apply knn
    knn_accuracy = self.apply_knn(jnp.split(x1, 2, axis=0)[0], labels, train=(train and update))

    # predictor
    p0 = self.predictor(p0)

    # compute loss
    loss = self.compute_loss(p0, p1)

    if self.visualize:
      raise NotImplementedError
    else:
      outcome = None

    return loss, outcome, knn_accuracy

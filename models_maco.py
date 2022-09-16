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

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

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


class EncoderTransformer(nn.Module):
  """EncoderTransformer."""
  sincos: bool
  patches: Any
  transformer: Any
  image_size: int
  hidden_size: int
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

  def random_mask(self, x, mask_ratio):
    ids_shuffle, ids_restore = None, None

    if mask_ratio > 0.:
      L = x.shape[1]
      len_keep = int(L * (1. - mask_ratio))

      rng = self.make_rng('dropout')
      noise = random.uniform(rng, shape=x.shape[:2])

      ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
      ids_restore = jnp.argsort(ids_shuffle, axis=1)

      # keep the first subset
      ids_keep = ids_shuffle[:, :len_keep]
      x = gather_by_einsum(x, ids_keep)

    x = t5x.layers.with_sharding_constraint(x, ('batch', 'length', 'embed'))

    return x, (ids_shuffle, ids_restore)

  def __call__(self, inputs, train, mask_ratio=0.):
    x = self.conv_0(inputs)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    x = self.pos_embed(x)

    # masking: length -> length * mask_ratio
    x, ids = self.random_mask(x, mask_ratio)

    # apply the encoder
    x = self.encoder(x, train=train)

    return x, ids


class DecoderTransformer(nn.Module):
  """DecoderTransformer."""
  sincos: bool
  patches: Any
  transformer: Any
  image_size: int
  hidden_size: int
  classifier: str = 'token' # not used
  dtype: Any = jnp.float32

  def setup(self):
    self.L = (self.image_size // self.patches.size[0]) * (self.image_size // self.patches.size[1])
    self.conv_0 = t5x.layers.Dense(
      features=self.hidden_size,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      kernel_axes=('mlp', 'embed'),  # 'mlp' is split first
      name='bottleneck',
    )

    self.mask_token = t5x.layers.param_with_axes('mask_token',
                                            masktoken_init,
                                            (1, 1, self.hidden_size),
                                            jnp.float32,
                                            axes=('_null0', '_null1', 'embed'))

    self.pos_embed = AddPositionEmbs(
        sincos=self.sincos,
        use_cls_token=False,
        img_shape=(self.image_size // self.patches.size[0],
                  self.image_size // self.patches.size[1],
                  self.hidden_size),
        name='posembed_decoder',
    )

    self.encoder = Encoder(name='Transformer',
                          hidden_size=self.hidden_size,
                          **self.transformer,
                          prefix='decoder')

  def __call__(self, inputs, ids_restore, train, mask_ratio=0.):
    x = self.conv_0(inputs)
    len_keep = int(self.L * (1. - mask_ratio))
    len_mask = self.L - len_keep
    mask_tokens = jnp.tile(self.mask_token, [x.shape[0], len_mask, 1])
    x = jnp.concatenate([x, mask_tokens], axis=1)

    # restore positions
    x = gather_by_einsum(x, ids_restore)

    # add position embedding
    x = self.pos_embed(x)

    # apply the encoder
    x = self.encoder(x, train=train)

    return x, len_keep


class SiameseLearner(nn.Module):
  """SiameseLearner."""

  encoder: Any
  decoder: Any
  image_size: int
  mask_ratio: float
  temp: float
  loss_type: str = 'cos'
  intra_weight: float = 1.0
  visualize: bool = False
  knn: Any = None
  dtype: Any = jnp.float32

  def setup(self):
    # source
    self.encoder.name = 'Source' # hack to change name
    self.source_encoder = EncoderTransformer(image_size=self.image_size, **self.encoder)

    # target
    self.encoder.name = 'Target'
    self.target_encoder = EncoderTransformer(image_size=self.image_size, **self.encoder)

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

    # decoder
    self.decoder.name = 'Decoder'
    self.source_decoder = DecoderTransformer(image_size=self.image_size, **self.decoder)

    # predictor with encoder dimension
    self.predictor = t5x.layers.Dense(
        features=self.encoder.hidden_size,
        kernel_init=mlp_kernel_init,
        bias_init=mlp_bias_init,
        kernel_axes=('embed', 'classes'),  # 'mlp' is split first
        name='predictor',
    )

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

    if self.loss_type == 'cos':
      return self.cosine(source, target)
    elif self.loss_type == 'info-nce':
      return self.info_nce(source, target)
    else:
      raise NotImplementedError

  def cosine(self, source, target):
    logits = jnp.einsum('nqc,nqc->nq', source, target)
    return -logits.mean()

  def info_nce(self, source, target):
    # batch size
    N, L, _ = source.shape

    # inter-image contrast
    logits_inter = jnp.einsum('nqc,mqc->nqm', source, target) / self.temp
    labels_inter = jnp.tile(jnp.arange(N, dtype=jnp.int32)[:, None], (1, L))
    xent_inter = optax.softmax_cross_entropy_with_integer_labels(logits=logits_inter, labels=labels_inter)

    # intra-image contrast
    logits_intra = jnp.einsum('nqc,npc->nqp', source, target) / self.temp
    labels_intra = jnp.tile(jnp.arange(L, dtype=jnp.int32)[None, :], (N, 1))
    xent_intra = optax.softmax_cross_entropy_with_integer_labels(logits=logits_intra, labels=labels_intra)

    loss = xent_inter.mean() + xent_intra.mean() * self.intra_weight

    return loss * self.temp

  def __call__(self, inputs, *, train, update=True):
    imgs = inputs['image']
    labels = inputs['label']

    # the augmentations are shared, just w/ or w/o masking
    x0, ids = self.source_encoder(imgs, train, self.mask_ratio)
    x1, _ = self.target_encoder(imgs, train)

    # optionally apply knn
    knn_accuracy = self.apply_knn(x1, labels, train=(train and update))

    # decoder
    x0, len_keep = self.source_decoder(x0, ids[1], train, self.mask_ratio)

    # only compute loss on the masked tokens
    ids_masked = ids[0][:, len_keep:]
    x0 = gather_by_einsum(x0, ids_masked)
    x1 = gather_by_einsum(x1, ids_masked)

    # predictor
    x0 = self.predictor(x0)

    # compute loss
    loss = self.compute_loss(x0, x1)

    if self.visualize:
      raise NotImplementedError
    else:
      outcome = None

    return loss, outcome, knn_accuracy

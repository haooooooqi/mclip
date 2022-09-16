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


class VisionTransformer(nn.Module):
  """VisionTransformer."""

  mask_ratio: float
  sincos: bool
  norm_pix_loss: bool
  patches: Any
  transformer: Any
  hidden_size: int
  classifier: str = 'token'
  dtype: Any = jnp.float32
  decoder: Any = None
  visualize: bool = False
  knn: Any = None

  def random_mask(self, x):

    N, L, _ = x.shape  # batch, length, dim
    len_keep = int(L * (1 - self.mask_ratio))

    rng = self.make_rng('dropout')
    noise = random.uniform(rng, shape=x.shape[:2])

    ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
    ids_restore = jnp.argsort(ids_shuffle, axis=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = gather_by_einsum(x, ids_keep)

    x_masked = t5x.layers.with_sharding_constraint(x_masked, ('batch', 'length', 'embed'))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = jnp.ones([N, L])
    mask = t5x.layers.with_sharding_constraint(mask, ('batch', 'length'))
    mask = mask.at[:, :len_keep].set(0)
    # unshuffle to get the binary mask
    mask = gather_by_einsum(mask, ids_restore)
    mask = t5x.layers.with_sharding_constraint(mask, ('batch', 'length'))

    return x_masked, mask, ids_restore

  def patchify(self, imgs):
      """
      imgs: (N, H, W, 3)
      x: (N, L, patch_size**2 *3)
      """
      p, q = self.patches.size
      h, w = imgs.shape[1] // p, imgs.shape[2] // q

      x = jnp.reshape(imgs, (imgs.shape[0], h, p, w, q, 3))
      x = jnp.einsum('nhpwqc->nhwpqc', x)
      x = jnp.reshape(x, (imgs.shape[0], h * w, p * q * 3))
      return x

  def unpatchify(self, x):
      """
      x: (N, L, patch_size**2 *3)
      imgs: (N, H, W, 3)
      """
      p, q = self.patches.size
      h = w = int(x.shape[1]**.5)

      x = jnp.reshape(x, (x.shape[0], h, w, p, q, 3))
      x = jnp.einsum('nhwpqc->nhpwqc', x)
      imgs = jnp.reshape(x, (x.shape[0], h * p, w * q, 3))
      return imgs

  def compute_loss(self, imgs, pred, mask):
    """
    imgs: [N, H, W, 3]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    target = self.patchify(imgs)
    if self.norm_pix_loss:
      # target = jax.nn.normalize(target, axis=-1, epsilon=1.e-6)
      mean = jnp.mean(target, axis=-1, keepdims=True)
      var = jnp.var(target, axis=-1, keepdims=True)
      target = (target - mean) / (var + 1.e-6)**.5

    loss = jnp.square(pred - target)
    loss = jnp.mean(loss, axis=-1)  # [N, L], mean loss per patch

    loss = jnp.sum(loss * mask) / jnp.sum(mask)  # mean loss on removed patches
    return loss

  def visualization(self, imgs, pred, mask):
    """
    imgs: [N, H, W, 3]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    imgs_pred = self.unpatchify(pred)

    mask = jnp.repeat(jnp.expand_dims(mask, axis=-1), repeats=pred.shape[-1], axis=-1)
    mask = self.unpatchify(mask)  # 0 is keep, 1 is remove
    imgs_mask = imgs * (1 - mask)

    imgs_plus = imgs * (1 - mask) + imgs_pred * mask

    imgs_vis = jnp.concatenate(
    [jnp.concatenate([imgs, imgs_mask], axis=2),
     jnp.concatenate([imgs_pred, imgs_plus], axis=2)],
    axis=1)
    return imgs_vis

  def apply_encoder(self, inputs, train):
    use_cls_token = (self.classifier in ('token', 'tgap'))
    assert use_cls_token  # kaiming: TODO: support both?

    x = t5x.layers.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding',
        kernel_init=patch_kernel_init,
        bias_init=patch_bias_init,
        kernel_axes=('_null0', '_null1', '_null2', 'embed'),
        )(inputs)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    x = AddPositionEmbs(sincos=self.sincos,
                        use_cls_token=use_cls_token,
                        img_shape=(h, w, c),
                        name='posembed_encoder')(x)

    # masking: length -> length * mask_ratio
    x, mask, ids_restore = self.random_mask(x)
    ids_restore = jnp.reshape(ids_restore, [n, h, w])  # carries the shape info

    if use_cls_token:
      cls = t5x.layers.param_with_axes('cls',
                                      clstoken_init,
                                      (1, 1, c),
                                      jnp.float32,
                                      axes=('_null0', '_null1', 'embed'))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    # apply the encoder
    x = Encoder(name='Transformer', hidden_size=self.hidden_size,
                **self.transformer, prefix='encoder')(x, train=train)

    return x, mask, ids_restore

  def apply_decoder(self, x, ids_restore, train):
    use_cls_token = (self.classifier == 'token')

    n, h, w = ids_restore.shape
    ids_restore = jnp.reshape(ids_restore, [n, h * w])

    # apply the encoder-decoder bottleneck
    x = t5x.layers.Dense(
      features=self.decoder.hidden_size,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      kernel_axes=('mlp', 'embed'),  # 'mlp' is split first
      name='bottleneck')(x)

    # append mask token
    num_clstokens = 1 if use_cls_token else 0
    mask_token = t5x.layers.param_with_axes('mask_token',
                                            masktoken_init,
                                            (1, 1, self.decoder.hidden_size),
                                            jnp.float32,
                                            axes=('_null0', '_null1', 'embed'))
    mask_tokens = jnp.tile(mask_token, [n, ids_restore.shape[1] + num_clstokens - x.shape[1], 1])
    x_ = jnp.concatenate([x[:, num_clstokens:, :], mask_tokens], axis=1)  # no cls token
    x_ = gather_by_einsum(x_, ids_restore)

    # add decoder posembed (before cls token)
    x_ = AddPositionEmbs(sincos=self.sincos,
                        use_cls_token=use_cls_token,
                        img_shape=(h, w, self.decoder.hidden_size),
                        name='posembed_decoder')(x_)

    x = jnp.concatenate([x[:, :num_clstokens, :], x_], axis=1)  # append cls token

    # apply the decoder
    x = Encoder(name='TransformerDecoder', hidden_size=self.decoder.hidden_size,
                **self.decoder.transformer, prefix='decoder')(x, train=train)

    # apply the predictor
    x = t5x.layers.Dense(
      features=self.patches.size[0] * self.patches.size[1] * 3,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      kernel_axes=('embed', 'classes'),  # 'mlp' is split first
      name='pred')(x)

    # remove cls token
    pred = x[:, num_clstokens:, :]

    return pred

  def apply_knn(self, x, labels, train):
    if not self.knn.on:
      return

    # => [N, E]
    if self.knn.pool == 'gap':
      x = jnp.mean(x, axis=1)
    else:
      raise NotImplementedError

    if self.knn.postnorm == 'SBN0':
      x = t5x.layers.TrainOnlyBatchNorm(use_bias=False, use_scale=False,
                              dtype=self.dtype, axes=('embed',),
                              name='knn_postnorm')(x)
    elif self.knn.postnorm != 'None':
      raise NotImplementedError

    if self.knn.l2norm:
      l2norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1.e-12)
      x /= l2norm

    knn_accuracy = onlineknn_util.OnlineKNN(knn=self.knn)(x, labels, train=train)

    return knn_accuracy

  @nn.compact
  def __call__(self, inputs, *, train, update=True):
    imgs = inputs['image']
    labels = inputs['label']

    # apply encoder
    x, mask, ids_restore = self.apply_encoder(imgs, train=train)

    # optionally apply knn
    knn_accuracy = self.apply_knn(x, labels, train=(train and update))

    # apply decoder
    pred = self.apply_decoder(x, ids_restore, train=train)
    x = pred

    # compute loss
    loss = self.compute_loss(imgs, pred, mask)

    if self.visualize: # and not train:
      outcome = self.visualization(imgs, pred, mask)
    else:
      outcome = pred  # not used

    return loss, outcome, knn_accuracy
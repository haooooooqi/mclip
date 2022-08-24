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


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.
  """
  sincos: bool
  use_cls_token: bool
  img_shape: Shape  # [h, w, c]
  dtype: Any = jnp.float32

  def setup(self):
    h, w, c = self.img_shape

    num_clstokens = 1 if self.use_cls_token else 0
    pos_emb_shape = (1, num_clstokens + h * w, c)  # (batch_size, seq_len, emb_dim).

    if not self.sincos:
      init_fn = posemb_init
    else:
      pe_array = posembed_util.get_2d_sincos_pos_embed(c, (h, w), cls_token=self.use_cls_token)  # in numpy array
      init_fn = initializers_util.constant(value=pe_array, dtype=self.dtype)

    self.pe = t5x.layers.param_with_axes(
        'pos_embedding',
        init_fn,
        pos_emb_shape,
        jnp.float32,
        axes=('_null0', 'length', 'embed'))

    # kaiming: in MAE, we should always set posembed for cls_token as zero.
    # when loading for finetuning, this zero posembed can be tuned.
    # but this is not addressed here if sincos=False

  def __call__(self, inputs):
    """Applies AddPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """

    pe = jax.lax.stop_gradient(self.pe) if self.sincos else self.pe

    if self.use_cls_token:
      output = inputs + pe[:, 1:, :]
    else:
      output = inputs + pe

    return output


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  dtype: Dtype = jnp.float32
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)

  @nn.compact
  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = t5x.layers.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        kernel_axes=('embed', 'mlp'),
        name='Dense_0',
    )(inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    x = t5x.layers.with_sharding_constraint(x, ('batch', 'length', 'mlp'))
    output = t5x.layers.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        kernel_axes=('mlp', 'embed'),
        name='Dense_1',
    )(x)
    output = nn.Dropout(
        rate=self.dropout_rate)(
            output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    inputs: input data.
    mlp_dim: dimension of the mlp on top of attention block.
    dtype: the dtype of the computation (default: float32).
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout for attention heads.
    deterministic: bool, deterministic or not (to apply dropout).
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
  """

  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  droppath_rate: float = 0.0
  layer_id: int = None
  rescale_init: float = 1.

  @nn.compact
  def __call__(self, inputs, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(inputs)

    # ----------------------------------------------------
    # t5x
    MsaBlock = functools.partial(
      t5x.layers.MultiHeadDotProductAttention,
      qkv_kernel_init=lambda *args: qkv_kernel_init(*args) * self.rescale_init,
      out_kernel_init=lambda *args: out_kernel_init(*args) * self.rescale_init,
    )
    # ----------------------------------------------------

    x = MsaBlock(
        dtype=self.dtype,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
    )(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    # droppath
    x = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name='droppath_msa')(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate,
        kernel_init=lambda *args: mlp_kernel_init(*args) * self.rescale_init,
        bias_init=mlp_bias_init,
        )(y, deterministic=deterministic)
    # droppath
    y = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name='droppath_mlp')(y, deterministic=deterministic)

    return x + y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation.

  Attributes:
    num_layers: number of layers
    mlp_dim: dimension of the mlp on top of attention block
    num_heads: Number of heads in nn.MultiHeadDotProductAttention
    dropout_rate: dropout rate.
    attention_dropout_rate: dropout rate in self attention.
  """

  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  droppath_rate: float = 0.0
  prefix: str = 'encoder'
  rescale_init: float = 1.0
  remat_policy: str = 'none'

  @nn.compact
  def __call__(self, inputs, *, train):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 3  # (batch, len, emb)

    BlockLayer = Encoder1DBlock
    if self.remat_policy not in (None, 'none'):
      if self.remat_policy == 'minimal':
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      else:
        policy = None
      BlockLayer = remat(  # pylint: disable=invalid-name
          Encoder1DBlock,
          prevent_cse=True,
          policy=policy,
          static_argnums=(1,))  # "deterministic" is a static argument in Encoder1DBlock

    x = inputs
    for lyr in range(self.num_layers):
      deterministic = not train
      x = BlockLayer(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1) if self.droppath_rate > 0. else 0.,
          name=self.prefix + 'block_{:02d}'.format(lyr),
          num_heads=self.num_heads,
          layer_id=lyr,
          rescale_init=self.rescale_init,
        )(x, deterministic)
    encoded = t5x.layers.LayerNorm(name=self.prefix + '_norm', axes=('embed',))(x)

    return encoded


# the implementation for pjit
def gather_by_einsum(x, ids):
  """kaiming: vmap + gather is slow with pjit; use einsum instead
  Args:
    x: [N, L, ...]
    ids: [N, K]
  """
  mat = jax.nn.one_hot(ids, x.shape[1])  # [N, K, L]
  x = jnp.einsum('nl...,nkl->nk...', x, mat)
  return x


class VisionTransformer(nn.Module):
  """VisionTransformer."""

  mask_ratio: float
  sincos: bool
  patches: Any
  transformer: Any
  hidden_size: int
  classifier: str = 'token'
  dtype: Any = jnp.float32
  visualize: bool = False
  knn: Any = None
  clr: Any = None

  def random_mask(self, x, mask_ratio):

    if mask_ratio > 0.:
      N, L, _ = x.shape  # batch, length, dim
      len_keep = int(L * (1 - mask_ratio))

      rng = self.make_rng('dropout')
      noise = random.uniform(rng, shape=x.shape[:2])

      ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove

      # keep the first subset
      ids_keep = ids_shuffle[:, :len_keep]
      x = gather_by_einsum(x, ids_keep)

    x = t5x.layers.with_sharding_constraint(x, ('batch', 'length', 'embed'))

    return x

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

  def apply_encoder(self, inputs, train, mask_ratio=0.):
    use_cls_token = (self.classifier in {'token', 'tgap'})
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
    x = self.random_mask(x, mask_ratio)

    if use_cls_token:
      cls = t5x.layers.param_with_axes('cls',
                                      clstoken_init,
                                      (1, 1, c),
                                      jnp.float32,
                                      axes=('_null0', '_null1', 'embed'))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    # apply the encoder
    x = Encoder(name='Transformer', **self.transformer, prefix='encoder')(x, train=train)

    return x

  def apply_knn(self, x, labels, train):
    if not self.knn.on:
      return

    x = jax.lax.stop_gradient(x)
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

  def apply_projector(self, x):
    flip = False
    for i in range(self.clr.proj_layers):
      kernel_axes = ('mlp', 'embed') if flip else ('embed', 'mlp')
      if i < self.clr.proj_layers - 1:
        dim = self.clr.proj_dim_hidden
        post_relu = True
      else:
        dim = self.clr.proj_dim_out
        post_relu = False

      x = t5x.layers.Dense(
          features=dim,
          dtype=self.dtype,
          kernel_init=mlp_kernel_init,
          bias_init=mlp_bias_init,
          kernel_axes=kernel_axes,
          name='proj{}'.format(i),
      )(x)
      if post_relu:
        x = nn.relu(x)

      flip = not flip

    return x

  def compute_loss(self, source, target):
    # l2 normalization
    source /= jnp.sqrt(jnp.sum(source**2, axis=-1, keepdims=True) + 1.e-12)
    target /= jnp.sqrt(jnp.sum(target**2, axis=-1, keepdims=True) + 1.e-12)

    logits = jnp.einsum('nc,mc->nm', source, target) / self.clr.tau
    labels_one_hot = jnp.eye(logits.shape[0])

    # asymmetric loss
    xent = optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot)
    xent = xent.mean() * self.clr.tau
    return xent

  @nn.compact
  def __call__(self, inputs, *, train, train_knn=True):
    imgs = inputs['image']
    labels = inputs['label']

    # split the images
    assert len(imgs.shape) == 5 and imgs.shape[1] == 2
    imgs0 = imgs[:, 0, :, :, :]
    imgs1 = imgs[:, 1, :, :, :]

    # apply encoder to masked imgs0
    x0 = self.apply_encoder(imgs0, train, self.mask_ratio)
    x1 = self.apply_encoder(imgs1, train)

    # optionally apply knn
    knn_accuracy = self.apply_knn(x1, labels, train=(train and train_knn))

    # get the feature for contrastive learning
    if self.classifier == 'token':
      x0 = x0[:, 0]
      x1 = x1[:, 0]
    elif classifier == 'tgap':
      x0 = jnp.mean(x0[:, 1:], axis=1)
      x1 = jnp.mean(x1[:, 1:], axis=1)
    elif classifier == 'gap':
      x0 = jnp.mean(x0, axis=1)
      x1 = jnp.mean(x1, axis=1)
    else:
      raise NotImplementedError

    # apply projector
    x0 = self.apply_projector(x0)
    x1 = self.apply_projector(x1)

    # compute loss
    loss = self.compute_loss(x0, x1)

    if self.visualize: # and not train:
      raise NotImplementedError
    else:
      outcome = None  # not used

    return loss, outcome, knn_accuracy

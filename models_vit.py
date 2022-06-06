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

from absl import logging
import functools
from typing import Any, Callable, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as random

import t5x.layers

from utils import posembed_util
from utils import initializers_util


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# init hacks
# v1: JAX ViT; v2: PyTorch ViT; v3: v2 with fix
INIT_VER = 'v2'

fixed_gaussian_init = nn.initializers.normal(stddev=0.02)
if INIT_VER == 'v1':
  clstoken_init = nn.initializers.zeros
  posemb_init = fixed_gaussian_init
  patch_kernel_init = nn.initializers.lecun_uniform()
  patch_bias_init = nn.initializers.zeros
  msa_kernel_init = nn.initializers.xavier_uniform()
  mlp_kernel_init = nn.initializers.xavier_uniform()
  mlp_bias_init = nn.initializers.normal(stddev=1e-6)
  head_kernel_init=nn.initializers.zeros
elif INIT_VER == 'v2':
  clstoken_init = fixed_gaussian_init
  posemb_init = fixed_gaussian_init
  patch_kernel_init = fixed_gaussian_init
  patch_bias_init = fixed_gaussian_init  # bug from PyTorch code?
  msa_kernel_init = fixed_gaussian_init
  mlp_kernel_init = fixed_gaussian_init
  mlp_bias_init = nn.initializers.zeros
  # head_kernel_init = nn.initializers.normal(stddev=2e-5)
  head_kernel_init = fixed_gaussian_init
else:
  raise NotImplementedError


class IdentityLayer(nn.Module):
  """Identity layer, convenient for giving a name to an array."""

  @nn.compact
  def __call__(self, x):
    return x


# class AddPositionEmbs(nn.Module):
#   """Adds (optionally learned) positional embeddings to the inputs.

#   Attributes:
#     posemb_init: positional embedding initializer.
#   """

#   posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

#   @nn.compact
#   def __call__(self, inputs):
#     """Applies AddPositionEmbs module.

#     By default this layer uses a fixed sinusoidal embedding table. If a
#     learned position embedding is desired, pass an initializer to
#     posemb_init.

#     Args:
#       inputs: Inputs to the layer.

#     Returns:
#       Output tensor with shape `(bs, timesteps, in_dim)`.
#     """
#     # inputs.shape is (batch_size, seq_len, emb_dim).
#     assert inputs.ndim == 3, ('Number of dimensions should be 3,'
#                               ' but it is: %d' % inputs.ndim)
#     pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
#     pe = t5x.layers.param_with_axes(
#         'pos_embedding',
#         self.posemb_init,
#         pos_emb_shape,
#         jnp.float32,
#         axes=('_null0', 'length', 'embed'))
#     return inputs + pe


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
      raise NotImplementedError
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

    if self.use_cls_token and inputs.shape[1] == pe.shape[1] - 1:
      output = inputs + pe[:, 1:, :]
    else:
      assert inputs.shape[1] == pe.shape[1]
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
  adapter: Any = None

  def apply_adapter(self, x, deterministic, name):
    adater_mlp_dim = round(self.adapter.mlp_dim_ratio * x.shape[-1])
    z = MlpBlock(
        mlp_dim=adater_mlp_dim,
        dtype=self.dtype,
        dropout_rate=0.,
        kernel_init=lambda *args: mlp_kernel_init(*args) * self.adapter.rescale_init,
        bias_init=mlp_bias_init,
        name=name,
        )(x, deterministic=deterministic)
    x = z + x
    return x


  @nn.compact
  def __call__(self, inputs, *, deterministic):
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
      kernel_init=msa_kernel_init,
    )
    # original
    # MsaBlock = functools.partial(
    #   nn.MultiHeadDotProductAttention,
    #   kernel_init=msa_kernel_init,
    #   broadcast_dropout=False,
    #   deterministic=deterministic,
    # )
    # ----------------------------------------------------

    x = MsaBlock(
        dtype=self.dtype,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
    )(x, x)
    assert self.dropout_rate == 0.
    # x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)  # not used

    # adapter
    if self.adapter and self.adapter.on_use:
      x = self.apply_adapter(x, deterministic, name='adapter_msa')

    # droppath
    x = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name='droppath_msa')(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate,
        kernel_init=mlp_kernel_init,
        bias_init=mlp_bias_init,
        )(y, deterministic=deterministic)

    # adapter
    if self.adapter and self.adapter.on_use:
      y = self.apply_adapter(y, deterministic, name='adapter_mlp')

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
  adapter: Any = None

  @nn.compact
  def __call__(self, inputs, *, train, encoder_norm=True, stopgrad_blocks=None):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert inputs.ndim == 3  # (batch, len, emb)

    x = inputs
    
    # Input Encoder
    for lyr in range(self.num_layers):
      dp = self.droppath_rate * lyr / (self.num_layers - 1) if self.droppath_rate > 0. else 0.
      name = self.prefix + 'block_{:02d}'.format(lyr)
      # logging.info('layer: {}, dp: {}'.format(name, dp))
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droppath_rate=dp,
          name=name,
          num_heads=self.num_heads,
          layer_id=lyr,
          adapter=self.adapter,
        )(x, deterministic=not train)
      if stopgrad_blocks is not None and stopgrad_blocks == lyr + 1:
        x = jax.lax.stop_gradient(x)
        logging.info('Stop gradient after block: {}'.format(name))
    encoded = t5x.layers.LayerNorm(name=self.prefix + '_norm', axes=('embed',))(x) if encoder_norm else x

    return encoded


# the implemention for pjit
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

  num_classes: int
  patches: Any
  transformer: Any
  hidden_size: int
  resnet: Optional[Any] = None
  classifier: str = 'token'
  dtype: Any = jnp.float32
  rescale_head_init: float = 1.
  stopgrad_blocks: int = -1
  predictor: Any = None
  adapter: Any = None
  sincos: bool = True
  split: Any = None

  def apply_predictor(self, x, train, img_shape):

    # apply the encoder-predictor bottleneck
    x = t5x.layers.Dense(
      features=self.predictor.hidden_size,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      kernel_axes=('mlp', 'embed'),  # 'mlp' is split first
      name='pred_bottleneck')(x)

    # add predictor pos emb
    # x = AddPositionEmbs(posemb_init=posemb_init, name='posembed_encoder')(x)
    use_cls_token = (self.classifier in {'token', 'tgap'})
    x = AddPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, img_shape=img_shape + (x.shape[-1],), name='pred_posembed')(x)

    # apply the predictor
    x = Encoder(name='pred', **self.predictor.transformer)(x, train=train)

    return x

  def apply_split(self, x):
    """ x: [N, L, C] -> [N * 4, L // 4, C] """
    N, L, C = x.shape

    rng = self.make_rng('dropout')
    noise = random.uniform(rng, shape=x.shape[:2])

    ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
    # ids_restore = jnp.argsort(ids_shuffle, axis=1)

    x_shuffled = gather_by_einsum(x, ids_shuffle)
    x_shuffled = t5x.layers.with_sharding_constraint(x_shuffled, ('batch', 'length', 'embed'))

    splits = self.split.splits
    x_shuffled = jnp.reshape(x_shuffled, [N * splits, L // splits, C])
    x_shuffled = t5x.layers.with_sharding_constraint(x_shuffled, ('batch', 'length', 'embed'))

    return x_shuffled

  def undo_split(self, x):
    """ x: [N * 4, C] -> [N, C] """
    N, C = x.shape
    splits = self.split.splits
    x = jnp.reshape(x, [N // splits, splits, C])
    x = t5x.layers.with_sharding_constraint(x, ('batch', 'length', 'embed'))
    x = jnp.mean(x, axis=1)  # average pool
    return x

  @nn.compact
  def __call__(self, inputs, *, train):
    x = inputs
    # (Possibly partial) ResNet root.
    assert self.resnet == None

    n, h, w, c = x.shape
    # We can merge s2d+emb into a single conv; it's the same.
    # x = nn.Conv(
    #     features=self.hidden_size,
    #     kernel_size=self.patches.size,
    #     strides=self.patches.size,
    #     padding='VALID',
    #     name='embedding',
    #     kernel_init=patch_kernel_init,
    #     bias_init=patch_bias_init,
    #     )(x)
    x = t5x.layers.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding',
        kernel_init=patch_kernel_init,
        bias_init=patch_bias_init,
        kernel_axes=('_null0', '_null1', '_null2', 'embed'),
        )(x)

    # Here, x is a grid of embeddings.

    # Transformer.
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # we add posemb here
    # x = AddPositionEmbs(posemb_init=posemb_init, name='posembed_encoder')(x)
    use_cls_token = (self.classifier in {'token', 'tgap'})
    x = AddPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, img_shape=(h, w, c), name='posembed_encoder')(x)

    if self.split and self.split.on_use:
      x = self.apply_split(x)
      n = x.shape[0]

    # If we want to add a class token, add it here.
    if self.classifier in {'token', 'tgap'}:
      cls = t5x.layers.param_with_axes('cls', clstoken_init, (1, 1, c), jnp.float32, axes=('_null0', '_null1', 'embed'))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    use_encoder_norm = (self.predictor == None and self.classifier == 'token') or (self.predictor != None)
    x = Encoder(name='Transformer', **self.transformer, adapter=self.adapter)(
      x, train=train, encoder_norm=use_encoder_norm, stopgrad_blocks=self.stopgrad_blocks)

    if not self.adapter.on_use and self.stopgrad_blocks == self.transformer.num_layers + 1:
      x = jax.lax.stop_gradient(x)
      logging.info('Stop gradient.')

    # apply the predictor
    if self.predictor.transformer.num_layers > 0:
      x = self.apply_predictor(x, train, img_shape=(h, w,))

    if self.classifier == 'token':
      x = x[:, 0]
    elif self.classifier == 'tgap':
      x = x[:, 1:]
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
      x = t5x.layers.LayerNorm(name='fc_norm', axes=('embed',))(x)
    elif self.classifier == 'gap':
      x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
      x = t5x.layers.LayerNorm(name='fc_norm', axes=('embed',))(x)
    else:
      raise ValueError(f'Invalid classifier={self.classifier}')

    x = IdentityLayer(name='pre_logits')(x)
    
    if self.split and self.split.on_use:
      x = self.undo_split(x)

    if self.num_classes:
      x = t5x.layers.Dense(
          features=self.num_classes,
          kernel_init=lambda *args: head_kernel_init(*args) * self.rescale_head_init,
          kernel_axes=('embed', 'classes'),
          name='head',
      )(x)

    return x
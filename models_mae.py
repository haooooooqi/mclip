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
from xml.sax.xmlreader import InputSource

import jax
import jax.numpy as jnp
import jax.random as random
import optax

import flax.linen as nn

import t5x.layers

from utils import posembed_util
from utils import initializers_util


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
  
  # patch_kernel_init = fixed_gaussian_init
  patch_kernel_init = initializers_util.patch_kernel()
  patch_bias_init = nn.initializers.zeros  # different from PyTorch?

  # msa_kernel_init = fixed_gaussian_init

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


class Add2DPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.
  """
  sincos: bool
  use_cls_token: bool
  dtype: Any = jnp.float32

  def get_pos_emb(self, x):
    _, l, c = x.shape
    h = w = int(l**.5)
    assert h * w == l

    num_clstokens = 1 if self.use_cls_token else 0
    pos_emb_shape = (1, num_clstokens + h * w, c)  # (batch_size, seq_len, emb_dim).

    if not self.sincos:
      raise NotImplementedError
      init_fn = posemb_init
    else:
      pe_array = posembed_util.get_2d_sincos_pos_embed(c, (h, w), cls_token=self.use_cls_token)  # in numpy array
      init_fn = initializers_util.constant(value=pe_array, dtype=self.dtype)

    pe = t5x.layers.param_with_axes(
        'pos_embedding',
        init_fn,
        pos_emb_shape,
        jnp.float32,
        axes=('_null0', 'length', 'embed'))

    # kaiming: in MAE, we should always set posembed for cls_token as zero.
    # when loading for finetuning, this zero posembed can be tuned.
    # but this is not addressed here if sincos=False

    return pe

  @nn.compact
  def __call__(self, inputs):
    """Applies Add2DPositionEmbs module.

    By default this layer uses a fixed sinusoidal embedding table. If a
    learned position embedding is desired, pass an initializer to
    posemb_init.

    Args:
      inputs: Inputs to the layer.

    Returns:
      Output tensor with shape `(bs, timesteps, in_dim)`.
    """
    pe = self.get_pos_emb(inputs)
    pe = jax.lax.stop_gradient(pe) if self.sincos else self.pe

    if self.use_cls_token:
      output = inputs + pe[:, 1:, :]
    else:
      output = inputs + pe

    return output


class Add1DPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs.
  """
  sincos: bool
  dtype: Any = jnp.float32
  posemb_init: Any = None

  def get_pos_emb(self, x):
    _, l, c = x.shape
    pos_emb_shape = (1, l, c)  # (batch_size, seq_len, emb_dim).

    if not self.sincos:
      init_fn = self.posemb_init
    else:
      raise NotImplementedError

    pe = t5x.layers.param_with_axes(
        'pos_embedding',
        init_fn,
        pos_emb_shape,
        jnp.float32,
        axes=('_null0', 'length', 'embed'))

    return pe

  @nn.compact
  def __call__(self, inputs):
    pe = self.get_pos_emb(inputs)    
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


class Encoder1DBlockCross(nn.Module):
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
  def __call__(self, q, kv, *, deterministic):
    """Applies Encoder1DBlock module.

    Args:
      inputs: Inputs to the layer.
      deterministic: Dropout will not be applied when set to true.

    Returns:
      output after transformer encoder block.
    """
    assert self.dropout_rate == 0  # for now
    assert self.droppath_rate == 0  # for now

    # Attention block.
    assert q.ndim == 3, f'Expected (batch, seq, hidden) got {q.shape}'
    assert kv.ndim == 3, f'Expected (batch, seq, hidden) got {kv.shape}'
    kv = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(kv)

    # ----------------------------------------------------
    # declare block type
    MsaBlock = functools.partial(
      t5x.layers.MultiHeadDotProductAttention,
      qkv_kernel_init=lambda *args: qkv_kernel_init(*args) * self.rescale_init,
      out_kernel_init=lambda *args: out_kernel_init(*args) * self.rescale_init,
    )
    # ----------------------------------------------------

    # first block: self-attention
    identity = q
    x = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(q)
    x = MsaBlock(
        dtype=self.dtype,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        name='SelfAtt'
    )(x, x)
    x = x + identity

    # second block: cross-attention
    identity = x
    x = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(x)
    x = MsaBlock(
        dtype=self.dtype,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        name='CrossAtt'
    )(x, kv)
    x = x + identity

    # MLP block.
    identity = x
    y = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate,
        kernel_init=lambda *args: mlp_kernel_init(*args) * self.rescale_init,
        bias_init=mlp_bias_init,
        )(y, deterministic=deterministic)
    y = y + identity
    return y


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

    x = inputs
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1) if self.droppath_rate > 0. else 0.,
          name=self.prefix + 'block_{:02d}'.format(lyr),
          num_heads=self.num_heads,
          layer_id=lyr,
          rescale_init=self.rescale_init,
        )(x, deterministic=not train)
    encoded = t5x.layers.LayerNorm(name=self.prefix + '_norm', axes=('embed',))(x)

    return encoded


class EncoderCross(nn.Module):
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

  @nn.compact
  def __call__(self, inputs, *, train):
    """Applies Transformer model on the inputs.

    Args:
      inputs: Inputs to the layer.
      train: Set to `True` when training.

    Returns:
      output of a transformer encoder.
    """
    assert isinstance(inputs, tuple)
    assert len(inputs) == 2

    q, kv = inputs
    assert q.ndim == 3  # (batch, len of q, emb)
    assert kv.ndim == 3  # (batch, len of kv, emb)

    x = q
    for lyr in range(self.num_layers):
      x = Encoder1DBlockCross(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1) if self.droppath_rate > 0. else 0.,
          name=self.prefix + 'block_{:02d}'.format(lyr),
          num_heads=self.num_heads,
          layer_id=lyr,
          rescale_init=self.rescale_init,
        )(q=x, kv=kv, deterministic=not train)
    encoded = t5x.layers.LayerNorm(name=self.prefix + '_norm', axes=('embed',))(x)

    return encoded


# the implemention for pmap
def gather(x, ids):
  return x[ids, :]
vmapped_gather = jax.vmap(gather, in_axes=(0, 0), out_axes=0, axis_name='batch')


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


def random_mask(rng, x, mask_ratio, bias=None):
  """
  x: [N, L, C] input
  bias: [N, L], an additional map to the noise map (small is keep, large is remove)
  """

  N, L, _ = x.shape  # batch, length, dim
  len_keep = int(L * (1 - mask_ratio))

  noise = random.uniform(rng, shape=x.shape[:2])  

  if bias is not None:
    noise += bias

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


class LanguageTransformer(nn.Module):
  """LanguageTransformer."""

  mask_ratio: float
  sincos: bool
  vocab_size: int
  transformer: Any
  hidden_size: int
  dtype: Any = jnp.float32
  decoder: Any = None

  def setup(self):
    """
    declare all param layers based on inputs
    """
    # ------------------------
    # define encoder
    # ------------------------
    encoder_layers = {}
    encoder_layers['token_emb'] = t5x.layers.Embed(
      num_embeddings=self.vocab_size,
      features=self.hidden_size,
      embedding_init=fixed_gaussian_init,
      one_hot=True,
      axes=['classes', 'embed'],  # do not use 'vocab' 
      name='token_embedding')
    encoder_layers['pos_emb'] = Add1DPositionEmbs(sincos=self.sincos, posemb_init=fixed_gaussian_init, name='posembed_encoder')
    encoder_layers['blocks'] = Encoder(name='Transformer', **self.transformer, prefix='encoder')
    self.encoder_layers = encoder_layers

    # ------------------------
    # define decoder
    # ------------------------
    decoder_layers = {}
    decoder_layers['bottleneck'] = t5x.layers.Dense(
      features=self.decoder.hidden_size,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      kernel_axes=('mlp', 'embed'),  # 'mlp' is split first
      name='bottleneck')
    decoder_layers['mask_token'] = t5x.layers.param_with_axes(
      'mask_token', masktoken_init, (1, 1, self.decoder.hidden_size),
      jnp.float32, axes=('_null0', '_null1', 'embed'))
    decoder_layers['txt_modal_token'] = t5x.layers.param_with_axes(
      'txt_modal_token', masktoken_init, (1, 1, self.decoder.hidden_size),
      jnp.float32, axes=('_null0', '_null1', 'embed'))
    decoder_layers['posemb'] = Add1DPositionEmbs(sincos=self.sincos, posemb_init=fixed_gaussian_init, name='posembed_decoder')
    # decoder_layers['blocks'] = Encoder(name='TransformerDecoder', **self.decoder.transformer, prefix='decoder')
    # decoder_layers['pred'] = t5x.layers.Dense(
    #   features=self.vocab_size,
    #   kernel_init=mlp_kernel_init,
    #   bias_init=mlp_bias_init,
    #   kernel_axes=('embed', 'classes'),  # 'mlp' is split first
    #   name='pred')
    self.decoder_layers = decoder_layers

  def compute_loss(self, txt, pred, mask, is_valid):
    """
    txt: [N, L]
    pred: [N, L, K]
    mask: [N, L], 0 is keep (known), 1 is remove (unknown)
    is_valid: [N, L], 1 is real, 0 is pad
    """    
    labels_one_hot = jax.nn.one_hot(txt, self.vocab_size)  # [N, L, K]
    loss = optax.softmax_cross_entropy(pred, labels_one_hot)  # [N, L]

    # has loss: mask=1 (unknown) and is_valid=1 (real)
    mask = jnp.logical_and(jnp.bool_(mask), jnp.bool_(is_valid))
    mask = jnp.float32(mask)

    loss = jnp.sum(loss * mask) / jnp.sum(mask)  # mean loss on removed patches

    return loss

  def apply_encoder(self, inputs, train):
    x = inputs
    
    x = self.encoder_layers['token_emb'](x)

    x = self.encoder_layers['pos_emb'](x)

    # masking: length -> length * mask_ratio
    x, mask, ids_restore = random_mask(self.make_rng('dropout'), x, self.mask_ratio)

    # apply the encoder
    x = self.encoder_layers['blocks'](x, train=train)

    return x, mask, ids_restore

  def apply_unshuffle(self, x, ids_restore):
    """bottleneck projection, unshuffle, add pos emb"""
    n, l = ids_restore.shape

    # apply the encoder-decoder bottleneck
    x = self.decoder_layers['bottleneck'](x)
    x_part = x  # take a copy

    # append mask token
    mask_token = self.decoder_layers['mask_token']
    mask_tokens = jnp.tile(mask_token, [n, ids_restore.shape[1] - x.shape[1], 1])
    x_ = jnp.concatenate([x, mask_tokens], axis=1)  # no cls token
    x_ = gather_by_einsum(x_, ids_restore)

    # add decoder posembed
    x_full = self.decoder_layers['posemb'](x_)

    return x_full, x_part

  def apply_decoder(self, x, train):

    # apply the decoder
    x = self.decoder_layers['blocks'](x, train=train)

    # apply the predictor
    x = self.decoder_layers['pred'](x)

    return x


class VisionTransformer(nn.Module):
  """VisionTransformer."""

  mask_ratio: float
  sincos: bool
  norm_pix_loss: bool
  patches: Any
  transformer: Any
  hidden_size: int
  vocab_size: int  # for joint decoder
  classifier: str = 'token'
  dtype: Any = jnp.float32
  decoder: Any = None

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

  def remove_cls_token(self, x):
    use_cls_token = (self.classifier == 'token')
    num_clstokens = 1 if use_cls_token else 0
    # remove cls token
    x = x[:, num_clstokens:, :]
    return x

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
    use_cls_token = (self.classifier == 'token')
    assert use_cls_token  # kaiming: TODO: support both?

    x = self.encoder_layers['patch_emb'](inputs)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    x = self.encoder_layers['pos_emb'](x)

    # masking: length -> length * mask_ratio
    x, mask, ids_restore = random_mask(self.make_rng('dropout'), x, self.mask_ratio)
    ids_restore = jnp.reshape(ids_restore, [n, h, w])  # carries the shape info

    if use_cls_token:
      cls = self.encoder_layers['cls_token']
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    # apply the encoder
    x = self.encoder_layers['blocks'](x, train=train)

    return x, mask, ids_restore

  def apply_unshuffle(self, x, ids_restore):
    """bottleneck projection, unshuffle, add pos emb"""
    use_cls_token = (self.classifier == 'token')

    n, h, w = ids_restore.shape
    ids_restore = jnp.reshape(ids_restore, [n, h * w])

    # apply the encoder-decoder bottleneck
    x = self.decoder_layers['bottleneck'](x)
    x_part = x  # take a copy

    # append mask token
    num_clstokens = 1 if use_cls_token else 0
    mask_token = self.decoder_layers['mask_token']
    mask_tokens = jnp.tile(mask_token, [n, ids_restore.shape[1] + num_clstokens - x.shape[1], 1])
    x_ = jnp.concatenate([x[:, num_clstokens:, :], mask_tokens], axis=1)  # no cls token
    x_ = gather_by_einsum(x_, ids_restore)

    # add decoder posembed (before cls token)
    x_ = self.decoder_layers['pos_emb'](x_)

    x_full = jnp.concatenate([x[:, :num_clstokens, :], x_], axis=1)  # append cls token
    return x_full, x_part

  def apply_decoder(self, x, train):

    # apply the decoder
    x = self.decoder_layers['blocks'](x, train=train)

    # apply the predictor
    x = self.decoder_layers['pred'](x)

    # use_cls_token = (self.classifier == 'token')
    # num_clstokens = 1 if use_cls_token else 0
    # # remove cls token
    # pred = x[:, num_clstokens:, :]

    return x  # including cls token

  def setup(self):
    """
    declare all param layers based on inputs
    """
    # ------------------------
    # define encoder
    # ------------------------
    use_cls_token = (self.classifier == 'token')
    assert use_cls_token  # kaiming: TODO: support both?

    encoder_layers = {}  # cannot directly declare self.encoder_layers
    encoder_layers['patch_emb'] = t5x.layers.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding',
        kernel_init=patch_kernel_init,
        bias_init=patch_bias_init,
        kernel_axes=('_null0', '_null1', '_null2', 'embed'))
    encoder_layers['pos_emb'] = Add2DPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, name='posembed_encoder')
    if use_cls_token:
      encoder_layers['cls_token'] = t5x.layers.param_with_axes('cls', clstoken_init, (1, 1, self.hidden_size), jnp.float32, axes=('_null0', '_null1', 'embed'))
    encoder_layers['blocks'] = Encoder(name='Transformer', **self.transformer, prefix='encoder')
    self.encoder_layers = encoder_layers

    # ------------------------
    # define decoder
    # ------------------------
    decoder_layers = {}
    decoder_layers['bottleneck'] = t5x.layers.Dense(
      features=self.decoder.hidden_size,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      kernel_axes=('mlp', 'embed'),  # 'mlp' is split first
      name='bottleneck')
    decoder_layers['mask_token'] = t5x.layers.param_with_axes(
      'mask_token', masktoken_init, (1, 1, self.decoder.hidden_size),
      jnp.float32, axes=('_null0', '_null1', 'embed'))
    decoder_layers['img_modal_token'] = t5x.layers.param_with_axes(
      'img_modal_token', masktoken_init, (1, 1, self.decoder.hidden_size),
      jnp.float32, axes=('_null0', '_null1', 'embed'))
    decoder_layers['pos_emb'] = Add2DPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, name='posembed_decoder')
    decoder_layers['blocks'] = Encoder(name='TransformerDecoder', **self.decoder.transformer, prefix='decoder')
    decoder_layers['pred'] = t5x.layers.Dense(
      features=self.patches.size[0] * self.patches.size[1] * 3 + self.vocab_size,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      kernel_axes=('embed', 'classes'),  # 'mlp' is split first
      name='pred')
    self.decoder_layers = decoder_layers

  
class ImageTextLearner(nn.Module):
  """ContrastiveLearner with Vision Transformer
  """
  config: Any = None  # model config
  dtype: Any = jnp.float32

  def get_config_img(self):
    cfg = self.config.model_img.copy_and_resolve_references()  # copy
    cfg.name = 'img_encoder'  # force name
    cfg.unlock()
    cfg.vocab_size = self.config.model_txt.vocab_size
    cfg.lock()
    return cfg

  def get_config_txt(self):
    cfg = self.config.model_txt.copy_and_resolve_references()  # copy
    cfg.name = 'txt_encoder'  # force name
    return cfg

  def setup(self):
    self.img_encoder = VisionTransformer(**self.get_config_img())
    self.txt_encoder = LanguageTransformer(**self.get_config_txt())

  def __call__(self, inputs, *, train):
    img = inputs['image']
    # txt = {'txt': inputs['txt'], 'txt_is_valid': inputs['txt_is_valid']}
    txt = inputs['txt']
    is_valid = inputs['txt_is_valid']


    # ------------------------
    # img (used to be self.img_encoder.__call__)
    # loss_img, vis = self.img_encoder(img, train=train)
    # ------------------------
    # ------------------------
    # txt
    # loss_txt = self.txt_encoder(txt, train=train)
    # ------------------------

    # apply both encoders
    x_img, mask_img, ids_restore_img = self.img_encoder.apply_encoder(img, train=train)
    x_txt, mask_txt, ids_restore_txt = self.txt_encoder.apply_encoder(txt, train=train)

    # apply both decoders
    x_img_full, x_img_part = self.img_encoder.apply_unshuffle(x_img, ids_restore_img)
    x_txt_full, x_txt_part = self.txt_encoder.apply_unshuffle(x_txt, ids_restore_txt)

    # merge the sequence
    x_img_full += self.img_encoder.decoder_layers['img_modal_token']
    x_txt_full += self.txt_encoder.decoder_layers['txt_modal_token']
    x_imgtxt_full = jnp.concatenate([x_img_full, x_txt_full], axis=1)

    # joint decoder
    pred_imgtxt = self.img_encoder.apply_decoder(x_imgtxt_full, train=train)

    # split the sequences
    pred_img, pred_txt = jnp.split(pred_imgtxt, (x_img_full.shape[1],), axis=1)
    vocab_size = self.txt_encoder.vocab_size
    pred_img = pred_img[:, :, :-vocab_size]  # first part of pred
    pred_txt = pred_txt[:, :, -vocab_size:]  # second part of pred

    pred_img = self.img_encoder.remove_cls_token(pred_img)

    # compute losses
    loss_img = self.img_encoder.compute_loss(img, pred_img, mask_img)
    loss_txt = self.txt_encoder.compute_loss(txt, pred_txt, mask_txt, is_valid)

    loss_tot = loss_img + loss_txt

    # visualize
    vis = self.img_encoder.visualization(img, pred_img, mask_img)

    artifacts = {
      'loss': loss_img,  # always plot loss_img in the 'loss' metric
      'loss_img': loss_img,
      'loss_txt': loss_txt}

    return loss_tot, vis, artifacts

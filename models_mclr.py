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


class AddPositionEmbs(nn.Module):
  """Adds (optionally learned) positional embeddings to the inputs."""
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
    """Applies AddPositionEmbs module."""
    # PE is always fixed in this case, directly excluded in the optimizer
    if self.use_cls_token:
      pe = self.pe[:, 1:, :]
    else:
      pe = self.pe

    output = inputs + pe

    return output


class MsaBlock(nn.Module):
  """Transformer MSA / feed-forward block."""
  num_heads: int
  qkv_features: int
  out_features: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.
  qkv_kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
  out_kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.zeros

  def setup(self):
    features = self.out_features
    qkv_features = self.qkv_features
    assert qkv_features % self.num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // self.num_heads
    projection = functools.partial(
        t5x.layers.DenseGeneral,
        axis=-1,
        features=(self.num_heads, head_dim),
        kernel_axes=('embed', 'joined_kv'),
        dtype=self.dtype
    )
    self.query_proj = projection(kernel_init=self.qkv_kernel_init, name='query')
    self.key_proj = projection(kernel_init=self.qkv_kernel_init, name='key')
    self.value_proj = projection(kernel_init=self.qkv_kernel_init, name='value')

    self.out_proj = t5x.layers.DenseGeneral(
        features=features,
        axis=(-2, -1),
        kernel_init=self.out_kernel_init,
        kernel_axes=('joined_kv', 'embed'),
        dtype=self.dtype,
        name='out')

  def __call__(self, inputs_q, inputs_kv, *, deterministic):
    query = self.query_proj(inputs_q)
    key = self.key_proj(inputs_kv)
    value = self.value_proj(inputs_kv)

    query = t5x.layers.with_sharding_constraint(query, ('batch', 'length', 'heads', 'kv'))
    key = t5x.layers.with_sharding_constraint(key, ('batch', 'length', 'heads', 'kv'))
    value = t5x.layers.with_sharding_constraint(value, ('batch', 'length', 'heads', 'kv'))

    dropout_rng = None
    if not deterministic and self.dropout_rate > 0.:
      dropout_rng = self.make_rng('dropout')

    # Apply attention.
    x = t5x.layers.dot_product_attention(
        query,
        key,
        value,
        bias=None,
        dropout_rng=dropout_rng,
        dropout_rate=self.dropout_rate,
        deterministic=deterministic,
        dtype=self.dtype,
        float32_logits=True)

    # Back to the original inputs dimensions.
    out = self.out_proj(x)
    return out


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
  rescale_out: float = 1.0

  def setup(self):
    self.dense_0 = t5x.layers.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        kernel_axes=('embed', 'mlp'),
        name='Dense_0',
    )
    self.dropout_0 = nn.Dropout(rate=self.dropout_rate)
    self.dense_1 = t5x.layers.Dense(
        features=self.out_dim,
        dtype=self.dtype,
        kernel_init=lambda *args: self.kernel_init(*args) * self.rescale_out,
        bias_init=self.bias_init,
        kernel_axes=('mlp', 'embed'),
        name='Dense_1',
    )
    self.dropout_1 = nn.Dropout(rate=self.dropout_rate)

  def __call__(self, inputs, *, deterministic):
    """Applies Transformer MlpBlock module."""
    x = self.dense_0(inputs)
    x = nn.gelu(x)
    x = self.dropout_0(x, deterministic=deterministic)
    x = t5x.layers.with_sharding_constraint(x, ('batch', 'length', 'mlp'))
    output = self.dense_1(x)
    output = self.dropout_1(output, deterministic=deterministic)
    return output


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer."""
  hidden_size: int
  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  droppath_rate: float = 0.0
  layer_id: int = None
  rescale_out: float = 1.0
  res_attn: bool = True
  res_mlp: bool = True

  def setup(self):
    self.ln_0 = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))
    self.self_attention = MsaBlock(
        dtype=self.dtype,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads,
        qkv_features=self.hidden_size,
        out_features=self.hidden_size,
        qkv_kernel_init=qkv_kernel_init,
        out_kernel_init=lambda *args: out_kernel_init(*args) * self.rescale_out,
    )
    self.dropout_0 = nn.Dropout(rate=self.dropout_rate)
    self.droppath_0 = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name='droppath_msa')

    self.ln_1 = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))
    self.mlp = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, out_dim=self.hidden_size, dropout_rate=self.dropout_rate,
        kernel_init=mlp_kernel_init,
        bias_init=mlp_bias_init,
        rescale_out=self.rescale_out,
    )
    self.droppath_1 = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name='droppath_mlp')

  def __call__(self, inputs, deterministic):
    """Applies Encoder1DBlock module."""

    # Attention block.
    assert inputs.ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
    x = self.ln_0(inputs)
    x = self.self_attention(x, x, deterministic=deterministic)
    x = self.dropout_0(x, deterministic=deterministic)
    x = self.droppath_0(x, deterministic=deterministic)
    if self.res_attn:
      x = x + inputs

    # MLP block.
    y = self.ln_1(x)
    y = self.mlp(y, deterministic=deterministic)
    y = self.droppath_1(y, deterministic=deterministic)
    if self.res_mlp:
      y = x + y
    return y


class Encoder(nn.Module):
  """Transformer Model Encoder for sequence to sequence translation."""

  num_layers: int
  hidden_size: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  droppath_rate: float = 0.0
  prefix: str = 'encoder'
  rescale_out: float = 1.0
  res_attn: bool = True
  res_mlp: bool = True
  remat_policy: str = 'none'

  def setup(self):
    # this should be the activation check-pointing trigger
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

    blocks = []
    for lyr in range(self.num_layers):
      blocks.append(BlockLayer(
          hidden_size=self.hidden_size,
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1) if self.droppath_rate > 0. else 0.,
          name=self.prefix + 'block_{:02d}'.format(lyr),
          num_heads=self.num_heads,
          layer_id=lyr,
          rescale_out=self.rescale_out,
          res_attn=self.res_attn,
          res_mlp=self.res_mlp,
      ))
    self.blocks = blocks

    self.ln = t5x.layers.LayerNorm(name=self.prefix + '_norm', axes=('embed',))

  def __call__(self, inputs, *, train):
    """Applies Transformer model on the inputs."""
    assert inputs.ndim == 3  # (batch, len, emb)

    x = inputs
    deterministic = not train
    for block in self.blocks:
      x = block(x, deterministic)
    encoded = self.ln(x)

    return encoded


class CrossDecoder1DBlock(nn.Module):
  """Transformer decoder layer."""
  hidden_size: int
  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  layer_id: int = None
  rescale_out: float = 1.0

  def setup(self):
    self.ln_0 = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))
    self.self_attention = MsaBlock(
        dtype=self.dtype,
        dropout_rate=0.,
        num_heads=self.num_heads,
        qkv_features=self.hidden_size,
        out_features=self.hidden_size,
        qkv_kernel_init=qkv_kernel_init,
        out_kernel_init=lambda *args: out_kernel_init(*args) * self.rescale_out,
    )

    self.ln_1 = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))
    self.cross_attention = MsaBlock(
        dtype=self.dtype,
        dropout_rate=0.,
        num_heads=self.num_heads,
        qkv_features=self.hidden_size,
        out_features=self.hidden_size,
        qkv_kernel_init=qkv_kernel_init,
        out_kernel_init=lambda *args: out_kernel_init(*args) * self.rescale_out,
    )

    self.ln_2 = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))
    self.mlp = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        out_dim=self.hidden_size,
        dropout_rate=0.,
        kernel_init=mlp_kernel_init,
        bias_init=mlp_bias_init,
        rescale_out=self.rescale_out,
    )

  def __call__(self, inputs, queries, deterministic):
    """Applies decoder module."""

    # Self attention.
    q = self.ln_0(queries)
    q = self.self_attention(q, q, deterministic=deterministic)
    q = q + queries

    # Cross attention.
    x = self.ln_1(q)
    x = self.cross_attention(x, inputs, deterministic=deterministic)
    x = x + q

    # MLP block.
    y = self.ln_2(x)
    y = self.mlp(y, deterministic=deterministic)

    return y + x


class ClassDecoder1DBlock(nn.Module):
  """Transformer decoder layer from CaiT."""
  hidden_size: int
  mlp_dim: int
  num_heads: int
  dtype: Dtype = jnp.float32
  layer_id: int = None
  rescale_out: float = 1.0

  def setup(self):
    self.ln_0 = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))
    self.self_attention = MsaBlock(
        dtype=self.dtype,
        dropout_rate=0.,
        num_heads=self.num_heads,
        qkv_features=self.hidden_size,
        out_features=self.hidden_size,
        qkv_kernel_init=qkv_kernel_init,
        out_kernel_init=lambda *args: out_kernel_init(*args) * self.rescale_out,
    )

    self.ln_1 = t5x.layers.LayerNorm(dtype=self.dtype, axes=('embed',))
    self.mlp = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        out_dim=self.hidden_size,
        dropout_rate=0.,
        kernel_init=mlp_kernel_init,
        bias_init=mlp_bias_init,
        rescale_out=self.rescale_out,
    )

  def __call__(self, inputs, queries, deterministic):
    """Applies decoder module."""

    # Self attention.
    q = self.ln_0(queries)
    x = jnp.concatenate([q, inputs], axis=1)
    t = self.self_attention(q, x, deterministic=deterministic)
    t = t + queries

    # MLP block.
    y = self.ln_1(t)
    y = self.mlp(y, deterministic=deterministic)

    return y + t


class Decoder(nn.Module):
  """Transformer Model Decoder for sequence to sequence translation."""

  decoder_type: str
  hidden_size: int
  mlp_dim: int
  num_heads: int
  num_layers: int = 2
  prefix: str = 'decoder'
  rescale_out: float = 1.0
  remat_policy: str = 'none'

  def setup(self):
    # projection layer first
    self.neck = t5x.layers.Dense(
      features=self.hidden_size,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      kernel_axes=('mlp', 'embed'),  # 'mlp' is split first
      name='bottleneck'
    )
    self.neck_ln = t5x.layers.LayerNorm(name=self.prefix + '_neck_norm', axes=('embed',))

    if self.decoder_type == 'cross':
      BlockLayer = CrossDecoder1DBlock
    elif self.decoder_type == 'class':
      BlockLayer = ClassDecoder1DBlock
    else:
      raise NotImplementedError

    if self.remat_policy not in (None, 'none'):
      if self.remat_policy == 'minimal':
        policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
      else:
        policy = None
      BlockLayer = remat(
          BlockLayer,
          prevent_cse=True,
          policy=policy,
          static_argnums=(1,))

    blocks = []
    for lyr in range(self.num_layers):
      blocks.append(BlockLayer(
          hidden_size=self.hidden_size,
          mlp_dim=self.mlp_dim,
          name=self.prefix + 'block_{:02d}'.format(lyr),
          num_heads=self.num_heads,
          layer_id=lyr,
          rescale_out=self.rescale_out,
      ))
    self.blocks = blocks

    self.ln = t5x.layers.LayerNorm(name=self.prefix + '_norm', axes=('embed',))

  def __call__(self, inputs, queries, *, train):
    """Applies Transformer model on the inputs."""
    assert inputs.ndim == 3
    # match dimensions
    inputs = self.neck_ln(self.neck(inputs))

    q = queries
    deterministic = not train
    for block in self.blocks:
      q = block(inputs, q, deterministic)
    decoded = self.ln(q)

    return decoded


# the implementation for pjit
def gather_by_einsum(x, ids):
  """kaiming: vmap + gather is slow with pjit; use einsum instead"""
  mat = jax.nn.one_hot(ids, x.shape[1])  # [N, K, L]
  x = jnp.einsum('nl...,nkl->nk...', x, mat)
  return x


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
  num_queries: int
  decoder_type: str
  decoder_dim: int
  num_decoder_heads: int
  num_decoder_layer: int
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

    self.decoder_query = t5x.layers.param_with_axes('decoder_query',
      clstoken_init,
      (1, self.num_queries, self.decoder_dim),
      self.dtype,
      axes=('_null0', '_null1', 'embed')
    )

    self.decoder = Decoder(name='DecoderTransformer',
                          decoder_type=self.decoder_type,
                          hidden_size=self.decoder_dim,
                          mlp_dim=self.decoder_dim * 4,
                          num_heads=self.num_decoder_heads,
                          num_layers=self.num_decoder_layer,
                          prefix='decoder')

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
          bias_init=mlp_bias_init,
          kernel_axes=kernel_axes,
          name='proj{}'.format(i),
      ))
      if post_act:
        projector.append(t5x.layers.LayerNorm(
          dtype=self.dtype,
          axes=(kernel_axes[-1],),
          name='proj_ln{}'.format(i),
        ))
        projector.append(nn.gelu)

      flip = not flip

    self.projector = nn.Sequential(projector)

  def random_mask(self, x, mask_ratio, num_crops):
    if mask_ratio > 0.:
      N, L, C = x.shape  # batch, length, dim

      len_keep = int(L * (1. - mask_ratio)) * num_crops

      rng = self.make_rng('dropout')
      noise = random.uniform(rng, shape=x.shape[:2])

      ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove

      # keep the first subset
      ids_keep = ids_shuffle[:, :len_keep]
      x = gather_by_einsum(x, ids_keep)

      # multi-crop
      x = jnp.reshape(x, [N * num_crops, len_keep // num_crops, C])

    x = t5x.layers.with_sharding_constraint(x, ('batch', 'length', 'embed'))

    return x

  def __call__(self, inputs, train, mask_ratio=0., num_crops=1):
    x = self.conv_0(inputs)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    x = self.pos_embed(x)

    # masking: length -> length * mask_ratio
    x = self.random_mask(x, mask_ratio, num_crops)

    # apply the encoder
    x = self.encoder(x, train=train)

    # append class token
    q = jnp.tile(self.decoder_query, [n * num_crops, 1, 1])

    # cross attention with class token
    q = self.decoder(x, q, train=train)

    # apply projector
    p = self.projector(q)

    return x, p


class SiameseLearner(nn.Module):
  """SiameseLearner."""

  encoder: Any
  image_size: int
  mask_ratio: float
  num_crops: int
  temp: float
  pred_layers: int
  pred_dim_hidden: int
  loss_type: str = 'cos'
  intra_weight: float = 1.0
  visualize: bool = False
  knn: Any = None
  dtype: Any = jnp.float32

  def setup(self):
    # source
    self.encoder.name = 'Source' # hack to change name
    self.source_encoder = VisionTransformer(image_size=self.image_size, **self.encoder)

    # assert the number is the same
    assert (1. - self.mask_ratio) * self.num_crops <= 1.

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
          bias_init=mlp_bias_init,
          kernel_axes=kernel_axes,
          name='pred{}'.format(i),
      ))
      if post_act:
        predictor.append(t5x.layers.LayerNorm(
          dtype=self.dtype,
          axes=(kernel_axes[-1],),
          name='pred_ln{}'.format(i),
        ))
        predictor.append(nn.gelu)

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

    # the dimensions are made more clear
    source = jnp.reshape(source, [source.shape[0] // self.num_crops, self.num_crops, self.encoder.num_queries, -1])

    if self.loss_type == 'cos':
      return self.cosine(source, target)
    elif self.loss_type == 'info-nce':
      return self.info_nce(source, target)
    else:
      raise NotImplementedError

  def cosine(self, source, target):
    logits = jnp.einsum('nvqc,nqc->nvq', source, target)
    return -logits.mean()

  def info_nce(self, source, target):
    # batch size
    N = source.shape[0]

    # inter-image contrast
    logits_inter = jnp.einsum('nvqc,mqc->nvqm', source, target) / self.temp
    labels_inter = jnp.tile(jnp.arange(N, dtype=jnp.int32)[:, None, None], (1, self.num_crops, self.encoder.num_queries))
    xent_inter = optax.softmax_cross_entropy_with_integer_labels(logits=logits_inter, labels=labels_inter)

    # intra-image contrast
    logits_intra = jnp.einsum('nvqc,npc->nvqp', source, target) / self.temp
    labels_intra = jnp.tile(jnp.arange(self.encoder.num_queries, dtype=jnp.int32)[None, None, :], (N, self.num_crops, 1))
    xent_intra = optax.softmax_cross_entropy_with_integer_labels(logits=logits_intra, labels=labels_intra)

    loss = xent_inter.mean() + xent_intra.mean() * self.intra_weight

    return loss * self.temp

  def __call__(self, inputs, *, train, update=True):
    imgs = inputs['image']
    labels = inputs['label']

    # the augmentations are shared, just w/ or w/o masking
    _, p0 = self.source_encoder(imgs, train, self.mask_ratio, self.num_crops)
    x1, p1 = self.target_encoder(imgs, train)

    # optionally apply knn
    knn_accuracy = self.apply_knn(x1, labels, train=(train and update))

    # predictor
    p0 = self.predictor(p0)

    # compute loss
    loss = self.compute_loss(p0, p1)

    if self.visualize:
      raise NotImplementedError
    else:
      outcome = None

    return loss, outcome, knn_accuracy

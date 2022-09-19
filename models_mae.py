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
from uritemplate import partial


from utils import posembed_util
from utils import initializers_util
from utils import attention_util
from utils import dist_util
from utils.onlineknn_util import OnlineKNN

from utils import attention_mask_util

Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# init hacks
INIT_VER = 'mae_jax_v2'

fixed_gaussian_init = nn.initializers.normal(stddev=0.02)
if INIT_VER == 'vit_v1':  # JAX ViT
  clstoken_init = nn.initializers.zeros
  masktoken_init = fixed_gaussian_init
  posemb_init = fixed_gaussian_init
  patch_kernel_init = nn.initializers.lecun_uniform()
  patch_bias_init = nn.initializers.zeros
  msa_kernel_init = nn.initializers.xavier_uniform()
  mlp_kernel_init = nn.initializers.xavier_uniform()
  mlp_bias_init = nn.initializers.normal(stddev=1e-6)
elif INIT_VER == 'vit_v2':  # PyTorch ViT, used for debugging
  clstoken_init = fixed_gaussian_init
  masktoken_init = fixed_gaussian_init
  posemb_init = fixed_gaussian_init
  patch_kernel_init = fixed_gaussian_init
  patch_bias_init = fixed_gaussian_init  # bug from PyTorch code?
  msa_kernel_init = fixed_gaussian_init
  mlp_kernel_init = fixed_gaussian_init
  mlp_bias_init = nn.initializers.zeros
elif INIT_VER == 'mae_jax_v2':  # like PyTorch/TF ViT, with some differences
  clstoken_init = fixed_gaussian_init
  masktoken_init = fixed_gaussian_init
  posemb_init = fixed_gaussian_init  # not used if sincos
  # patch_kernel_init = nn.initializers.xavier_uniform()  # known to be different: were like nn.Linear in TF
  patch_kernel_init = initializers_util.patch_kernel()
  patch_bias_init = nn.initializers.zeros  # different from PyTorch?

  # msa_kernel_init = nn.initializers.xavier_uniform()  # known to be different: q, k, v are separated kernels in JAX

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

  Attributes:
    posemb_init: positional embedding initializer.
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
      self.pe = self.param('pos_embedding', posemb_init, pos_emb_shape)
    else:
      pe_array = posembed_util.get_2d_sincos_pos_embed(c, (h, w), cls_token=self.use_cls_token)  # in numpy array

      sincos_init = initializers_util.constant(value=pe_array, dtype=self.dtype)
      self.pe = self.param('pos_embedding', sincos_init, pos_emb_shape)

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


class FixedShuffler(nn.Module):
  """ Perform fixed shuffle
  """
  length: int

  def setup(self):
    rng = self.make_rng('dropout')
    noise = random.uniform(rng, shape=(self.length,))

    ids_shuffle = jnp.argsort(noise, axis=0)
    ids_restore = jnp.argsort(ids_shuffle, axis=0)

    self.ids_shuffle = self.variable('ids', 'ids_shuffle',
      lambda s: ids_shuffle, ids_shuffle.shape)

    self.ids_restore = self.variable('ids', 'ids_restore',
      lambda s: ids_restore, ids_restore.shape)
    
  def __call__(self, inputs):
    outputs = inputs[:, self.ids_shuffle.value, :]
    return outputs

  def restore(self, inputs):
    outputs = inputs[:, self.ids_restore.value, :]
    return outputs


class FarthestShuffler(nn.Module):
  """ Perform fixed shuffle
  """
  length: int

  def setup(self):
    assert self.length == 196  # hack
    ids_shuffle = [  0, 195,  13, 182,  90, 110, 175,   6,  84,  45,  51, 129, 135,  69,
          186,   3,   9,  42,  48,  87,  93, 126, 132, 152, 192,  25,  81, 155,
          159,  41,  53, 157, 163, 184,  15,  18,  21,  30,  33,  36,  38,  57,
            60,  63,  66,  72,  75,  78,  97,  99, 102, 105, 108, 114, 117, 120,
          123, 125, 142, 144, 147, 150, 165, 167, 180, 188, 190,   1,   2,   4,
            5,   7,   8,  10,  11,  12,  14,  16,  17,  19,  20,  22,  23,  24,
            26,  27,  28,  29,  31,  32,  34,  35,  37,  39,  40,  43,  44,  46,
            47,  49,  50,  52,  54,  55,  56,  58,  59,  61,  62,  64,  65,  67,
            68,  70,  71,  73,  74,  76,  77,  79,  80,  82,  83,  85,  86,  88,
            89,  91,  92,  94,  95,  96,  98, 100, 101, 103, 104, 106, 107, 109,
          111, 112, 113, 115, 116, 118, 119, 121, 122, 124, 127, 128, 130, 131,
          133, 134, 136, 137, 138, 139, 140, 141, 143, 145, 146, 148, 149, 151,
          153, 154, 156, 158, 160, 161, 162, 164, 166, 168, 169, 170, 171, 172,
          173, 174, 176, 177, 178, 179, 181, 183, 185, 187, 189, 191, 193, 194]
    ids_shuffle = jnp.array(ids_shuffle)

    ids_restore = jnp.argsort(ids_shuffle, axis=0)

    self.ids_shuffle = self.variable('ids', 'ids_shuffle',
      lambda s: ids_shuffle, ids_shuffle.shape)

    self.ids_restore = self.variable('ids', 'ids_restore',
      lambda s: ids_restore, ids_restore.shape)
    
  def __call__(self, inputs):
    outputs = inputs[:, self.ids_shuffle.value, :]
    return outputs

  def restore(self, inputs):
    outputs = inputs[:, self.ids_restore.value, :]
    return outputs


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
    x = nn.Dense(
        features=self.mlp_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            inputs)
    x = nn.gelu(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        features=actual_out_dim,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init)(  # pytype: disable=wrong-arg-types
            x)
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
  torch_qkv: bool = False
  mask: Any = None

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
    x = nn.LayerNorm(dtype=self.dtype)(inputs)

    # ----------------------------------------------------
    if self.torch_qkv:
      raise NotImplementedError
      # revised, QKV
      MsaBlock = functools.partial(
        attention_util.MultiHeadDotProductAttentionQKV,
        out_kernel_init=out_kernel_init)
    else:
      # revised
      MsaBlock = functools.partial(
        attention_util.MultiHeadDotProductAttention,
        qkv_kernel_init=qkv_kernel_init,
        out_kernel_init=out_kernel_init)

    # original
    # MsaBlock = functools.partial(
    #   nn.MultiHeadDotProductAttention,
    #   kernel_init=msa_kernel_init,)
    # ----------------------------------------------------

    msablock = MsaBlock(
        dtype=self.dtype,
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)
    x = msablock(x, x, mask=self.mask)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    # droppath
    x = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name='droppath_msa')(x, deterministic=deterministic)
    x = x + inputs

    # MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim, dtype=self.dtype, dropout_rate=self.dropout_rate,
        kernel_init=mlp_kernel_init,
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
  torch_qkv: bool = False
  sequentialize: str = ''

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

    if self.sequentialize == 'row':
      mask = attention_mask_util.get_row_mask(inputs)
    elif self.sequentialize == 'p2x':
      mask = attention_mask_util.get_p2x_mask(inputs)
    else:  # raster
      mask = attention_mask_util.get_causal_mask(inputs)

    x = inputs
    # Input Encoder
    for lyr in range(self.num_layers):
      block = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1) if self.droppath_rate > 0. else 0.,
          name=self.prefix + 'block_{:02d}'.format(lyr),  # 'encoderblock_'
          num_heads=self.num_heads,
          layer_id=lyr,
          torch_qkv=self.torch_qkv,
          mask=mask)
      x = block(x, deterministic=not train)
    encoded = nn.LayerNorm(name=self.prefix + '_norm')(x)  # 'encoder_norm'
    return encoded


def gather(x, ids):
  return x[ids]
vmapped_gather = jax.jit(jax.vmap(gather, in_axes=(0, 0), out_axes=0))


class VisionTransformer(nn.Module):
  """VisionTransformer."""

  num_classes: int
  mask_ratio: float
  sincos: bool
  norm_pix_loss: bool
  patches: Any
  transformer: Any
  hidden_size: int
  representation_size: Optional[int] = None
  classifier: str = 'token'
  dtype: Any = jnp.float32
  decoder: Any = None
  visualize: bool = False
  knn: Any = None
  num_ohem: int = 0
  pred_offset: int = 0
  sequentialize: str = 'raster'
  pred_outside: bool = False
  use_start_token: bool = False
  use_decoder_pos: bool = False

  def random_mask(self, x):
    
    N, L, _ = x.shape  # batch, length, dim
    len_keep = int(L * (1 - self.mask_ratio))

    rng = self.make_rng('dropout')
    noise = random.uniform(rng, shape=(N, L))

    ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
    ids_restore = jnp.argsort(ids_shuffle, axis=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]    
    x_masked = vmapped_gather(x, ids_keep)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = jnp.ones([N, L])
    mask = mask.at[:, :len_keep].set(0)
    # unshuffle to get the binary mask
    mask = vmapped_gather(mask, ids_restore)

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

  def compute_loss(self, pred, target):
    """
    pred: [N, L-1, p*p*3]
    target: [N, L-1, p*p*3]
    """
    # target = self.patchify(imgs)
    if self.norm_pix_loss:
      # target = jax.nn.normalize(target, axis=-1, epsilon=1.e-6)
      mean = jnp.mean(target, axis=-1, keepdims=True)
      var = jnp.var(target, axis=-1, keepdims=True)
      target = (target - mean) / (var + 1.e-6)**.5

    # if self.pred_offset > 0:
    #   target = target[:, self.pred_offset:, :] # remove the head
    #   pred = pred[:, :-self.pred_offset, :]  # remove the tail

    loss = jnp.square(pred - target)

    if self.num_ohem > 0:
      # remove the last ones
      loss = jnp.mean(loss, axis=-1)  # average per patch
      loss = jax.lax.top_k(loss, self.num_ohem)[0]
      loss = jnp.mean(loss, axis=-1)  # [N, L], mean loss per patch
    else:
      loss = jnp.mean(loss, axis=-1)  # [N, L], mean loss per patch

    # loss = jnp.sum(loss * mask) / jnp.sum(mask)  # mean loss on removed patches
    loss = jnp.mean(loss)
    return loss

  def visualization(self, imgs, pred, target, shuffler):
    """
    imgs: [N, H, W, 3]
    pred: [N, L-1, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove, 
    """
    assert target.shape == pred.shape

    # if self.sequentialize == 'raster':
    #   pred = jnp.pad(pred, ((0, 0), (1, 0), (0, 0)))  # pad zero at the beginning of L
    #   target = jnp.pad(target, ((0, 0), (1, 0), (0, 0)))  # pad zero at the beginning of L
    # else:
    #   raise NotImplementedError

    if shuffler is not None:
      pred = shuffler.restore(pred)
      target = shuffler.restore(target)

    imgs_pred = self.unpatchify(pred)
    imgs_target = self.unpatchify(target)

    imgs_vis = jnp.concatenate([imgs, imgs_pred, imgs_target], axis=2)
    return imgs_vis

  def apply_reorder(self, x, target):
    """ random one of four raster orders """
    assert x.shape[:2] == target.shape[:2]

    rng = self.make_rng('dropout')

    id = jax.random.randint(rng, shape=(), minval=0, maxval=4)  # maxval: exclusive

    xt = jnp.concatenate([x, target], axis=-1)

    N, L, C = xt.shape
    H = W = int(L**.5)
    assert H * W == L

    # yt = jnp.einsum('nhwc->nwhc', xt.reshape([N, H, W, C])).reshape([N, L, C])  # transpose

    xt_new = jnp.where((id <= 1),
      jnp.where((id == 0), xt, xt[:, ::-1, :]),
      jnp.where((id == 2),
        jnp.einsum('nhwc->nwhc', xt.reshape([N, H, W, C])).reshape([N, L, C]),
        jnp.einsum('nhwc->nwhc', xt.reshape([N, H, W, C])).reshape([N, L, C])[:, ::-1, :]),
    )

    x, target = jnp.split(xt_new, (x.shape[-1],), axis=-1)
    return x, target

  def prepare_inputs(self, inputs):

    if not self.pred_outside:
      x = inputs
      target = inputs
      return x, target

    p = self.patches.size[0]
    offset = self.pred_offset + 1
    poff = offset * p

    n, h, w, c = inputs.shape

    if self.sequentialize == 'raster':
      x = inputs[:, :-poff, :-poff, :]  # remove last of w-axis
      target = inputs[:, :-poff, poff:, :]  # remove first of w-axis

    else:
      raise NotImplementedError

    assert x.shape == target.shape

    return x, target

  def apply_encoder(self, inputs, train):
    use_cls_token=(self.classifier == 'token')
    assert not use_cls_token  # kaiming: TODO: support both?

    img_inputs, target = self.prepare_inputs(inputs)
      
    target = self.patchify(target)

    x = img_inputs

    # We can merge s2d+emb into a single conv; it's the same.
    x = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding',
        kernel_init=patch_kernel_init,
        bias_init=patch_bias_init,
        )(x)

    # Here, x is a grid of embeddings.

    # Transformer.
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    x = AddPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, img_shape=(h, w, c), name='posembed_encoder')(x)

    shuffler = None
    if self.sequentialize == 'raster':
      pass
    elif self.sequentialize == 'row':
      pass
    elif self.sequentialize == 'p2x':
      pass
    # elif self.sequentialize == 'fixed_shuffle':
    #   # apply fixed shuffle
    #   shuffler = FixedShuffler(length=h * w)
    #   x = shuffler(x)
    #   target = shuffler(target)
    # elif self.sequentialize == 'reorder':
    #   # random one of four raster orders
    #   x, target = self.apply_reorder(x, target)
    # elif self.sequentialize == 'farthest':
    #   shuffler = FarthestShuffler(length=h * w)
    #   x = shuffler(x)
    #   target = shuffler(target)
    else:
      raise NotImplementedError

    # row-order: we don't shift here
    # shift by one
    # x_encode = x[:, :-1, :] # remove the last one
    # target = target[:, 1:, :]  # remove the first one

    if self.use_start_token:
      raise NotImplementedError
      # use start_token
      start_token = self.param('start_token', masktoken_init, (1, 1, c))
      start_token = jnp.tile(start_token, [n, 1, 1])
      x_encode = jnp.concatenate([start_token, x_encode], axis=1)

    # apply the encoder
    x_encode = Encoder(name='Transformer', **self.transformer, prefix='encoder', sequentialize=self.sequentialize)(x, train=train)

    return x_encode, target, img_inputs, shuffler

  def apply_decoder(self, x, train):
    use_cls_token=(self.classifier == 'token')
    assert not use_cls_token  # kaiming: TODO: support both?

    # apply the encoder-decoder bottleneck
    x = nn.Dense(
      features=self.decoder.hidden_size,
      dtype=self.dtype,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      name='bottleneck')(x)    

    # append mask token
    # num_clstokens = 1 if use_cls_token else 0
    # mask_token = self.param('mask_token', masktoken_init, (1, 1, self.decoder.hidden_size))
    # mask_tokens = jnp.tile(mask_token, [n, ids_restore.shape[1] + num_clstokens - x.shape[1], 1])
    # x_ = jnp.concatenate([x[:, num_clstokens:, :], mask_tokens], axis=1)  # no cls token
    # x_ = vmapped_gather(x_, ids_restore)

    if self.use_decoder_pos:
      raise NotImplementedError
      assert not self.use_start_token
      N, L, C = x.shape
      L = x.shape[1] + 1
      h = w = int(L**.5)
      assert L == h * w
      zeros = jnp.zeros(shape=[N, L, C])

      posemb = AddPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, img_shape=(h, w, self.decoder.hidden_size), name='posembed_decoder')(zeros)

      # use the input position?
      posemb = posemb[:, :-1, :]
      x += posemb

    # add decoder posembed (before cls token)
    # x_ = AddPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, img_shape=(h, w, self.decoder.hidden_size), name='posembed_decoder')(x_)

    # x = jnp.concatenate([x[:, :num_clstokens, :], x_], axis=1)  # append cls token

    # apply the decoder
    x = Encoder(name='TransformerDecoder', **self.decoder.transformer, prefix='decoder', sequentialize=self.sequentialize)(x, train=train)

    # apply the predictor
    x = nn.Dense(
      features=self.patches.size[0] * self.patches.size[1] * 3,
      dtype=self.dtype,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      name='pred')(x)

    # remove cls token
    # pred = x[:, num_clstokens:, :]

    if self.use_start_token:
      x = x[:, 1:, :]  # remove the output at the start token

    pred = x
    return pred

  def apply_knn(self, x, labels, train):
    if not self.knn.on:
      return
    if self.knn.postprocess in ('tgap', 'gap'):
      x = jnp.mean(x, axis=1)
    else:
      raise NotImplementedError

    if self.knn.postnorm == 'LayerNorm':
      x = nn.LayerNorm(use_bias=False, use_scale=False, name='knn_postnorm')(x)
    elif self.knn.postnorm == 'SyncBatchNorm':  # no gamma/beta
      x = dist_util.SyncBatchNorm(x, eps=1.e-6)
    else:
      raise NotImplementedError

    if self.knn.l2norm:
      l2norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1.e-12)
      x /= l2norm

    knn_accuracy = OnlineKNN(knn=self.knn)(x, labels, train=train)
    return knn_accuracy

  def shift_pred_and_target(self, pred, target):
    assert target.shape == pred.shape

    offset = self.pred_offset + 1  # by default, offset = 1

    n, L, c = pred.shape
    h = w = int(L**.5)
    assert h * w == L  # no cls token for now

    if self.sequentialize == 'row':
      pred = pred.reshape([n, h, w, c])
      target = target.reshape([n, h, w, c])

      pred = pred[:, :, :-offset, :] # remove the last one along w
      target = target[:, :, offset:, :] # remove the first one along w

      pred_vis = jnp.pad(pred, ((0, 0), (0, 0), (offset, 0), (0, 0)))  # pad zero at the beginning of L
      target_vis = jnp.pad(target, ((0, 0), (0, 0), (offset, 0), (0, 0)))  # pad zero at the beginning of L

    elif self.sequentialize == 'p2x':
      assert self.pred_offset == 0
      pred = pred.reshape([n, h, w, c])
      target = target.reshape([n, h, w, c])

      # pred = pred[:, :-2, :-2, :]
      # target = target[:, 2:, 2:, :]

      pred = pred[:, :, :-2, :]
      target = target[:, :, 2:, :]

      pred_vis = jnp.pad(pred, ((0, 0), (0, 0), (2, 0), (0, 0)))  # pad zero at the beginning of L
      target_vis = jnp.pad(target, ((0, 0), (0, 0), (2, 0), (0, 0)))  # pad zero at the beginning of L

    elif self.sequentialize == 'raster':
      # shift by one
      pred = pred[:, :-offset, :] # remove the last one
      target = target[:, offset:, :]  # remove the first one

      pred_vis = jnp.pad(pred, ((0, 0), (offset, 0), (0, 0)))  # pad zero at the beginning of L
      target_vis = jnp.pad(target, ((0, 0), (offset, 0), (0, 0)))  # pad zero at the beginning of L
    else:
      raise NotImplementedError

    pred = pred.reshape([n, -1, c])
    target = target.reshape([n, -1, c])
    pred_vis = pred_vis.reshape([n, -1, c])
    target_vis = target_vis.reshape([n, -1, c])

    return pred, target, pred_vis, target_vis

  @nn.compact
  def __call__(self, inputs, *, train):
    imgs = inputs['image']
    labels = inputs['label']

    # apply encoder
    x, target, img_inputs, shuffler = self.apply_encoder(imgs, train=train)

    # optionally apply knn
    knn_accuracy = self.apply_knn(x, labels, train=train)

    # apply decoder
    pred = self.apply_decoder(x, train=train)

    # shift pred and target
    if not self.pred_outside:
      pred, target, pred_vis, target_vis = self.shift_pred_and_target(pred, target)
    else:
      pred_vis, target_vis = pred, target, 

    # compute loss
    loss = self.compute_loss(pred, target)

    if self.visualize and not train:
      outcome = self.visualization(img_inputs, pred_vis, target_vis, shuffler)
    else:
      outcome = pred  # not used

    return loss, outcome, knn_accuracy
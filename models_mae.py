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

    x = MsaBlock(
        dtype=self.dtype,
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate,
        num_heads=self.num_heads)(
            x, x)
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
    # Input Encoder
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1) if self.droppath_rate > 0. else 0.,
          name=self.prefix + 'block_{:02d}'.format(lyr),  # 'encoderblock_'
          num_heads=self.num_heads,
          layer_id=lyr,
          torch_qkv=self.torch_qkv)(
              x, deterministic=not train)
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
    use_cls_token=(self.classifier == 'token')
    assert use_cls_token  # kaiming: TODO: support both?

    x = inputs

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

    # masking: length -> length * mask_ratio
    x, mask, ids_restore = self.random_mask(x)
    ids_restore = jnp.reshape(ids_restore, [n, h, w])  # carries the shape info

    # If we want to add a class token, add it here.
    if use_cls_token:
      cls = self.param('cls', clstoken_init, (1, 1, c))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    # apply the encoder
    x = Encoder(name='Transformer', **self.transformer, prefix='encoder')(x, train=train)

    return x, mask, ids_restore

  def apply_decoder(self, x, ids_restore, train):
    use_cls_token=(self.classifier == 'token')

    n, h, w = ids_restore.shape
    ids_restore = jnp.reshape(ids_restore, [n, h * w])

    # apply the encoder-decoder bottleneck
    x = nn.Dense(
      features=self.decoder.hidden_size,
      dtype=self.dtype,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      name='bottleneck')(x)    

    # append mask token
    num_clstokens = 1 if use_cls_token else 0
    mask_token = self.param('mask_token', masktoken_init, (1, 1, self.decoder.hidden_size))
    mask_tokens = jnp.tile(mask_token, [n, ids_restore.shape[1] + num_clstokens - x.shape[1], 1])
    x_ = jnp.concatenate([x[:, num_clstokens:, :], mask_tokens], axis=1)  # no cls token
    x_ = vmapped_gather(x_, ids_restore)

    # add decoder posembed (before cls token)
    x_ = AddPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, img_shape=(h, w, self.decoder.hidden_size), name='posembed_decoder')(x_)

    x = jnp.concatenate([x[:, :num_clstokens, :], x_], axis=1)  # append cls token

    # apply the decoder
    x = Encoder(name='TransformerDecoder', **self.decoder.transformer, prefix='decoder')(x, train=train)

    # apply the predictor
    x = nn.Dense(
      features=self.patches.size[0] * self.patches.size[1] * 3,
      dtype=self.dtype,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      name='pred')(x)

    # remove cls token
    pred = x[:, num_clstokens:, :]
    return pred

  def apply_knn(self, x, labels, train):
    if not self.knn.on:
      return
    if self.knn.postprocess == 'tgap':
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


  @nn.compact
  def __call__(self, inputs, *, train):
    imgs = inputs['image']
    labels = inputs['label']

    # apply encoder
    x, mask, ids_restore = self.apply_encoder(imgs, train=train)

    # optionally apply knn
    knn_accuracy = self.apply_knn(x, labels, train=train)

    # apply decoder
    pred = self.apply_decoder(x, ids_restore, train=train)

    # compute loss
    loss = self.compute_loss(imgs, pred, mask)

    if self.visualize and not train:
      outcome = self.visualization(imgs, pred, mask)
    else:
      outcome = pred  # not used

    return loss, outcome, knn_accuracy
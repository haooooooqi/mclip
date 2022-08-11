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

import jax
import jax.numpy as jnp
import jax.random as random

import flax.linen as nn
import optax


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
  start_idx: int = 0
  torch_qkv: bool = False

  @nn.compact
  def __call__(self, inputs, *, train, num_layers=None):
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
    if num_layers is None:
      num_layers = self.num_layers
    for lyr in range(num_layers):
      name = self.prefix + 'block_{:02d}'.format(lyr + self.start_idx)
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1) if self.droppath_rate > 0. else 0.,
          name=name,  # 'encoderblock_'
          num_heads=self.num_heads,
          layer_id=lyr,
          torch_qkv=self.torch_qkv)(
              x, deterministic=not train)
      logging.info('Block: {}/{}'.format(self.name, name))
    if num_layers == self.num_layers:
      name = self.prefix + '_norm'
      X = nn.LayerNorm(name=name)(x)  # 'encoder_norm'
      logging.info('Block: {}/{}'.format(self.name, name))

    return x


class VisionTransformer(nn.Module):
  """VisionTransformer."""
  sincos: bool
  patches: Any
  transformer: Any
  hidden_size: int
  classifier: str = 'token'
  dtype: Any = jnp.float32
  clr: Any = None

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

  def apply_encoder(self, inputs, train):
    use_cls_token=(self.classifier == 'token')
    assert use_cls_token  # kaiming: TODO: support both?

    x = inputs

    # We can merge s2d+emb into a single conv; it's the same.
    patch_embed = nn.Conv(
        features=self.hidden_size,
        kernel_size=self.patches.size,
        strides=self.patches.size,
        padding='VALID',
        name='embedding',
        kernel_init=patch_kernel_init,
        bias_init=patch_bias_init,
        )
    x = patch_embed(x)

    # Here, x is a grid of embeddings.

    # Transformer.
    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    posembed = AddPositionEmbs(sincos=self.sincos, use_cls_token=use_cls_token, img_shape=(h, w, c), name='posembed_encoder')
    x = posembed(x)

    # If we want to add a class token, add it here.
    if use_cls_token:
      clstoken = self.param('cls', clstoken_init, (1, 1, c))
      cls = jnp.tile(clstoken, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)
    else:
      clstoken = None

    # apply the encoder
    blocks = Encoder(name='Transformer', **self.transformer, prefix='encoder')
    x = blocks(x, train=train)

    return x

  @nn.compact
  def __call__(self, imgs, *, train):
    """
    imgs: [2*N, H, W, 3]
    return:
    z: [2*N, C]
    x: [2*N, L, D]
    """
    # apply encoder
    x_enc = self.apply_encoder(imgs, train=train)

    # reduce the feature. TODO: add config here
    x_proj = x_enc.mean(axis=1)  # [2*N, C]

    # apply proj head
    x_proj = self.apply_projection_head(x_proj)  # [2*N, C]

    return x_proj, x_enc

  def apply_projection_head(self, z):
    for i in range(self.clr.proj_layers - 1):
      z = nn.Dense(
        features=self.clr.proj_dim_hidden,
        dtype=self.dtype,
        kernel_init=mlp_kernel_init,
        bias_init=mlp_bias_init,
        name='mlp_pred{}'.format(i))(z)
      z = nn.gelu(z)

    z = nn.Dense(
      features=self.clr.proj_dim_out,
      dtype=self.dtype,
      kernel_init=mlp_kernel_init,
      bias_init=mlp_bias_init,
      name='mlp_pred{}'.format(self.clr.proj_layers))(z)

    return z

  
class ContrastiveLearner(nn.Module):
  """ContrastiveLearner with Vision Transformer
  """
  config: Any = None  # model config
  dtype: Any = jnp.float32

  def get_base_config(self):
    cfg = self.config.copy_and_resolve_references()  # copy
    cfg.name = 'base_encoder'
    
    # delete unused fields
    cfg.unlock()
    del cfg.decoder
    del cfg.knn
    del cfg.visualize
    cfg.lock()
    return cfg

  def get_auxi_config(self):
    cfg = self.config.copy_and_resolve_references()  # copy
    cfg.name = 'auxi_encoder'
    
    # delete unused fields
    cfg.unlock()
    cfg.update(cfg.decoder)  # replace with the decoder cfg
    del cfg.decoder
    del cfg.knn
    del cfg.visualize
    cfg.lock()
    return cfg

  @nn.compact
  def __call__(self, inputs, *, train):
    """
    inputs['image']: [N, V, H, W, 3], V for viewss
    """

    imgs = inputs['image']
    labels = inputs['label']

    assert len(imgs.shape) == 5 and imgs.shape[1] == 2

    # split the images
    imgs0 = imgs[:, 0, :, :, :]
    imgs1 = imgs[:, 1, :, :, :]
    imgs = jnp.concatenate([imgs0, imgs1], axis=0)

    # define the encoder
    base_encoder = VisionTransformer(**self.get_base_config())
    auxi_encoder = VisionTransformer(**self.get_auxi_config())

    # run the encoders
    x_base, x_enc = base_encoder(imgs, train=train)  # [2*N, C]
    x_auxi, _ = auxi_encoder(imgs, train=train)  # [2*N, C]

    # apply knn on x
    knn_accuracy = self.apply_knn(jnp.split(x_enc, 2, axis=0)[0], labels, train=train)

    # compute the loss
    loss = self.compute_auxi_contrastive_loss(x_base, x_auxi)

    if self.config.visualize and not train:
      vis = self.visualization(imgs)
    else:
      vis = None  # not used

    artifacts = {'knn_accuracy': knn_accuracy}

    return loss, vis, artifacts

  def compute_auxi_contrastive_loss(self, x_base, x_auxi):
    """
    x_base: [2*N, C] of 2 views (source)
    x_auxi: [2*N, C] of 2 views (target)
    """
    x_base /= jnp.linalg.norm(x_base, axis=1, keepdims=True) + 1e-8
    x_auxi /= jnp.linalg.norm(x_auxi, axis=1, keepdims=True) + 1e-8


    x_base0, x_base1 = jnp.split(x_base, 2, axis=0)
    x_auxi0, x_auxi1 = jnp.split(x_auxi, 2, axis=0)

    loss01 = self.compute_asymmetric_contrastive_loss(x_base0, x_auxi1)
    loss10 = self.compute_asymmetric_contrastive_loss(x_base1, x_auxi0)
    loss = (loss01 + loss10) / 2
    return loss

  def compute_asymmetric_contrastive_loss(self, z_src, z_tgt):
    z0 = z_src
    z1 = z_tgt

    # for simplicity we gather both for now
    z0_all = dist_util.all_gather(z0, axis_name='batch')
    z1_all = dist_util.all_gather(z1, axis_name='batch')

    z0 = z0_all.reshape([-1, z0.shape[-1]])
    z1 = z1_all.reshape([-1, z1.shape[-1]])

    tau = self.config.clr.tau

    logits = jnp.einsum('nc,mc->nm', z0, z1)
    logits /= tau
    labels_one_hot = jnp.eye(logits.shape[0])

    # asymmetric loss for simclr
    loss = optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot)  # over the last axis
    loss = loss.mean()
    loss *= 2 * tau
    return loss

  def compute_symmetric_contrastive_loss(self, z):
    z /= jnp.linalg.norm(z, axis=1, keepdims=True) + 1e-8

    z0, z1 = jnp.split(z, 2, axis=0)

    z0_all = dist_util.all_gather(z0, axis_name='batch')
    z1_all = dist_util.all_gather(z1, axis_name='batch')

    z0 = z0_all.reshape([-1, z0.shape[-1]])
    z1 = z1_all.reshape([-1, z1.shape[-1]])

    tau = self.config.clr.tau

    logits = jnp.einsum('nc,mc->nm', z0, z1)
    logits /= tau
    labels_one_hot = jnp.eye(logits.shape[0])

    # symmetric loss for simclr
    loss01 = optax.softmax_cross_entropy(logits=logits, labels=labels_one_hot)
    loss10 = optax.softmax_cross_entropy(logits=logits.transpose(), labels=labels_one_hot)
    loss = (loss01 + loss10) / 2
    loss = loss.mean()
    loss *= 2 * tau
    return loss

  def apply_knn(self, x, labels, train):
    if not self.config.knn.on:
      return
    if self.config.knn.postprocess == 'tgap':
      x = jnp.mean(x, axis=1)
    else:
      raise NotImplementedError

    if self.config.knn.postnorm == 'LayerNorm':
      x = nn.LayerNorm(use_bias=False, use_scale=False, name='knn_postnorm')(x)
    elif self.config.knn.postnorm == 'SyncBatchNorm':  # no gamma/beta
      x = dist_util.SyncBatchNorm(x, eps=1.e-6)
    else:
      raise NotImplementedError

    if self.config.knn.l2norm:
      l2norm = jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True) + 1.e-12)
      x /= l2norm

    knn_accuracy = OnlineKNN(knn=self.config.knn)(x, labels, train=train)
    return knn_accuracy

  def visualization(self, imgs):
    """
    imgs: [2*N, H, W, 3]
    """
    imgs0, imgs1 = jnp.split(imgs, 2, axis=0)
    imgs_vis = jnp.concatenate([imgs0, imgs1], axis=2)
    return imgs_vis

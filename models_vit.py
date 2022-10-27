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
from flax.linen.partitioning import remat
import jax
import jax.numpy as jnp
import jax.random as random

import t5x.layers


Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

# init hacks
# v1: JAX ViT; v2: PyTorch ViT; v3: v2 with fix
INIT_VER = "v2"

fixed_gaussian_init = nn.initializers.normal(stddev=0.02)
if INIT_VER == "v1":
    clstoken_init = nn.initializers.zeros
    posemb_init = fixed_gaussian_init
    patch_kernel_init = nn.initializers.lecun_uniform()
    patch_bias_init = nn.initializers.zeros
    msa_kernel_init = nn.initializers.xavier_uniform()
    mlp_kernel_init = nn.initializers.xavier_uniform()
    mlp_bias_init = nn.initializers.normal(stddev=1e-6)
    head_kernel_init = nn.initializers.zeros
elif INIT_VER == "v2":
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


class QuickGELU(nn.Module):
    @nn.compact
    def __call__(self, x):
        return x * nn.sigmoid(1.702 * x)


class AddPositionEmbs(nn.Module):
    """Adds (optionally learned) positional embeddings to the inputs.

    Attributes:
      posemb_init: positional embedding initializer.
    """

    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

    @nn.compact
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
        # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs.ndim == 3, "Number of dimensions should be 3," " but it is: %d" % inputs.ndim
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pe = t5x.layers.param_with_axes(
            "pos_embedding", self.posemb_init, pos_emb_shape, jnp.float32, axes=("_null0", "length", "embed")
        )
        return inputs + pe


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    mlp_dim: int
    dtype: Dtype = jnp.float32
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.normal(stddev=1e-6)
    quick_gelu: bool = False

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """Applies Transformer MlpBlock module."""
        actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
        x = t5x.layers.Dense(
            features=self.mlp_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            kernel_axes=("embed", "mlp"),
            name="Dense_0",
        )(inputs)
        # print(self.quick_gelu)
        x = QuickGELU()(x) if self.quick_gelu else nn.gelu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        x = t5x.layers.with_sharding_constraint(x, ("batch", "length", "mlp"))
        output = t5x.layers.Dense(
            features=actual_out_dim,
            dtype=self.dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            kernel_axes=("mlp", "embed"),
            name="Dense_1",
        )(x)
        output = nn.Dropout(rate=self.dropout_rate)(output, deterministic=deterministic)
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
    quick_gelu: bool = False
    ln_eps: float = 1e-6

    @nn.compact
    def __call__(self, inputs, deterministic, att_mask=None):
        """Applies Encoder1DBlock module.

        Args:
          inputs: Inputs to the layer.
          deterministic: Dropout will not be applied when set to true.

        Returns:
          output after transformer encoder block.
        """

        # Attention block.
        assert inputs.ndim == 3, f"Expected (batch, seq, hidden) got {inputs.shape}"
        x = t5x.layers.LayerNorm(dtype=self.dtype, epsilon=self.ln_eps, axes=("embed",))(inputs)

        # ----------------------------------------------------
        # t5x
        MsaBlock = functools.partial(
            t5x.layers.MultiHeadDotProductAttention, qkv_kernel_init=msa_kernel_init, out_kernel_init=msa_kernel_init,
        )
        # original
        # MsaBlock = functools.partial(
        #   nn.MultiHeadDotProductAttention,
        #   kernel_init=msa_kernel_init,
        #   broadcast_dropout=False,
        #   deterministic=deterministic,
        # )
        # ----------------------------------------------------

        x = MsaBlock(dtype=self.dtype, dropout_rate=self.attention_dropout_rate, num_heads=self.num_heads,)(x, x, mask=att_mask)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
        # droppath
        x = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name="droppath_msa")(
            x, deterministic=deterministic
        )
        x = x + inputs

        # MLP block.
        y = t5x.layers.LayerNorm(dtype=self.dtype, epsilon=self.ln_eps, axes=("embed",))(x)
        y = MlpBlock(
            mlp_dim=self.mlp_dim,
            dtype=self.dtype,
            dropout_rate=self.dropout_rate,
            kernel_init=mlp_kernel_init,
            bias_init=mlp_bias_init,
            quick_gelu=self.quick_gelu,
        )(y, deterministic=deterministic)
        # droppath
        y = nn.Dropout(rate=self.droppath_rate, broadcast_dims=(1, 2), name="droppath_mlp")(
            y, deterministic=deterministic
        )

        return x + y


class Projection(nn.Module):
  """Transformer MLP / feed-forward block."""

  proj_dim_hidden: int
  proj_layers: int
  proj_dim_out: int
  prefix: str
  dtype: Dtype = jnp.float32
  kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
  bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.zeros

  @nn.compact
  def __call__(self, z):
    """Applies Transformer MlpBlock module."""
    for i in range(self.proj_layers - 1):
      z = t5x.layers.Dense(
        features=self.proj_dim_hidden,
        dtype=self.dtype,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        kernel_axes=('_null0', '_null1'),
        name='{}_mlp{}'.format(self.prefix, i))(z)
      z = nn.gelu(z)
    z = t5x.layers.Dense(
      features=self.proj_dim_out,
      dtype=self.dtype,
      kernel_init=self.kernel_init,
      bias_init=self.bias_init,
      kernel_axes=('_null0', '_null1'),
      name='{}_mlp{}'.format(self.prefix, self.proj_layers))(z)
    return z


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
    remat_policy: str = "none"
    use_encoder_prenorm: bool = False
    quick_gelu: bool = False
    ln_eps: float = 1e-6
    use_encoder_proj: bool = False
    proj_dim: int = 512
    use_encoder_proj_bias: bool = False

    @nn.compact
    def __call__(self, inputs, *, train, encoder_norm=True, att_mask=None):
        """Applies Transformer model on the inputs.

        Args:
          inputs: Inputs to the layer.
          train: Set to `True` when training.

        Returns:
          output of a transformer encoder.
        """
        assert inputs.ndim == 3  # (batch, len, emb)

        BlockLayer = Encoder1DBlock
        if self.remat_policy not in (None, "none"):
            if self.remat_policy == "minimal":
                policy = jax.checkpoint_policies.checkpoint_dots_with_no_batch_dims
            else:
                policy = None
            BlockLayer = remat(  # pylint: disable=invalid-name
                Encoder1DBlock, prevent_cse=True, policy=policy, static_argnums=(1,)
            )  # "deterministic" is a static argument in Encoder1DBlock

        x = inputs
        # x = AddPositionEmbs(
        #     posemb_init=posemb_init,  # from BERT.
        #     name='posembed_input')(
        #         inputs)
        # x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

        if self.use_encoder_prenorm:
            x = t5x.layers.LayerNorm(name="encoder_prenorm", epsilon=self.ln_eps, axes=("embed",))(x) # add an encoder norm as in CLIP

        # Input Encoder
        for lyr in range(self.num_layers):
            deterministic = not train
            x = BlockLayer(
                mlp_dim=self.mlp_dim,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                droppath_rate=self.droppath_rate * lyr / (self.num_layers - 1),
                name="encoderblock_{:02d}".format(lyr),
                num_heads=self.num_heads,
                layer_id=lyr,
                quick_gelu=self.quick_gelu,
                ln_eps=self.ln_eps,
            )(x, deterministic, att_mask=att_mask)
        encoded = t5x.layers.LayerNorm(name="encoder_norm", epsilon=self.ln_eps, axes=("embed",))(x) if encoder_norm else x
        if self.use_encoder_proj:
          # print(encoded.shape, self.proj_dim)
          encoded = t5x.layers.Dense(
            features=self.proj_dim,
            use_bias=self.use_encoder_proj_bias,
            kernel_init=mlp_kernel_init,
            # kernel_axes=("embed", "proj"),
            kernel_axes=("embed", "_null0"),
            name="encoder_proj",
          )(encoded)
          # print(encoded.shape)

        return encoded


class VisionTransformer(nn.Module):
    """VisionTransformer."""

    num_classes: int
    patches: Any
    transformer: Any
    hidden_size: int
    representation_size: Optional[int] = None
    classifier: str = "token"
    dtype: Any = jnp.float32
    rescale_head_init: float = 1.0

    @nn.compact
    def __call__(self, inputs, *, train):
        x = inputs

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
            padding="VALID",
            name="embedding",
            kernel_init=patch_kernel_init,
            bias_init=patch_bias_init,
            kernel_axes=("_null0", "_null1", "_null2", "embed"),
        )(x)

        # Here, x is a grid of embeddings.

        # Transformer.
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # If we want to add a class token, add it here.
        if self.classifier in {"token", "tgap"}:
            cls = t5x.layers.param_with_axes(
                "cls", clstoken_init, (1, 1, c), jnp.float32, axes=("_null0", "_null1", "embed")
            )
            cls = jnp.tile(cls, [n, 1, 1])
            x = jnp.concatenate([cls, x], axis=1)

        # we add posemb here
        x = AddPositionEmbs(posemb_init=posemb_init, name="posembed_encoder")(x)

        x = Encoder(name="Transformer", **self.transformer)(x, train=train, encoder_norm=(self.classifier == "token"))

        if self.classifier == "token":
            x = x[:, 0]
        elif self.classifier == "tgap":
            x = x[:, 1:]
            x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
            # x = nn.LayerNorm(name='fc_norm')(x)
            x = t5x.layers.LayerNorm(name="fc_norm", axes=("embed",))(x)
        elif self.classifier == "gap":
            x = jnp.mean(x, axis=list(range(1, x.ndim - 1)))  # (1,) or (1,2)
            # x = nn.LayerNorm(name='fc_norm')(x)
            x = t5x.layers.LayerNorm(name="fc_norm", axes=("embed",))(x)
        else:
            raise ValueError(f"Invalid classifier={self.classifier}")

        if self.representation_size is not None:
            raise NotImplementedError
            # x = nn.Dense(features=self.representation_size, name='pre_logits')(x)
            # x = nn.tanh(x)
        else:
            x = IdentityLayer(name="pre_logits")(x)

        # ------------------------------------------------
        # debugging BN or state
        # x = nn.BatchNorm(
        #   use_running_average=not train,
        #   momentum=0.9,
        #   epsilon=1e-5,
        #   name='bn_debug'
        # )(x)
        # var_bias = t5x.layers.variable_with_axes(
        #   'debug_vars', 'var_bias',
        #   lambda s: jnp.zeros(s, jnp.float32),
        #   (x.shape[-1],),
        #   axes=('embed',))
        # x += var_bias.value
        # if train:
        #   var_bias.value += 1.
        # ------------------------------------------------

        if self.num_classes:
            # x = nn.Dense(
            #   features=self.num_classes,
            #   name='head',
            #   kernel_init=head_kernel_init
            # )(x)
            x = t5x.layers.Dense(
                features=self.num_classes,
                kernel_init=lambda *args: head_kernel_init(*args) * self.rescale_head_init,
                kernel_axes=("embed", "classes"),
                name="head",
            )(x)

        return x


class ClipVisualBackbone(nn.Module):
    """VisionTransformer."""

    patches: Any
    transformer: Any
    hidden_size: int
    representation_size: Optional[int] = None
    classifier: str = "token"
    dtype: Any = jnp.float32
    rescale_head_init: float = 1.0
    encoder_norm: bool = False

    @nn.compact
    def __call__(self, inputs, *, train):
        x = inputs

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
            padding="VALID",
            name="embedding",
            kernel_init=patch_kernel_init,
            bias_init=patch_bias_init,
            kernel_axes=("_null0", "_null1", "_null2", "embed"),
        )(x)

        # Here, x is a grid of embeddings.

        # Transformer.
        n, h, w, c = x.shape
        x = jnp.reshape(x, [n, h * w, c])

        # If we want to add a class token, add it here.
        if self.classifier in {"token", "tgap"}:
            cls = t5x.layers.param_with_axes(
                "cls", clstoken_init, (1, 1, c), jnp.float32, axes=("_null0", "_null1", "embed")
            )
            cls = jnp.tile(cls, [n, 1, 1])
            x = jnp.concatenate([cls, x], axis=1)

        # we add posemb here
        x = AddPositionEmbs(posemb_init=posemb_init, name="posembed_encoder")(x)

        x = Encoder(name="Transformer", **self.transformer)(
          x, train=train, encoder_norm=self.encoder_norm or (self.classifier == "token"),
        )

        return x


class ClipTextualBackbone(nn.Module):
    """CLIP Text Transformer."""

    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    quick_gelu: bool = True  # use True for OpenAI-pretrained CLIP, and False for LAION-pretrained CLIP
    mlp_ratio: float = 4.0
    proj_dim: int = 512

    @nn.compact
    def __call__(self, txt_inds, *, train):
        # txt_inds = jnp.squeeze(txt_inds, axis=1)
        x = t5x.layers.Embed(
            num_embeddings=self.vocab_size,  # CLIP text vocab size
            features=self.width,
            one_hot=True,
            name="token_embedding",
        )(txt_inds)

        logging.info("------")
        logging.info(txt_inds)
        logging.info(x)
        logging.info(x.shape)
        logging.info("------")

        n, seq_length, c = x.shape

        arr_inds = jnp.arange(seq_length)
        att_mask = arr_inds[..., None] >= arr_inds[None, ...]  # row inds >= col inds for lower triangular

        # we add posemb here
        x = AddPositionEmbs(posemb_init=posemb_init, name="posembed_encoder")(x)

        x = Encoder(
            name="Transformer",
            num_layers=self.layers,
            mlp_dim=int(self.width * self.mlp_ratio),
            num_heads=self.heads,
            dropout_rate=0,
            attention_dropout_rate=0,
            droppath_rate=0,
            use_encoder_prenorm=False,
            quick_gelu=self.quick_gelu,
            ln_eps=1e-5,
            use_encoder_proj=True,
            proj_dim=self.proj_dim,
        )(x, train=train, encoder_norm=True, att_mask=att_mask)

        # Extract the CLIP text feature from the EOS token
        # (the CLIP tokenizer has EOS being the largest token)
        eos_inds = jnp.argmax(txt_inds, axis=1)
        out_txt_feat = x[jnp.arange(x.shape[0]), eos_inds]

        return out_txt_feat

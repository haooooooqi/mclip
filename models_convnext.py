# Copyright 2022 The Flax Authors.
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

"""Flax implementation of ResNet V1."""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Optional, Callable, Sequence, Tuple, Union, Iterable


import jax
from flax import linen as nn
import jax.numpy as jnp
import jax.random as random

ModuleDef = Any

class DepthwiseConv2D(nn.Module):
    kernel_shape: Union[int, Sequence[int]] = (1, 1)
    stride: Union[int, Sequence[int]] = (1, 1)
    padding: str or Sequence[Tuple[int, int]] = "SAME"
    channel_multiplier: int = 1
    use_bias: bool = True
    weights_init: Callable = nn.initializers.lecun_uniform()
    bias_init: Optional[Callable] = nn.initializers.zeros

    @nn.compact
    def __call__(self, input):
        w = self.param(
            "kernel",
            self.weights_init,
            self.kernel_shape + (1, self.channel_multiplier * input.shape[-1]),
        )
        if self.use_bias:
            b = self.param(
                "bias", self.bias_init, (self.channel_multiplier * input.shape[-1],)
            )

        conv = jax.lax.conv_general_dilated(
            input,
            w,
            self.stride,
            self.padding,
            (1,) * len(self.kernel_shape),
            (1,) * len(self.kernel_shape),
            ("NHWC", "HWIO", "NHWC"),
            input.shape[-1],
        )
        if self.use_bias:
            bias = jnp.broadcast_to(b, conv.shape)
            return conv + bias
        else:
            return conv

# initializer = nn.initializers.variance_scaling(
#     0.02, "fan_in", distribution="truncated_normal"
# )
fixed_gaussian_init = nn.initializers.normal(stddev=0.02)

initializer = fixed_gaussian_init

class ConvNeXtBlock(nn.Module):
    dim: int = 256
    layer_scale_init_value: float = 1e-6
    drop_path: float = 0.1
    deterministic: Optional[bool] = None

    def init_fn(self, key, shape, fill_value):
        return jnp.full(shape, fill_value)

    @nn.compact
    def __call__(self, inputs, deterministic=None):
        x = DepthwiseConv2D((7, 7), weights_init=initializer, name="dwconv")(inputs)
        x = nn.LayerNorm(name="norm")(x)
        x = nn.Dense(4 * self.dim, kernel_init=initializer, name="pwconv1")(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim, kernel_init=initializer, name="pwconv2")(x)
        if self.layer_scale_init_value > 0:
            gamma = self.param(
                "gamma", self.init_fn, (self.dim,), self.layer_scale_init_value
            )
            x = gamma * x
        
        x = nn.Dropout(rate=self.drop_path, broadcast_dims=(1, 2), name='droppath_block')(x, deterministic=deterministic)
        x = x + inputs
        # x = inputs + DropPath(self.drop_path)(x, deterministic)
        return x


class ConvNeXt(nn.Module):
    """
    ConvNeXt Module
    Attributes:
        depths (list or tuple): Depths for every block
        dims (list or tuple): Embedding dimension for every stage.
        drop_path (float): Dropout value for DropPath. Default is 0.1
        layer_scale_init_value (float): Initialization value for scale. Default is 1e-6.
        head_init_scale (float): Initialization value for head. Default is 1.0.
        attach_head (bool): Whether to attach classification head. Default is False.
        num_classes (int): Number of classification classes. Only works if attach_head is True. Default is 1000.
        deterministic (bool): Optional argument, if True, network becomes deterministic and dropout is not applied.
    """

    depths: Iterable = (3, 3, 9, 3)
    dims: Iterable = (96, 192, 384, 768)
    drop_path: float = 0.0
    layer_scale_init_value: float = 1e-6
    head_init_scale: float = 1.0
    attach_head: bool = True
    num_classes: int = 1000
    dtype: Any = jnp.float32
    deterministic: Optional[bool] = None

    @nn.compact
    def __call__(self, inputs, *, train):

        # deterministic = nn.merge_param(
        #     "deterministic", self.deterministic, deterministic
        # )
        deterministic = not train

        dp_rates = [x for x in jnp.linspace(0, self.drop_path, sum(self.depths))]
        curr = 0
        # self.droppath_rate * lyr / (self.num_layers - 1),
        # Stem
        x = nn.Conv(
            self.dims[0], (4, 4), 4, kernel_init=initializer, name="downsample_layers00"
        )(inputs)
        x = nn.LayerNorm(name="downsample_layers01")(x)

        for j in range(self.depths[0]):
            x = ConvNeXtBlock(
                self.dims[0],
                drop_path=self.drop_path * (curr+j) / (sum(self.depths) - 1),
                layer_scale_init_value=self.layer_scale_init_value,
                name=f"stages0{j}",
            )(x, deterministic)
        curr += self.depths[0]

        # Downsample layers
        for i in range(3):
            x = nn.LayerNorm(name=f"downsample_layers{i+1}0")(x)
            x = nn.Conv(
                self.dims[i + 1],
                (2, 2),
                2,
                kernel_init=initializer,
                name=f"downsample_layers{i+1}1",
            )(x)

            for j in range(self.depths[i + 1]):
                x = ConvNeXtBlock(
                    self.dims[i + 1],
                    drop_path=self.drop_path * (curr+j) / (sum(self.depths) - 1),
                    layer_scale_init_value=self.layer_scale_init_value,
                    name=f"stages{i+1}{j}",
                )(x, deterministic)

            curr += self.depths[i + 1]

        if self.attach_head:
            x = nn.LayerNorm(name="norm")(jnp.mean(x, [1, 2]))
            x = nn.Dense(self.num_classes, kernel_init=initializer, name="head")(x)
        return x

ConvNeXt_tiny = partial(ConvNeXt, depths=(3,3,9,3), dims=(96, 192, 384, 768), drop_path=0.1, layer_scale_init_value=1e-6)
ConvNeXt_base = partial(ConvNeXt, depths=(3,3,27,3), dims=(128, 256, 512, 1024), drop_path=0.2, layer_scale_init_value=1e-6)
ConvNeXt_large = partial(ConvNeXt, depths=(3,3,27,3), dims=(192, 384, 768, 1536), drop_path=0.5, layer_scale_init_value=1e-6)
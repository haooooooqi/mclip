# Copyright 2022 The T5X Authors.
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

"""Dense attention classes and mask/weighting functions."""

# pylint: disable=attribute-defined-outside-init,g-bare-generic

import dataclasses
import functools
import operator
from typing import Any, Callable, Iterable, Optional, Sequence, Tuple, Union

from flax import linen as nn
from flax.linen import partitioning as nn_partitioning
import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import numpy as np


param_with_axes = nn_partitioning.param_with_axes
variable_with_axes = nn_partitioning.variable_with_axes
with_sharding_constraint = nn_partitioning.with_sharding_constraint


# Type annotations
Array = jnp.ndarray
Axes = Union[int, Iterable[int]]
DType = jnp.dtype
PRNGKey = jnp.ndarray
Shape = Iterable[int]
Activation = Callable[..., Array]
PrecisionLike = Union[None, str, lax.Precision, Tuple[str, str],
                      Tuple[lax.Precision, lax.Precision]]
# Parameter initializers.
Initializer = Callable[[PRNGKey, Shape, DType], Array]

default_embed_init = nn.initializers.variance_scaling(
    1.0, 'fan_in', 'normal', out_axis=0)


def dot_product_attention(query: Array,
                          key: Array,
                          value: Array,
                          bias: Optional[Array] = None,
                          dropout_rng: Optional[PRNGKey] = None,
                          dropout_rate: float = 0.,
                          deterministic: bool = False,
                          dtype: DType = jnp.float32,
                          float32_logits: bool = False):
  """Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights.

  Args:
    query: queries for calculating attention with shape of `[batch, q_length,
      num_heads, qk_depth_per_head]`.
    key: keys for calculating attention with shape of `[batch, kv_length,
      num_heads, qk_depth_per_head]`.
    value: values to be used in attention with shape of `[batch, kv_length,
      num_heads, v_depth_per_head]`.
    bias: bias for the attention weights. This should be broadcastable to the
      shape `[batch, num_heads, q_length, kv_length]` This can be used for
      incorporating causal masks, padding masks, proximity bias, etc.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    dtype: the dtype of the computation (default: float32)
    float32_logits: bool, if True then compute logits in float32 to avoid
      numerical issues with bfloat16.

  Returns:
    Output of shape `[batch, length, num_heads, v_depth_per_head]`.
  """
  assert key.ndim == query.ndim == value.ndim, 'q, k, v must have same rank.'
  assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], (
      'q, k, v batch dims must match.')
  assert query.shape[-2] == key.shape[-2] == value.shape[-2], (
      'q, k, v num_heads must match.')
  assert key.shape[-3] == value.shape[-3], 'k, v lengths must match.'
  assert query.shape[-1] == key.shape[-1], 'q, k depths must match.'

  # Casting logits and softmax computation for float32 for model stability.
  if float32_logits:
    query = query.astype(jnp.float32)
    key = key.astype(jnp.float32)

  depth = query.shape[-1]
  query = query / jnp.sqrt(depth).astype(dtype)
  # `attn_weights`: [batch, num_heads, q_length, kv_length]
  attn_weights = jnp.einsum('bqhd,bkhd->bhqk', query, key)

  # Apply attention bias: masking, dropout, proximity bias, etc.
  if bias is not None:
    attn_weights = attn_weights + bias.astype(attn_weights.dtype)

  # Normalize the attention weights across `kv_length` dimension.
  attn_weights = jax.nn.softmax(attn_weights).astype(dtype)

  # Apply attention dropout.
  if not deterministic and dropout_rate > 0.:
    raise NotImplementedError
    keep_prob = 1.0 - dropout_rate
    # T5 broadcasts along the "length" dim, but unclear which one that
    # corresponds to in positional dimensions here, assuming query dim.
    dropout_shape = list(attn_weights.shape)
    dropout_shape[-2] = 1
    keep = random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    keep = jnp.broadcast_to(keep, attn_weights.shape)
    multiplier = (
        keep.astype(attn_weights.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    attn_weights = attn_weights * multiplier

  # Take the linear combination of `value`.
  return jnp.einsum('bhqk,bkhd->bqhd', attn_weights, value)


dynamic_vector_slice_in_dim = jax.vmap(
    lax.dynamic_slice_in_dim, in_axes=(None, 0, None, None))


def _normalize_axes(axes: Iterable[int], ndim: int) -> Tuple[int]:
  # A tuple by convention. len(axes_tuple) then also gives the rank efficiently.
  return tuple([ax if ax >= 0 else ndim + ax for ax in axes])


def _canonicalize_tuple(x):
  if isinstance(x, Iterable):
    return tuple(x)
  else:
    return (x,)


#------------------------------------------------------------------------------
# DenseGeneral for attention layers.
#------------------------------------------------------------------------------
class DenseGeneral(nn.Module):
  """A linear transformation with flexible axes.

    Attributes:
      features: tuple with numbers of output features.
      axis: tuple with axes to apply the transformation on.
      dtype: the dtype of the computation (default: float32).
      kernel_init: initializer function for the weight matrix.
  """
  features: Union[Iterable[int], int]
  use_bias: bool = True
  dtype: DType = jnp.float32
  kernel_init: Initializer = nn.initializers.variance_scaling(
      1.0, 'fan_in', 'truncated_normal')
  bias_init: Initializer = nn.initializers.zeros
  kernel_axes: Tuple[str, ...] = ()
  axis: Union[Iterable[int], int] = -1

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along multiple dimensions.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    features = _canonicalize_tuple(self.features)
    axis = _canonicalize_tuple(self.axis)

    inputs = jnp.asarray(inputs, self.dtype)
    axis = _normalize_axes(axis, inputs.ndim)

    kernel_shape = tuple([inputs.shape[ax] for ax in axis]) + features
    kernel_param_shape = (np.prod([inputs.shape[ax] for ax in axis]),
                          np.prod(features))
    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        kernel_param_shape,
        jnp.float32,
        axes=self.kernel_axes)
    kernel = jnp.asarray(kernel, self.dtype)
    kernel = jnp.reshape(kernel, kernel_shape)

    contract_ind = tuple(range(0, len(axis)))
    y = lax.dot_general(inputs, kernel, ((axis, contract_ind), ((), ())))

    if self.use_bias:
      bias = param_with_axes(
          'bias',
          self.bias_init,
          kernel_param_shape[-1],
          jnp.float32,
          axes=(self.kernel_axes[-1],))
      bias = jnp.asarray(bias, self.dtype)
      bias = jnp.reshape(bias, (1,) * (y.ndim - len(features)) + features)
      y += bias
    return y


class Dense(nn.Module):
  """A linear transformation applied over the last dimension of the input.

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: float32).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """
  features: int
  use_bias: bool = True
  dtype: DType = jnp.float32
  param_dtype: DType = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Callable[[PRNGKey, Shape, DType], Array] = nn.linear.default_kernel_init
  bias_init: Callable[[PRNGKey, Shape, DType], Array] = nn.initializers.zeros
  kernel_axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    inputs = jnp.asarray(inputs, self.dtype)
    kernel = param_with_axes(
        'kernel',
        self.kernel_init,
        (inputs.shape[-1], self.features),
        self.param_dtype,
        axes=self.kernel_axes)
    kernel = jnp.asarray(kernel, self.dtype)
    y = lax.dot_general(inputs, kernel,
                        (((inputs.ndim - 1,), (0,)), ((), ())),
                        precision=self.precision)
    if self.use_bias:
      bias = param_with_axes(
          'bias',
          self.bias_init,
          (self.features,),
          self.param_dtype,
          axes=(self.kernel_axes[-1],))
      bias = jnp.asarray(bias, self.dtype)
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y


#------------------------------------------------------------------------------
# Conv layers
#------------------------------------------------------------------------------
class Conv(nn.Conv):
  """Conv with axis names:
  Copied from flax.linen.linear, replace self.param
  """
  kernel_axes: Tuple[str, ...] = ()
  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a (potentially unshared) convolution to the inputs.

    Args:
      inputs: input data with dimensions (batch, spatial_dims..., features).
        This is the channels-last convention, i.e. NHWC for a 2d convolution
        and NDHWC for a 3D convolution. Note: this is different from the input
        convention used by `lax.conv_general_dilated`, which puts the spatial
        dimensions last.

    Returns:
      The convolved data.
    """

    inputs = jnp.asarray(inputs, self.dtype)

    if isinstance(self.kernel_size, int):
      raise TypeError('The kernel size must be specified as a'
                      ' tuple/list of integers (eg.: [3, 3]).')
    else:
      kernel_size = tuple(self.kernel_size)

    def maybe_broadcast(x: Optional[Union[int, Sequence[int]]]) -> (
        Tuple[int, ...]):
      if x is None:
        # backward compatibility with using None as sentinel for
        # broadcast 1
        x = 1
      if isinstance(x, int):
        return (x,) * len(kernel_size)
      return tuple(x)

    is_single_input = False
    if inputs.ndim == len(kernel_size) + 1:
      is_single_input = True
      inputs = jnp.expand_dims(inputs, axis=0)

    # self.strides or (1,) * (inputs.ndim - 2)
    strides = maybe_broadcast(self.strides)
    input_dilation = maybe_broadcast(self.input_dilation)
    kernel_dilation = maybe_broadcast(self.kernel_dilation)

    padding_lax: Union[str, Sequence[Tuple[int, int]]]
    if self.padding == 'CIRCULAR':
      kernel_size_dilated = [
          (k - 1) * d + 1 for k, d in zip(kernel_size, kernel_dilation)
      ]
      zero_pad: nn.linear.List[Tuple[int, int]] = [(0, 0)]
      pads = (zero_pad + [((k - 1) // 2, k // 2) for k in kernel_size_dilated] +
              [(0, 0)])
      inputs = jnp.pad(inputs, pads, mode='wrap')
      padding_lax = 'VALID'
    else:
      padding_lax = self.padding

    dimension_numbers = nn.linear._conv_dimension_numbers(inputs.shape)
    in_features = inputs.shape[-1]

    if self.shared_weights:
      # One shared convolutional kernel for all pixels in the output.
      assert in_features % self.feature_group_count == 0
      kernel_shape = kernel_size + (
          in_features // self.feature_group_count, self.features)

    else:
      if self.feature_group_count != 1:
        raise NotImplementedError(
            f'`lax.conv_general_dilated_local` does not support '
            f'`feature_group_count != 1`, got `{self.feature_group_count}`.'
        )

      # Need to know the spatial output shape of a standard convolution to
      # create the unshared convolution kernel.
      conv_output_shape = nn.linear.eval_shape(
          lambda lhs, rhs: lax.conv_general_dilated(  # pylint: disable=g-long-lambda
              lhs=lhs,
              rhs=rhs,
              window_strides=strides,
              padding=padding_lax,
              dimension_numbers=dimension_numbers
          ),
          inputs,
          nn.linear.ShapedArray(kernel_size + (in_features, self.features), inputs.dtype)
      ).shape

      # One (unshared) convolutional kernel per each pixel in the output.
      kernel_shape = conv_output_shape[1:-1] + (np.prod(kernel_size) *
                                                in_features, self.features)

    kernel = param_with_axes('kernel', self.kernel_init, kernel_shape,
                        self.param_dtype, axes=self.kernel_axes)
    # kernel = self.param('kernel', self.kernel_init, kernel_shape,
    #                     self.param_dtype)
    kernel = jnp.asarray(kernel, self.dtype)

    if self.shared_weights:
      y = lax.conv_general_dilated(
          inputs,
          kernel,
          strides,
          padding_lax,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
          dimension_numbers=dimension_numbers,
          feature_group_count=self.feature_group_count,
          precision=self.precision
      )
    else:
      y = lax.conv_general_dilated_local(
          lhs=inputs,
          rhs=kernel,
          window_strides=strides,
          padding=padding_lax,
          filter_shape=kernel_size,
          lhs_dilation=input_dilation,
          rhs_dilation=kernel_dilation,
          dimension_numbers=dimension_numbers,
          precision=self.precision
      )

    if self.use_bias:
      if self.shared_weights:
        # One bias weight per output channel, shared between pixels.
        bias_shape = (self.features,)
      else:
        # One bias weight per output entry, unshared betwen pixels.
        bias_shape = y.shape[1:]

      # bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
      bias = param_with_axes('bias', self.bias_init, bias_shape, self.param_dtype, axes=(self.kernel_axes[-1],))
      bias = jnp.asarray(bias, self.dtype)
      bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
      y += bias

    if is_single_input:
      y = jnp.squeeze(y, axis=0)
    return y


#------------------------------------------------------------------------------
# Normalization layers
#------------------------------------------------------------------------------
class LayerNorm(nn.Module):
  """Layer normalization with axis names."""
  epsilon: float = 1e-6
  dtype: Any = jnp.float32
  param_dtype: DType = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = nn.initializers.zeros
  scale_init: Initializer = nn.initializers.ones
  axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies layer normalization on the input."""
    reduction_axes = (-1,)
    feature_axes = (-1,)

    mean, var = nn.normalization._compute_stats(x, reduction_axes, None, None)

    return _normalize(
        self, x, mean, var, reduction_axes, feature_axes,
        self.dtype, self.param_dtype, self.epsilon,
        self.use_bias, self.use_scale,
        self.bias_init, self.scale_init,
        self.axes)


class TrainOnlyBatchNorm(nn.Module):
  """Batch normalization with axis names but without running averages (only useful during train)."""
  epsilon: float = 1e-5
  dtype: Any = jnp.float32
  param_dtype: DType = jnp.float32
  use_bias: bool = True
  use_scale: bool = True
  bias_init: Initializer = nn.initializers.zeros
  scale_init: Initializer = nn.initializers.ones
  axes: Tuple[str, ...] = ()

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    """Applies batch normalization on the input."""
    feature_axes = (x.ndim-1,)
    reduction_axes = tuple(i for i in range(x.ndim) if i not in feature_axes)

    mean, var = nn.normalization._compute_stats(x, reduction_axes, None, None)

    return _normalize(
        self, x, mean, var, reduction_axes, feature_axes,
        self.dtype, self.param_dtype, self.epsilon,
        self.use_bias, self.use_scale,
        self.bias_init, self.scale_init,
        self.axes)


def _normalize(mdl: nn.Module, x: Array, mean: Array, var: Array,
               reduction_axes: Axes, feature_axes: Axes,
               dtype: DType, param_dtype: DType,
               epsilon: float,
               use_bias: bool, use_scale: bool,
               bias_init: Callable[[PRNGKey, Shape, DType], Array],
               scale_init: Callable[[PRNGKey, Shape, DType], Array],
               axes: Tuple[str, ...] = ()):
  """"Normalizes the input of a normalization layer and optionally applies a learned scale and bias.

  Arguments:
    mdl: Module to apply the normalization in (normalization params will reside
      in this module).
    x: The input.
    mean: Mean to use for normalization.
    var: Variance to use for normalization.
    reduction_axes: The axes in ``x`` to reduce.
    feature_axes: Axes containing features. A separate bias and scale is learned
      for each specified feature.
    dtype: Dtype of the returned result.
    param_dtype: Dtype of the parameters.
    epsilon: Normalization epsilon.
    use_bias: If true, add a bias term to the output.
    use_scale: If true, scale the output.
    bias_init: Initialization function for the bias term.
    scale_init: Initialization function for the scaling function.

  Returns:
    The normalized input.
  """
  reduction_axes = nn.normalization._canonicalize_axes(x.ndim, reduction_axes)
  feature_axes = nn.normalization._canonicalize_axes(x.ndim, feature_axes)
  stats_shape = list(x.shape)
  for axis in reduction_axes:
    stats_shape[axis] = 1
  mean = mean.reshape(stats_shape)
  var = var.reshape(stats_shape)
  feature_shape = [1] * x.ndim
  reduced_feature_shape = []
  for ax in feature_axes:
    feature_shape[ax] = x.shape[ax]
    reduced_feature_shape.append(x.shape[ax])
  y = x - mean
  mul = lax.rsqrt(var + epsilon)
  if use_scale:
    scale = param_with_axes(
        'scale',
        scale_init,
        reduced_feature_shape,
        param_dtype,
        axes=axes).reshape(feature_shape)
    mul *= scale
  y *= mul
  if use_bias:
    bias = param_with_axes(
        'bias',
        bias_init,
        reduced_feature_shape,
        param_dtype,
        axes=axes).reshape(feature_shape)
    y += bias
  return jnp.asarray(y, dtype)
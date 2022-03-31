from typing import Tuple, Any

import functools
import tree as nest

import jax.numpy as jnp


# ---------------------------------------------------------
# rescale layers
# ---------------------------------------------------------
def layerwise_lr_decay(num_layers: int, lr_decay: float):
    """Get the lrd function."""
    return functools.partial(_compute_layerwise_lr_decay, num_layers=num_layers, lr_decay=lr_decay)


def _compute_layerwise_lr_decay(
        path: Tuple[Any], val: jnp.ndarray,
        num_layers: int, lr_decay: float):
    """Get the layerwise lr decay rate based on name."""
    del val

    layer_name = '.'.join(path)

    if layer_name.startswith('Transformer.encoderblock_'):
        layer_idx = path[1][len('encoderblock_'):]  # e.g., '01'
        layer_idx = int(layer_idx)
    elif layer_name.startswith('embedding.'):  # patch embedding
        layer_idx = 0
    elif layer_name.startswith('posembed_'):  # position embedding
        layer_idx = 0
    elif layer_name.startswith('cls.'):  # cls token
        layer_idx = 0
    elif layer_name.startswith('Transformer.encoder_norm.'):  # last norm
        layer_idx = num_layers  # the last layer
    elif layer_name.startswith('head.'):  # head
        layer_idx = num_layers  # the last layer
    else:
        raise NotImplementedError('lrd not defined: {}'.format(layer_name))
    layer_lrd = lr_decay ** (num_layers - layer_idx)
    return layer_lrd


# ---------------------------------------------------------
# the entrance function:
# ---------------------------------------------------------
def filter_parameters(params, filter_fn):
    """Filter the params based on filter_fn."""
    params_to_filter = nest.map_structure_with_path(filter_fn, params)
    return params_to_filter

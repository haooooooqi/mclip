from typing import Tuple, Any

import functools
import tree as nest

import jax
import jax.numpy as jnp

from optax._src import base
from optax._src import transform


# ---------------------------------------------------------
# rescale lr
# ---------------------------------------------------------
def lrd_func(num_layers: int, lr_decay: float):
    """Get the lrd function."""
    return functools.partial(_layerwise_lr_decay, num_layers=num_layers, lr_decay=lr_decay)


def _layerwise_lr_decay(
        path: Tuple[Any], val: jnp.ndarray,
        num_layers: int, lr_decay: float, pre_enc_stgs: int = 4):
    """Get the layerwise lr decay rate based on name."""
    del val

    layer_name = path[1]
    # print("Layer Name: ", layer_name)
    if layer_name.startswith("downsample_layers"):
        stage_id = int((layer_name.split('downsample_layers'))[1][0])
        # print("Stage ID: ", stage_id)
        if stage_id == 0:
            layer_idx = 0
        elif stage_id == 1 or stage_id == 2:
            layer_idx = stage_id + 1
        elif stage_id == 3:
            layer_idx = 12
    elif layer_name.startswith("stages"):
        stage_block_id = (layer_name.split('stages'))[1]
        stage_id = int(stage_block_id[0])
        block_id = int(stage_block_id[1:])
        # print("Stage Block ID: ", stage_id, " : ", block_id)
        if stage_id == 0 or stage_id == 1:
            layer_idx = stage_id + 1
        elif stage_id == 2:
            layer_idx = 3 + block_id // 3 
        elif stage_id == 3:
            layer_idx = 12
    else:
        layer_idx = num_layers + 1 if pre_enc_stgs == 4 else num_layers

    layer_lrd = lr_decay ** (num_layers + 1 - layer_idx)
    return layer_lrd

def _layerwise_lr_decay_masked_convnext(
        path: Tuple[Any], val: jnp.ndarray,
        num_layers: int, lr_decay: float, pre_enc_stgs: int = 4):
    """Get the layerwise lr decay rate based on name."""
    del val

    layer_name = path[1]
    # print("Layer Name: ", layer_name)
    if layer_name.startswith("downsample_layers"):
        stage_id = int((layer_name.split('downsample_layers'))[1][0])
        # print("Stage ID: ", stage_id)
        if stage_id == 0 or stage_id == 1:
            layer_idx = 0
        elif stage_id == 2:
            layer_idx = stage_id - 1
        elif stage_id == 3:
            layer_idx = num_layers
    elif layer_name.startswith("stages"):
        stage_block_id = (layer_name.split('stages'))[1]
        stage_id = int(stage_block_id[0])
        block_id = int(stage_block_id[1:])
        # print("Stage Block ID: ", stage_id, " : ", block_id)
        if stage_id == 0 or stage_id == 1:
            layer_idx = 0
        elif stage_id == 2:
            layer_idx = 1 + block_id // 3 
        elif stage_id == 3:
            layer_idx = num_layers
    else:
        layer_idx = num_layers + 1 if pre_enc_stgs == 4 else num_layers

    layer_lrd = lr_decay ** (num_layers + 1 - layer_idx)
    return layer_lrd


# ---------------------------------------------------------
# the entrance function:
# ---------------------------------------------------------
def filter_parameters(params, filter_fn):
    """Filter the params based on filter_fn."""
    params_to_filter = nest.map_structure_with_path(filter_fn, params)
    return params_to_filter


# ---------------------------------------------------------
# the entrance function:
# ---------------------------------------------------------
def scale_by_lrd(
    lrd: Any
) -> base.GradientTransformation:

  def init_fn(_):
    return transform.ScaleState()

  def update_fn(updates, state, params=None):
    del params
    updates = jax.tree_map(lambda s, g: s * g, lrd, updates)
    return updates, state

  return base.GradientTransformation(init_fn, update_fn)

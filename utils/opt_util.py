import jax.numpy as jnp

import tree as nest

from typing import Tuple, Any

# reference: https://github.com/rwightman/efficientnet-jax/blob/6e343d5ff4e02f4500c3d979cb75bb0f33e2c601/jeffnet/common/optim/lars.py#L36


# ---------------------------------------------------------
# rescale layers
# ---------------------------------------------------------
def layer_rescale(path: Tuple[Any], val: jnp.ndarray):
    """Rescale the last layer of each block by layer_id."""
    del val
    # path[0] = 'Transformer'
    if len(path) > 3 and path[1].startswith("encoderblock_") and path[-1] == 'kernel':
        if (path[-3] == 'self_attention' and path[-2] == 'out') or \
                (path[-3] == 'mlp' and path[-2] == 'Dense_1'):
            layer_id = path[1][len("encoderblock_"):]  # remove prefix
            layer_id = int(layer_id) + 1
            rescale = (2.0 * layer_id) ** -.5
            return rescale
    return 1.


# ---------------------------------------------------------
# exclude wd
# ---------------------------------------------------------
def filter_bias_and_norm(path: Tuple[Any], val: jnp.ndarray):
    """Filter to exclude biases and normalizations weights."""
    del val
    if path[-1] == "bias" or path[-1] == 'scale':
        return False
    return True


def filter_cls_and_posembed(path: Tuple[Any], val: jnp.ndarray):
    """Filter to exclude cls token and pos emb."""
    del val
    name = '.'.join(path)
    if 'pos_embedding' in name or 'cls' == name:
        return False
    return True


# ---------------------------------------------------------
# freeze for the predictor
# ---------------------------------------------------------
def filter_predictor(path: Tuple[Any], val: jnp.ndarray, config: Any):
    """Filter for predictor"""
    del val

    # hack: sanity check
    all_keys = [
        'Transformer', 'cls', 'embedding', 'posembed_encoder',
        'head', 'pred_posembed', 'pred', 'pred_bottleneck', 'bottleneck', 'fc_norm']
    assert path[0] in all_keys, 'key not valid: {}'.format(path[0])

    pretrained_keys = ['Transformer', 'cls', 'embedding', 'posembed_encoder']
    trainable_keys = ['pred', 'fc_norm', 'head']
    if config.model.predictor.sincos:
        pretrained_keys += ['pred_posembed']
    else:
        trainable_keys += ['pred_posembed']

    if config.model.load_bottleneck:
        trainable_keys += ['bottleneck']
    else:
        trainable_keys += ['pred_bottleneck']

    if path[0] in trainable_keys:
        return True
    elif path[0] in pretrained_keys:
        return False
    else:
        assert False, 'key not valid: {}'.format(path[0])


# ---------------------------------------------------------
# freeze layers in the backbone
# ---------------------------------------------------------
def filter_block(path: Tuple[Any], val: jnp.ndarray, config: Any):
    """Freeze/train blocks by layer_id."""
    del val

    layer_idx = _layerwise_index(path, config.model.transformer.num_layers)

    return layer_idx >= config.model.stopgrad_blocks


def _layerwise_index(path: Tuple[Any], num_layers: int):
    """Get the layerwise index based on name."""

    layer_name = '.'.join(path)

    if layer_name.startswith('Transformer.encoderblock_'):
        layer_idx = path[1][len('encoderblock_'):]  # e.g., '01'
        layer_idx = int(layer_idx)
    elif layer_name.startswith('embedding.'):  # patch embedding
        layer_idx = 0
    elif layer_name.startswith('posembed_'):  # position embedding
        layer_idx = 0
    elif layer_name.startswith('cls'):  # cls token
        layer_idx = 0
    elif layer_name.startswith('Transformer.encoder_norm.'):  # last norm
        layer_idx = num_layers
    elif layer_name.startswith('fc_norm.'):
        layer_idx = num_layers
    elif layer_name.startswith('head.'):
        layer_idx = num_layers
    elif layer_name.startswith('bn_debug.'):
        layer_idx = num_layers
    elif layer_name.startswith('pred'):
        layer_idx = num_layers
    else:
        raise NotImplementedError('lrd not defined: {}'.format(layer_name))

    return layer_idx


# ---------------------------------------------------------
# the entrance function:
# ---------------------------------------------------------
def filter_parameters(params, filter_fn):
    """Filter the params based on filter_fn."""
    params_to_filter = nest.map_structure_with_path(filter_fn, params)
    return params_to_filter

from typing import Tuple, Any

from absl import logging

import tree as nest
import numpy as np

import jax
import jaxlib
import flax
from flax.training import checkpoints
import jax.numpy as jnp

import jax.tree_util as tu
import t5x.checkpoints


# --------------------------------------------
# interfaces
# --------------------------------------------
def restore_checkpoint(checkpointer, path):
  step = t5x.checkpoints.latest_step(path)  # try to load the latest checkpoint if not specified
  path_chkpt = path if step is None else t5x.checkpoints.get_checkpoint_dir(path, step)
  state = checkpointer.restore(path=path_chkpt)
  return state


def restore_from_pretrain(state, config, partitioner, state_axes):
  if config.pretrain_fmt == 'jax':
    logging.info('Loading from JAX pre-training format:')
    state = load_from_pretrain(state, config.pretrain_dir)
  else:
    raise NotImplementedError

  params = partitioner.move_params_to_devices(state.params, state_axes.params)
  state = state.replace_params(params)
  return state


# --------------------------------------------
# load JAX checkpoint from pre-training
# --------------------------------------------
def load_from_pretrain(state, pretrain_dir):
  state_load = checkpoints.restore_checkpoint(pretrain_dir, target=None)
  load_params = flax.core.freeze(state_load.pop('params'))  # match the type of state.params

  if 'variables' in state_load.keys():
    variables_load = state_load.pop('variables')
    logging.info('Variables (not used): {}'.format(variables_load.keys()))
    # assert variables_load == {}  # no state variables in ViT (no BatchNorm)

  del state_load

  state_params = flax.core.freeze(state.params)

  named_load_params, _ = to_named_parameters(load_params)
  named_state_params, tree_ref = to_named_parameters(state_params)

  missing_keys = set(named_state_params.keys()) - set(named_load_params.keys())
  load_keys = set(named_state_params.keys()) - missing_keys
  ignored_keys = set(named_load_params.keys()) - load_keys - missing_keys

  # logging.info('Loaded keys: {}'.format(load_keys))
  logging.info('Missing keys: {}'.format(missing_keys))
  logging.info('Ignored keys: {}'.format(ignored_keys))

  assert len(missing_keys) == 2 or len(missing_keys) == 4

  named_params = {}
  for k in named_state_params.keys():
    if k in missing_keys:
      named_params[k] = named_state_params[k]
    elif k in load_keys:
      assert np.prod(named_state_params[k].shape) == np.prod(named_load_params[k].shape), \
        'Not matching: {}, {}, {}'.format(k, named_state_params[k].shape, named_load_params[k].shape)
      if named_state_params[k].shape != named_load_params[k].shape:
        logging.info('Reshaping: {}, {}, {}'.format(k, named_state_params[k].shape, named_load_params[k].shape))
        named_load_params[k] = named_load_params[k].reshape(named_state_params[k].shape)      
      named_params[k] = named_load_params[k]
    else:
      assert False, 'Error: {}'.format(k)
  
  list_params = []
  for k in named_params.keys():
    list_params.append(named_params[k])

  params = tu.tree_unflatten(tree_ref, list_params)
  params = flax.core.frozen_dict.freeze(params)
  
  # sanity
  tu.tree_map(lambda x, y: (x.shape, y.shape), params, state_params)
  verify = tu.tree_map(lambda x, y: (x.shape == y.shape), params, state_params)
  verify = tu.tree_leaves(verify)
  assert jnp.all(jnp.array(verify)).item()

  del state_params
  state = state.replace_params(params=params)
  return state


def get_name(path: Tuple[Any], val: jnp.ndarray):
  del val
  layer_name = '.'.join(path)
  return layer_name


def filter_parameters(params, filter_fn):
  """Filter the params based on filter_fn."""
  params_to_filter = nest.map_structure_with_path(filter_fn, params)
  return params_to_filter


def to_named_parameters(params):
  """ Turn a PyTree params to a dict of named_parameters
  """
  names = filter_parameters(params, get_name)
  list_params, tree_params = tu.tree_flatten(params)
  list_names, tree_names = tu.tree_flatten(names)
  assert tree_params == tree_names
  del tree_names
  named_params = dict((x, y) for x, y in zip(list_names, list_params))
  return named_params, tree_params

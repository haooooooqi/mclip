from termcolor import colored
from absl import logging

import jax
import jax.numpy as jnp
import optax
import functools

from flax import traverse_util
try:
  from jax.interpreters.sharded_jit import PartitionSpec
except ImportError:
  from jax.interpreters.pxla import PartitionSpec

import t5x.train_state as train_state_lib
import t5x.optimizers

from utils import opt_util
from utils import adamw


def init_fn(rng, image_size, num_views, model):
  if num_views == 1:
    input_shape = (1, image_size, image_size, 3)
  else:
    input_shape = (1, num_views, image_size, image_size, 3)
  variables = model.init({'params': rng, 'dropout': jax.random.PRNGKey(0)},
                        {'image': jnp.ones(input_shape, model.dtype), 'label': jnp.zeros((1,), jnp.int32)},
                        train=True, update=False)
  return variables


def init_shapes(rng, image_size, num_views, model):
  if num_views == 1:
    input_shape = (1, image_size, image_size, 3)
  else:
    input_shape = (1, num_views, image_size, image_size, 3)
  init = functools.partial(model.init, train=True, update=False)
  variables_shape = jax.eval_shape(init,
                                  {'params': rng, 'dropout': jax.random.PRNGKey(0)},
                                  {'image': jnp.ones(input_shape, model.dtype), 'label': jnp.zeros((1,), jnp.int32)})
  return variables_shape


def create_learning_rate_fn(
    config,
    base_learning_rate: float,
    steps_per_epoch: int):
  """Create learning rate schedule."""
  warmup_fn = optax.linear_schedule(
      init_value=config.warmup_abs_lr, end_value=base_learning_rate,
      transition_steps=config.warmup_epochs * steps_per_epoch)
  main_epochs = max(config.num_epochs - config.warmup_epochs, 1)
  if config.lr_schedule == 'cos':
    main_fn = optax.cosine_decay_schedule(
        init_value=base_learning_rate,
        decay_steps=main_epochs * steps_per_epoch,
        alpha=config.min_abs_lr / base_learning_rate)
  elif config.lr_schedule == 'linear':
    main_fn = optax.linear_schedule(
        init_value=base_learning_rate,
        transition_steps=main_epochs * steps_per_epoch,
        end_value=config.min_abs_lr)
  else:
    raise NotImplementedError
  schedule_fn = optax.join_schedules(
      schedules=[warmup_fn, main_fn],
      boundaries=[config.warmup_epochs * steps_per_epoch])
  return schedule_fn


def create_optimizer(config, params_names, steps_per_epoch):
  # create the lr schedule function
  abs_learning_rate = config.learning_rate * config.batch_size / 256.
  learning_rate_fn = create_learning_rate_fn(config, abs_learning_rate, steps_per_epoch)

  if config.opt_type in {'adamw',}:
    # optional: exclude some wd
    mask = None
    if config.exclude_wd:
      mask = jax.tree_util.tree_map(lambda x, y: bool(x and y),
        opt_util.filter_parameters(params_names, opt_util.filter_bias_and_norm),
        opt_util.filter_parameters(params_names, opt_util.filter_posembed)  # Note: we must exclude posembed wd in adamw
      )
    # logging.info('Apply wd: {}'.format(mask))

    if config.model_type in {'mclr',}:
      opt_inner = getattr(adamw, config.opt_type)  # optax.adamw
      mask_trainable = opt_util.filter_parameters(params_names, opt_util.filter_momentum_encoder)
      # logging.info(colored('Trainable: {}'.format(t5x.state_utils.str_flatten_dict(mask_trainable)), "red"))

      def opt(**kwargs) -> optax._src.base.GradientTransformation:  # same type as opt
        return adamw.masked(inner=opt_inner(**kwargs), mask=mask_trainable)
    elif config.model_type in {'mae',}:
      opt_inner = getattr(adamw, config.opt_type)  # optax.adamw
      mask_trainable = opt_util.filter_parameters(params_names, opt_util.filter_posembed)
      # logging.info(colored('Trainable: {}'.format(t5x.state_utils.str_flatten_dict(mask_trainable)), "red"))

      def opt(**kwargs) -> optax._src.base.GradientTransformation:  # same type as opt
        return adamw.masked(inner=opt_inner(**kwargs), mask=mask_trainable)
    else:
      opt = getattr(adamw, config.opt_type)

    # t5x will wrap the optimizer
    opt = t5x.optimizers.wrap_optax_optimizer(opt)
    opt = opt(learning_rate=learning_rate_fn,
              **config.opt,
              mask=mask,
              mu_dtype=getattr(jnp, config.opt_mu_dtype))
    opt.metric_learning_rate_fn = learning_rate_fn  # hack for metric

  else:
    raise NotImplementedError

  return opt


def create_train_state(config, model, steps_per_epoch, partitioner):
  """Create initial training state."""
  rng = jax.random.PRNGKey(0)  # for shape reference only
  # create optimizer first
  # shape of the parameters
  params_shapes = init_shapes(rng, config.image_size, config.num_views, model)
  # created the optimizer
  opt = create_optimizer(config, params_shapes['params'], steps_per_epoch)

  # ---------------------------------------------------------------------------
  def initialize_train_state(rng_init):
    # split rng for init and for state
    initial_variables = init_fn(rng_init, config.image_size, config.num_views, model)
    if opt: # train
      return train_state_lib.FlaxOptimTrainState.create(opt, initial_variables)
    else: # test
      return train_state_lib.InferenceState.create(initial_variables)

  train_state_shape = jax.eval_shape(initialize_train_state, rng_init=rng)
  train_state_axes = partitioner.get_mesh_axes(train_state_shape)

  p_init_fn = partitioner.partition(
      initialize_train_state,
      in_axis_resources=None,
      out_axis_resources=train_state_axes)

  return p_init_fn, train_state_axes, train_state_shape

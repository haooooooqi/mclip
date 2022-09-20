from termcolor import colored
from absl import logging
import copy

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
  # the variables are initialized here
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

  # optional: exclude some wd
  mask_wd = None
  if config.exclude_wd:
    mask_wd = jax.tree_map(lambda x: x,
      opt_util.filter_parameters(params_names, opt_util.filter_bias_and_norm)
    )
  logging.info(colored('Apply wd: {}'.format(t5x.state_utils.str_flatten_dict(mask_wd)), "blue"))

  opt_args = copy.deepcopy(config.opt)
  with opt_args.unlocked():
    opt_args.learning_rate = learning_rate_fn
    opt_args.mask = mask_wd
    opt_args.mu_dtype = getattr(jnp, config.opt_mu_dtype)

    del opt_args.ema_momentum
    del opt_args.ema_schedule

  momentum_models = ('mclr', 'maco', 'mv3', 'cder')

  if config.opt_type in ('adamw',):
    if config.model_type in momentum_models:
      opt_inner = getattr(adamw, config.opt_type)
      mask_trainable = opt_util.filter_parameters(params_names,
                            functools.partial(opt_util.filter_by_keywords,
                                              keywords=config.freeze_keywords))
      logging.info(colored('Trainable: {}'.format(t5x.state_utils.str_flatten_dict(mask_trainable)), "red"))

      def opt(ema_momentum, **kwargs) -> optax._src.base.GradientTransformation:  # same type as opt
        return opt_util.masked_with_momentum(inner=opt_inner(**kwargs),
                                            ema_momentum=ema_momentum,
                                            mask=mask_trainable)
      with opt_args.unlocked():
        if config.opt.ema_schedule == 'const':
          opt_args.ema_momentum = config.opt.ema_momentum
        elif config.opt.ema_schedule == 'cos':
          opt_args.ema_momentum = opt_util.cosine_increase_schedule(init_value=config.opt.ema_momentum,
                                                                  steps=config.num_epochs * steps_per_epoch)
        else:
          raise NotImplementedError
    elif len(config.freeze_keywords) > 0:
      opt_inner = getattr(adamw, config.opt_type)
      mask_trainable = opt_util.filter_parameters(params_names,
                            functools.partial(opt_util.filter_by_keywords,
                                              keywords=config.freeze_keywords))
      logging.info(colored('Trainable: {}'.format(t5x.state_utils.str_flatten_dict(mask_trainable)), "red"))

      def opt(**kwargs) -> optax._src.base.GradientTransformation:  # same type as opt
        return opt_util.masked(inner=opt_inner(**kwargs), mask=mask_trainable)
    else:
      opt = getattr(adamw, config.opt_type)

    # t5x will wrap the optimizer
    opt = t5x.optimizers.wrap_optax_optimizer(opt)
    opt = opt(**opt_args)

    # for metrics
    opt.metric_learning_rate_fn = learning_rate_fn
    if config.model_type in momentum_models and config.opt.ema_schedule == 'cos':
      opt.metric_momentum_fn = opt_args.ema_momentum

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
  params_names = params_shapes['params']
  opt = create_optimizer(config, params_names, steps_per_epoch)

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

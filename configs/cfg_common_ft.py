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

# Copyright 2021 The Flax Authors.
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
"""Hyperparameter configuration to run the example on TPUs."""

import ml_collections

import configs.vit as vit


def get_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = ml_collections.ConfigDict()

  # As defined in the `models` module.
  # config.model = 'ResNet50'
  # `name` argument of tensorflow_datasets.builder()
  config.dataset = 'imagenet2012:5.*.*'

  config.learning_rate = 1e-3  # this is the base lr
  config.warmup_epochs = 5.0
  config.min_abs_lr = 1e-6  # this is abs lr
  config.warmup_abs_lr = 1e-6  # this is abs lr

  config.learning_rate_decay = 0.75  # lrd

  config.num_epochs = 100.0
  config.log_every_steps = 100
  config.save_every_epochs = 10

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  # Consider setting the batch size to max(tpu_chips * 256, 8 * 1024) if you
  # train on a larger pod slice.
  config.batch_size = 4096
  config.cache = True

  # model config
  config.model = vit.get_b16_config()  # ViT-B/16
  config.model.transformer.dropout_rate = 0.0
  config.model.transformer.droppath_rate = 0.1
  config.model.num_classes = 1000
  # number of blocks with stop gradient (stopgrad_blocks=1 means stopgrad applied after block0 and before block1)
  # stopgrad_blocks=0 is for sanity check
  config.model.stopgrad_blocks = -1
  config.model.sincos = False

  config.model.predictor = ml_collections.ConfigDict()
  config.model.predictor.hidden_size = 768
  config.model.predictor.transformer = ml_collections.ConfigDict()
  config.model.predictor.transformer.mlp_dim = config.model.predictor.hidden_size * 4
  config.model.predictor.transformer.num_heads = 16
  config.model.predictor.transformer.num_layers = 2
  config.model.predictor.transformer.attention_dropout_rate = 0.0
  config.model.predictor.transformer.dropout_rate = 0.0
  config.model.predictor.transformer.droppath_rate = 0.0

  config.model.adapter = ml_collections.ConfigDict()
  config.model.adapter.on_use = False
  config.model.adapter.mlp_dim_ratio = 1 / 4.
  config.model.adapter.rescale_init = 1e-4

  config.model.split = ml_collections.ConfigDict()
  config.model.split.on_use = False
  config.model.split.splits = 4

  # optimizer config
  config.opt_type = 'adamw'
  config.opt = ml_collections.ConfigDict()
  config.opt.b1 = 0.9
  config.opt.b2 = 0.999
  config.opt.weight_decay = 0.05
  
  config.opt_mu_dtype = 'float32'

  config.exclude_wd = True  # exclude some weight decays (bias, norm, cls, posembed)
  config.exclude_wd_adapter = False

  # config.ema = False
  # config.ema_decay = 0.9999
  # config.ema_eval = False

  # aug config
  config.aug = ml_collections.ConfigDict()

  config.aug.area_range = (0.08, 1)
  config.aug.aspect_ratio_range = (3. / 4, 4. / 3.)
  config.aug.crop_ver = 'v4'  # v1, v3

  config.aug.label_smoothing = 0.1

  config.aug.autoaug = 'autoaug'  # autoaug, randaug, or None

  config.aug.color_jit = None  # [0.4, 0.4, 0.4]  # None to disable; [brightness, contrast, saturation]

  # mixup config
  config.aug.mix = ml_collections.ConfigDict()
  config.aug.mix.mixup = True
  config.aug.mix.mixup_alpha = 0.8

  config.aug.mix.cutmix = True
  config.aug.mix.cutmix_alpha = 1.0

  config.aug.mix.mode = 'batch'  # batch, pair, elem: for timm mixup only

  # rand erase config
  config.aug.randerase = ml_collections.ConfigDict()
  config.aug.randerase.on = False
  config.aug.randerase.prob = 0.25

  # init config
  config.rescale_init = False  # rescale initialized weights by layer id
  config.model.rescale_head_init = 0.001  # rescale the head initialized weights

  # memory
  config.profile_memory = False
  config.donate = False
  config.init_backend = 'tpu'

  # utils
  config.resume_dir = ''

  config.pretrain_dir = ''
  config.pretrain_fmt = 'jax'  # 't5x'

  config.eval_only = False

  # seeds
  config.seed_jax = 0
  config.seed_tf = 0
  config.seed_pt = 0

  # torchload
  config.torchload = ml_collections.ConfigDict()
  config.torchload.data_dir = '/kmh_data/imagenet_full_size/061417'
  config.torchload.num_workers = 32

  # partitioning
  config.partitioning = ml_collections.ConfigDict()
  config.partitioning.num_partitions = 1
  config.partitioning.partition_states = False

  return config

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

  config.learning_rate = 1e-4  # this is the base lr
  config.warmup_epochs = 20.0
  # config.momentum = 0.9

  config.num_epochs = 100.0
  config.log_every_steps = 100

  # If num_train_steps==-1 then the number of training steps is calculated from
  # num_epochs using the entire dataset. Similarly for steps_per_eval.
  config.num_train_steps = -1
  config.steps_per_eval = -1

  # Consider setting the batch size to max(tpu_chips * 256, 8 * 1024) if you
  # train on a larger pod slice.
  config.batch_size = 1024
  config.cache = True
  config.half_precision = False  # kaiming: TODO, support it

  # model config
  config.model = vit.get_b16_config()  # ViT-B/16

  # optimizer config
  config.opt_type = 'adamw'
  config.opt = ml_collections.ConfigDict()
  config.opt.b1 = 0.9
  config.opt.b2 = 0.95
  config.opt.weight_decay = 0.3

  return config
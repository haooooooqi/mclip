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
import configs.cfg_common_ft as cfg_common_ft


def get_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = cfg_common_ft.get_config()

  # convmae encoder config
  config.model.encoder = ml_collections.ConfigDict()
  config.model.encoder.name = 'convnext'
  config.model.encoder.dims = (256, 512, 1024, 2048)
  config.model.encoder.depths = (3,3,27,3)
  config.model.encoder.drop_path = 0.5
  config.model.encoder.layer_scale_init_value = 1e-6
  config.model.encoder.head_init_scale = 1.0
  config.model.encoder.attach_head = True
  config.model.encoder.num_classes = 1000

  return config

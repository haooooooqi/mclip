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
import configs.cfg_common_mclr as cfg_common_mclr


def get_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = cfg_common_mclr.get_config()

  config.model.encoder.update(vit.get_l16_config())
  config.model.encoder.hidden_size = 1280
  config.model.encoder.transformer.mlp_dim = config.model.hidden_size * 4
  config.model.encoder.transformer.num_layers = 32
  config.model.encoder.transformer.rescale_init = 1.0

  config.partitioning.num_partitions = 1
  config.partitioning.partition_states = False

  return config

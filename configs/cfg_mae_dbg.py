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
import configs.mae as mae
import configs.cfg_mae_common as cfg_mae_common


def get_config():
  """Get the hyperparameter configuration to train on TPUs."""
  config = cfg_mae_common.get_config()

  # mae config
  config.model.mask_ratio = 0.75

  config.model.update(vit.get_b16_config())
  config.model.transformer.dropout_rate = 0.0
  config.model.transformer.droppath_rate = 0.0
  config.model.transformer.num_layers = 3

  config.model.decoder.update(vit.get_testing_config())
  config.model.decoder.hidden_size = 8
  config.model.decoder.transformer.dropout_rate = 0.0
  config.model.decoder.transformer.droppath_rate = 0.0
  config.model.decoder.num_layers = 2

  # opt config
  config.opt_mu_dtype = 'float32'

  return config

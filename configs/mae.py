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

  config.sincos = True

  config.update(vit.get_b16_config())
  config.transformer.dropout_rate = 0.0
  config.transformer.droppath_rate = 0.0

  config.decoder = vit.get_testing_config()
  config.decoder.transformer.dropout_rate = 0.0
  config.decoder.transformer.droppath_rate = 0.0

  config.visualize = False

  return config

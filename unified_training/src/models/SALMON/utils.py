# Copyright (2024) Tsinghua University, Bytedance Ltd. and/or its affiliates
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

import torch
from transformers import StoppingCriteria


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = [s if isinstance(s, torch.Tensor) else torch.tensor(s) for s in stops]
        # Track which batch indices have already hit a stop sequence
        self.stopped_indices = set()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        batch_size = input_ids.shape[0]
        device, dtype = input_ids.device, input_ids.dtype

        for i in range(batch_size):
            if i in self.stopped_indices:
                continue
            for stop in self.stops:
                stop = stop.to(device=device, dtype=dtype)
                if stop.shape[0] <= input_ids.shape[1]:
                    if torch.all(stop == input_ids[i, -stop.shape[0]:]).item():
                        self.stopped_indices.add(i)
                        break

        return len(self.stopped_indices) == batch_size
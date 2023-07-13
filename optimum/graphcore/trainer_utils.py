#  Copyright 2021 The HuggingFace Team. All rights reserved.
#  Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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

import functools

import numpy as np
import poptorch
import torch

from optimum.utils import logging


logger = logging.get_logger(__name__)


class _WorkerInit:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed((self.seed + worker_id) % np.iinfo(np.uint32).max)


def to_poptorch_dataloader(for_training=False):
    def method_wrapper(func):
        def wrapper(*args, **kwargs):
            self = args[0]
            poptorch_specific_kwargs = {
                # Not dropping last will end up causing NaN during training if the combined batch size does not divide the number of steps
                "drop_last": True if for_training else self.args.dataloader_drop_last,
                # TODO: how to handle this case
                # "auto_distributed_partitioning": not isinstance(train_dataset, torch.utils.data.IterableDataset),
                "mode": self.args.dataloader_mode,
                "worker_init_fn": _WorkerInit(self.args.seed),
            }
            opts = self.opts if for_training else self.eval_opts
            orig_init = poptorch.DataLoader.__init__
            partial_init = functools.partialmethod(poptorch.DataLoader.__init__, opts, **poptorch_specific_kwargs)
            poptorch.DataLoader.__init__ = partial_init
            orig_dataloader = torch.utils.data.DataLoader
            torch.utils.data.DataLoader = poptorch.DataLoader
            res = func(*args, **kwargs)
            poptorch.DataLoader.__init__ = orig_init
            torch.utils.data.DataLoader = orig_dataloader
            return res

        return wrapper

    return method_wrapper

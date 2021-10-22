#  copyright 2021 the huggingface team. all rights reserved.
#
#  licensed under the apache license, version 2.0 (the "license");
#  you may not use this file except in compliance with the license.
#  you may obtain a copy of the license at
#
#      http://www.apache.org/licenses/license-2.0
#
#  unless required by applicable law or agreed to in writing, software
#  distributed under the license is distributed on an "as is" basis,
#  without warranties or conditions of any kind, either express or implied.
#  see the license for the specific language governing permissions and
#  limitations under the license.

import functools
from typing import Any, Dict, Optional

import numpy as np
import torch
import poptorch
from poptorch.enums import DataLoaderMode

import transformers


class _WorkerInit:
    def __init__(self, seed):
        self.seed = seed

    def __call__(self, worker_id):
        np.random.seed((self.seed + worker_id) % np.iinfo(np.uint32).max)


class IPUDataLoader(poptorch.DataLoader):

    def __init__(
        self,
        config: 'transformers.PretrainedConfig',
        options: 'poptorch.Options',
        dataset: 'torch.utils.data.Dataset',
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = True,
        persistent_workers: Optional[bool] = None,
        auto_distributed_partitioning: bool = True,
        mode: 'poptorch.DataLoaderMode' = DataLoaderMode.Sync,
        async_options: Optional[Dict[str, Any]] = None,
        rebatched_worker_size: Optional[int] = None,
        **kwargs
    ):
        worker_init_fn = _WorkerInit(config.random_seed)
        auto_distributed_partitioning = not isinstance(dataset, torch.utils.data.IterableDataset)
        mode = DataLoaderMode.AsyncRebatched if config.async_dataloader else DataLoaderMode.Sync
        return super().__init__(
            options,
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            auto_distributed_partitioning=auto_distributed_partitioning,
            worker_init_fn=worker_init_fn,
            mode=mode,
            async_options=async_options,
            rebatched_worker_size=rebatched_worker_size,
            **kwargs
        )


def dataloader_method_wrapper(func):

    def wrapper(*args, **kwargs):
        orig_init = IPUDataLoader.__init__
        partial_init = functools.partialmethod(IPUDataLoader.__init__, args[0].model.config, args[0].opts)
        IPUDataLoader.__init__ = partial_init
        orig_dataloader = torch.utils.data.DataLoader
        torch.utils.data.DataLoader = IPUDataLoader
        res = func(*args, **kwargs)
        IPUDataLoader.__init__ = orig_init
        torch.utils.data.DataLoader = orig_dataloader
        return res

    # TODO: make sure to bind the wrapped method.
    # wrapper = wrapper.__get__(func.__self__, func.__self__.__class__)

    return wrapper

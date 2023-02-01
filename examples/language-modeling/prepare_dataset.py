#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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

import os
import time
from argparse import ArgumentError, ArgumentParser
from pathlib import Path
from typing import Optional, Sequence, Union

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

from joblib import Parallel, delayed


def prepare_dataset(dataset_name: str, cache_dir: Path, data_dir: Path, data_files: Sequence[str]) -> Dataset:
    raw_dataset = load_dataset(
        dataset_name,
        data_files=data_files,
        cache_dir=cache_dir,
    )
    return raw_dataset


def main(
    num_workers: int,
    dataset_name: str,
    data_dir: Union[Path, str],
    cache_dir: Union[Path, str],
    output_dataset_name: Optional[str] = None,
    remove_intermediate_datasets_from_cache: bool = True,
    max_number_of_files: int = -1,
) -> Dataset:
    data_files = list(map(str, Path(data_dir).glob("*.tfrecord")))
    if max_number_of_files > 0:
        data_files = data_files[:max_number_of_files]
    if num_workers > len(data_files):
        raise ValueError(f"there are more workers ({num_workers}) than the number of files ({len(data_files)})")
    num_data_file_per_worker = len(data_files) // num_workers
    data_files_per_worker = [
        data_files[i * num_data_file_per_worker : (i + 1) * num_data_file_per_worker] for i in range(num_workers)
    ]
    remaining_files = len(data_files) % num_workers
    if remaining_files > 0:
        # Dispatching the remaning files to different workers
        for i in range(1, remaining_files + 1):
            data_files_per_worker[-i].append(data_files[-i])

    formatted_filenames = "\n".join(data_files)
    print(f"Found {len(data_files)} files:\n{formatted_filenames}")
    print(f"Number of files per worker ~ {num_data_file_per_worker}")

    print(f"Generating the dataset with {num_workers} workers...")
    start = time.time()
    sub_datasets = Parallel(n_jobs=num_workers)(
        delayed(prepare_dataset)(dataset_name, cache_dir, data_dir, data_files) for data_files in data_files_per_worker
    )
    final_datasets = DatasetDict()
    split_names = sub_datasets[0].keys()
    for split_name in split_names:
        final_datasets[split_name] = concatenate_datasets(
            [dataset_dict[split_name] for dataset_dict in sub_datasets], split=split_name
        )
    end = time.time()
    print(f"Dataset generation completed after {end - start}s")
    if output_dataset_name is None:
        final_dataset_filename = Path(cache_dir) / dataset_name.replace("/", "_")
    else:
        final_dataset_filename = Path(cache_dir) / output_dataset_name
    final_datasets.save_to_disk(final_dataset_filename)

    if remove_intermediate_datasets_from_cache:
        print("*** Cleaning up intermediate dataset cache files ***")
        for dataset in sub_datasets:
            for _, cache_files in dataset.cache_files.items():
                for cache_file in cache_files:
                    filename = cache_file.get("filename")
                    if filename is None:
                        continue
                    print(f"Removing {filename}")
                    Path(filename).unlink()
        print("Done!")

    print(f"Dataset saved at {final_dataset_filename}")
    return final_datasets


def get_args():
    parser = ArgumentParser(description="Utility tool to enable multiprocessing during dataset generation")
    parser.add_argument("--num_workers", required=True, type=int, help="The number of workers to use.")
    parser.add_argument(
        "--dataset_name", required=True, type=str, help="The name of the dataset, or the path to the dataset script."
    )
    parser.add_argument(
        "--data_dir", required=True, type=Path, help="The path to the directory containing the dataset files."
    )
    parser.add_argument("--cache_dir", default=None, type=Path, help="The path to the cache directory.")
    parser.add_argument(
        "--output_dataset_name",
        default=None,
        type=str,
        help="The resulting dataset folder name, --dataset_name is used if this field is None",
    )
    parser.add_argument(
        "--keep_intermediate_datasets",
        default=False,
        type=bool,
        help="Whether to keep intermediate dataset cache files or not.",
    )
    parser.add_argument("--max_number_of_files", default=-1, type=int, help="The maximum number of files to process.")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    if args.cache_dir is None:
        args.cache_dir = os.environ.get("HF_DATASETS_CACHE", None)
        if args.cache_dir is None:
            raise ArgumentError(
                "You must either specify a cache_dir or set the HF_DATASETS_CACHE environment variable"
            )

    print(f"Cache dir: {args.cache_dir}")

    main(
        args.num_workers,
        args.dataset_name,
        args.data_dir,
        args.cache_dir,
        args.output_dataset_name,
        remove_intermediate_datasets_from_cache=not args.keep_intermediate_datasets,
        max_number_of_files=args.max_number_of_files,
    )

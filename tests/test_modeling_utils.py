# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
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

import unittest
from abc import abstractmethod

import poptorch
import torch
from parameterized import parameterized

from optimum.graphcore.modeling_utils import SerializedEmbedding, SerializedLinear, SplitProjection


DATALOADER_BATCH_SIZE = 8


class RandomDataset:
    def __init__(self, num_features: int, num_samples: int, rand_func: torch.randn) -> None:
        self.in_features = num_features
        self.num_samples = num_samples
        self.dataset = rand_func((num_samples, num_features))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, i):
        return self.dataset[i]


class BaseEvaluator:
    def __init__(self, model) -> None:
        self.wrapped_model = model.eval()

    @abstractmethod
    def get_dataloader(self, dataset):
        ...

    def evaluate(self, dataset):
        out = []
        for sample in self.get_dataloader(dataset):
            out.append(self.wrapped_model(sample))
        return torch.vstack(out)


class PytorchEvaluator(BaseEvaluator):
    def get_dataloader(self, dataset):
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=DATALOADER_BATCH_SIZE)


class PoptorchEvaluator(BaseEvaluator):
    def __init__(self, model, poptorch_options) -> None:
        self.poptorch_options = poptorch_options
        self.wrapped_model = poptorch.inferenceModel(model.eval(), options=poptorch_options)

    def get_dataloader(self, dataset):
        return poptorch.DataLoader(dataset=dataset, batch_size=DATALOADER_BATCH_SIZE, options=self.poptorch_options)


class ModelComparer:
    def __init__(self, model, other_model, dataset) -> None:
        self.model = model
        self.other_model = other_model
        self.dataset = dataset

    def test(self):
        torch.testing.assert_close(self.model.evaluate(self.dataset), self.other_model.evaluate(self.dataset))


class SerializedLayerTester:
    @abstractmethod
    def setup_test(cls, ipu_model=False):
        ...

    @parameterized.expand([False, True])
    def test_model_equivalence(self, ipu_model):
        model, other_model, dataset = self.setup_test(ipu_model)
        model_comparer = ModelComparer(model, other_model, dataset=dataset)
        model_comparer.test()

    def test_to_model(self):
        model, other_model, _ = self.setup_test()
        torch.testing.assert_close(model.wrapped_model.weight, other_model.wrapped_model.to_model().weight)


class SplitProjectionTester(unittest.TestCase, SerializedLayerTester):
    @classmethod
    def setup_test(cls, ipu_model=False):
        num_samples = 32
        num_features = 32
        dataset = torch.randn((num_samples, num_features))

        # linear model to test SplitProjection against
        linear = torch.nn.Linear(in_features=num_features, out_features=8)
        evaluator = PytorchEvaluator(linear)

        serialized_projection_splits_per_ipu = [2, 2]
        projection_serialization_factor = sum(serialized_projection_splits_per_ipu)
        split_projection = SplitProjection.from_model(linear, projection_serialization_factor)

        # Split serialized linear layers across IPUs if using the IPU
        if ipu_model:
            options = poptorch.Options()
            options.deviceIterations(len(serialized_projection_splits_per_ipu))

            other_evaluator = PoptorchEvaluator(
                split_projection.parallelize(serialized_projection_splits_per_ipu),
                options,
            )
        else:
            other_evaluator = PytorchEvaluator(split_projection)

        return evaluator, other_evaluator, dataset


class SerializedLinearTester(unittest.TestCase, SerializedLayerTester):
    @classmethod
    def setup_test(cls, ipu_model=False):
        num_samples = 32
        num_features = 32
        dataset = torch.randn((num_samples, num_features))

        # linear model to test SerializedLinear against
        linear = torch.nn.Linear(in_features=num_features, out_features=8)
        evaluator = PytorchEvaluator(linear)

        serialization_factor = 4
        serialized_linear = SerializedLinear.from_model(linear, serialization_factor)

        if ipu_model:
            options = poptorch.Options()
            other_evaluator = PoptorchEvaluator(serialized_linear, options)
        else:
            other_evaluator = PytorchEvaluator(serialized_linear)

        return evaluator, other_evaluator, dataset


class SerializedEmbeddingTester(unittest.TestCase, SerializedLayerTester):
    @classmethod
    def setup_test(cls, ipu_model=False):
        num_samples = 32
        num_tokens = 32
        num_embeddings = 64
        embedding_dim = 8
        dataset = torch.randint(high=num_embeddings, size=(num_samples, num_tokens))

        # embedding layer to test against
        embedding = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        evaluator = PytorchEvaluator(embedding)

        serialized_embedding_splits_per_ipu = [3, 1]
        embedding_serialization_factor = sum(serialized_embedding_splits_per_ipu)
        serialized_embedding = SerializedEmbedding.from_model(embedding, embedding_serialization_factor)

        # Split serialized embedding layers across IPUs if using the IPU
        if ipu_model:
            options = poptorch.Options()
            options.deviceIterations(len(serialized_embedding_splits_per_ipu))

            other_evaluator = PoptorchEvaluator(
                serialized_embedding.parallelize(serialized_embedding_splits_per_ipu),
                options,
            )
        else:
            other_evaluator = PytorchEvaluator(serialized_embedding)

        return evaluator, other_evaluator, dataset

    @parameterized.expand(((1, 0, 1), (63, 3, 15)))
    def test_padding_idx(self, padding_idx, split, expected_padding_idx):
        embedding = torch.nn.Embedding(num_embeddings=64, embedding_dim=8, padding_idx=padding_idx)
        serialized_embedding = SerializedEmbedding.from_model(embedding, 4)
        assert serialized_embedding.split_embeddings[split].padding_idx == expected_padding_idx

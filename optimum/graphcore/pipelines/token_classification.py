import warnings
from typing import List, Optional, Tuple

from transformers import TokenClassificationPipeline
from transformers.pipelines.token_classification import AggregationStrategy


class IPUTokenClassificationPipeline(TokenClassificationPipeline):
    def _sanitize_parameters(
        self,
        ignore_labels=None,
        grouped_entities: Optional[bool] = None,
        ignore_subwords: Optional[bool] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        offset_mapping: Optional[List[Tuple[int, int]]] = None,
        stride: Optional[int] = None,
        **tokenizer_kwargs,
    ):
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(
            ignore_labels=ignore_labels,
            grouped_entities=grouped_entities,
            ignore_subwords=ignore_subwords,
            aggregation_strategy=aggregation_strategy,
            offset_mapping=offset_mapping,
            stride=stride,
        )
        if tokenizer_kwargs:
            preprocess_params["tokenizer_params"] = {**preprocess_params.get("tokenizer_params", {}), **tokenizer_kwargs}
        return preprocess_params, forward_params, postprocess_params

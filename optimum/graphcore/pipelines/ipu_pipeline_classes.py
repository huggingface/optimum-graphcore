import warnings
from typing import Dict, List, Optional, Tuple

from transformers import (
    AudioClassificationPipeline,
    AutomaticSpeechRecognitionPipeline,
    FillMaskPipeline,
    ImageClassificationPipeline,
    Pipeline,
    QuestionAnsweringPipeline,
    TextClassificationPipeline,
    TokenClassificationPipeline,
    ZeroShotClassificationPipeline,
)
from transformers.pipelines.base import PIPELINE_INIT_ARGS, GenericTensor, PipelineException


class IPUFillMaskPipeline(FillMaskPipeline):
    def _sanitize_parameters(self, top_k=None, targets=None, **tokenizer_kwargs):
        preprocess_params = tokenizer_kwargs

        postprocess_params = {}

        if targets is not None:
            target_ids = self.get_target_ids(targets, top_k)
            postprocess_params["target_ids"] = target_ids

        if top_k is not None:
            postprocess_params["top_k"] = top_k

        if self.tokenizer.mask_token_id is None:
            raise PipelineException(
                "fill-mask", self.model.base_model_prefix, "The tokenizer does not define a `mask_token`."
            )
        return preprocess_params, {}, postprocess_params

    def preprocess(self, inputs, return_tensors=None, **tokenizer_kwargs) -> Dict[str, GenericTensor]:
        if return_tensors is None:
            return_tensors = self.framework
        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors, **tokenizer_kwargs)
        self.ensure_exactly_one_mask_token(model_inputs)
        return model_inputs


from transformers.pipelines.token_classification import AggregationStrategy


class IPUTokenClassificationPipeline(TokenClassificationPipeline):
    def _sanitize_parameters(
        self,
        ignore_labels=None,
        grouped_entities: Optional[bool] = None,
        ignore_subwords: Optional[bool] = None,
        aggregation_strategy: Optional[AggregationStrategy] = None,
        offset_mapping: Optional[List[Tuple[int, int]]] = None,
        **tokenizer_kwargs,
    ):

        preprocess_params = tokenizer_kwargs
        if offset_mapping is not None:
            preprocess_params["offset_mapping"] = offset_mapping

        postprocess_params = {}
        if grouped_entities is not None or ignore_subwords is not None:
            if grouped_entities and ignore_subwords:
                aggregation_strategy = AggregationStrategy.FIRST
            elif grouped_entities and not ignore_subwords:
                aggregation_strategy = AggregationStrategy.SIMPLE
            else:
                aggregation_strategy = AggregationStrategy.NONE

            if grouped_entities is not None:
                warnings.warn(
                    "`grouped_entities` is deprecated and will be removed in version v5.0.0, defaulted to"
                    f' `aggregation_strategy="{aggregation_strategy}"` instead.'
                )
            if ignore_subwords is not None:
                warnings.warn(
                    "`ignore_subwords` is deprecated and will be removed in version v5.0.0, defaulted to"
                    f' `aggregation_strategy="{aggregation_strategy}"` instead.'
                )

        if aggregation_strategy is not None:
            if isinstance(aggregation_strategy, str):
                aggregation_strategy = AggregationStrategy[aggregation_strategy.upper()]
            if (
                aggregation_strategy
                in {AggregationStrategy.FIRST, AggregationStrategy.MAX, AggregationStrategy.AVERAGE}
                and not self.tokenizer.is_fast
            ):
                raise ValueError(
                    "Slow tokenizers cannot handle subwords. Please set the `aggregation_strategy` option"
                    'to `"simple"` or use a fast tokenizer.'
                )
            postprocess_params["aggregation_strategy"] = aggregation_strategy
        if ignore_labels is not None:
            postprocess_params["ignore_labels"] = ignore_labels
        return preprocess_params, {}, postprocess_params

    def preprocess(self, sentence, offset_mapping=None, **tokenizer_kwargs):
        truncation = True if self.tokenizer.model_max_length and self.tokenizer.model_max_length > 0 else False
        model_inputs = self.tokenizer(
            sentence,
            return_tensors=self.framework,
            truncation=truncation,
            return_special_tokens_mask=True,
            return_offsets_mapping=self.tokenizer.is_fast,
            **tokenizer_kwargs,
        )
        if offset_mapping:
            model_inputs["offset_mapping"] = offset_mapping

        model_inputs["sentence"] = sentence

        return model_inputs

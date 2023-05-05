#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import contextlib
import copy
import json
import os
import warnings
from typing import Any, Callable, Dict, Optional, Union

import torch
from torch import nn

import poptorch
from optimum.utils import logging
from transformers.generation.stopping_criteria import validate_stopping_criteria
from transformers.generation.utils import (
    BeamSampleDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput,
    BeamSampleOutput,
    BeamScorer,
    BeamSearchDecoderOnlyOutput,
    BeamSearchEncoderDecoderOutput,
    BeamSearchOutput,
    GenerationMixin,
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
    GreedySearchOutput,
    LogitsProcessorList,
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    SampleOutput,
    StoppingCriteriaList,
)
from transformers.modeling_outputs import BaseModelOutput, ModelOutput
from transformers.pytorch_utils import torch_int_div
from transformers.utils.versions import require_version

from .logits_process import IPULogitsProcessors
from .on_device_generation import (
    OnDeviceBeamSearch,
    OnDeviceGenerationModel,
    OnDeviceGenerationModelOutput,
    OnDeviceGreedySearch,
)


logger = logging.get_logger(__name__)

MODELS_SUPPORTING_KV_CACHE = set()


def supports_kv_cache(pipelined_cls):
    MODELS_SUPPORTING_KV_CACHE.add(pipelined_cls)
    return pipelined_cls


@contextlib.contextmanager
def graph_profile_dir_append(append: str):
    if poplar_engine_options_original := os.getenv("POPLAR_ENGINE_OPTIONS"):
        poplar_engine_options_modified = json.loads(poplar_engine_options_original)
        if autoreport_directory := poplar_engine_options_modified.get("autoReport.directory"):
            poplar_engine_options_modified["autoReport.directory"] = autoreport_directory + append
            os.environ["POPLAR_ENGINE_OPTIONS"] = json.dumps(poplar_engine_options_modified)
    try:
        yield
    finally:
        if poplar_engine_options_original:
            os.environ["POPLAR_ENGINE_OPTIONS"] = poplar_engine_options_original


class _IndexedInputLinear(nn.Module):
    """
    Wrapper layer for `Linear` that performs a `dynamic_slice` on the input
    before executing the linear. The intended use is as an optimized replacement of the
    LM Head in the Decoder for text generation inference when KV caching is disabled.
    The slice is performed on the position `self._generation_step` of the input tensor, where
    `self._generation_step` is a PyTorch buffer.
    """

    def __init__(self, linear_layer):
        super().__init__()
        self.wrapped_linear = linear_layer
        self.register_buffer("_generation_step", torch.tensor([0], dtype=torch.int32), persistent=False)

    def forward(self, x):
        x = poptorch.dynamic_slice(x, 1, self._generation_step, 1, 1)
        return self.wrapped_linear(x)


class DecoderWrapper(nn.Module):
    """
    Fast wrapper for decoder part of text generation models.

    Updates the appropriate buffers for the modules which need to know the current generation step.
    Only returns the logits from the last generated token to reduce IO costs.
    """

    def __init__(self, pipelined_model):
        super().__init__()
        self.pipelined_model = pipelined_model

        # With KV caching, some modules may need to know the current decoding step and beam indices.
        # Getting this information to them can either be done by copying it into buffers, or
        # by subclassing the entire decoder model just to change the forward signatures and passing these
        # as arguments. For now, go with the former, but it's not set in stone.
        self._modules_with_attributes_in_buffers = {
            attr: [module for module in self.pipelined_model.modules() if hasattr(module, attr)]
            for attr in ["_beam_idx", "_generation_step"]
        }

    def register_encoder_output_buffers(self, output_buffers: Dict[str, torch.Tensor]):
        for name in sorted(output_buffers):
            self.register_buffer(name, output_buffers[name], persistent=False)

    def _get_buffered_outputs(self) -> Dict:
        kwargs = {}
        if hasattr(self, "encoder_last_hidden_state"):
            kwargs["encoder_outputs"] = BaseModelOutput(last_hidden_state=self.encoder_last_hidden_state)
        if hasattr(self, "encoder_attention_mask"):
            kwargs["attention_mask"] = self.encoder_attention_mask
        return kwargs

    def forward(self, t, beam_idx=None, **model_inputs):
        """
        Args:
            t : (`torch.Tensor(int)`) Tensor with single int representing the current length of the sequence being generated
            beam_idx: (`torch.LongTensor` of shape `(batch_size * num_beams,)`):
                Beam indices indicating to which beam the tokens were added, required for reordering the on-device KV cache.
            model_inputs : Regular model_inputs passed to the wrapped model.
        Returns:
            The output logits at position `t` only
        """
        for module in self._modules_with_attributes_in_buffers["_generation_step"]:
            module._generation_step.copy_(t)

        # When generation is done on host, the beam_idx has to be provided as an input.
        # When generation is done on device, the beam_idx is stored in a separate buffer.
        if beam_idx is None:
            if hasattr(self.pipelined_model.generation_strategy, "_cached_beam_idx"):
                beam_idx = self.pipelined_model.generation_strategy._cached_beam_idx.int()
        for module in self._modules_with_attributes_in_buffers["_beam_idx"]:
            if beam_idx is None:
                raise ValueError(
                    "A module registered a `beam_idx` buffer, but the pipelined model is not called with such, "
                    "or the on device beam search did not register `_cached_beam_idx`. For the first case, "
                    "`beam_idx` can be provided to the model via `prepare_inputs_for_generation`."
                )
            module._beam_idx.copy_(beam_idx)

        # Run the decoder
        kwargs = self._get_buffered_outputs()
        outputs = self.pipelined_model(**model_inputs, **kwargs)
        if isinstance(outputs, ModelOutput) and not isinstance(outputs, OnDeviceGenerationModelOutput):
            outputs = type(outputs)(
                logits=outputs.logits,
            )
        return outputs


class IPUGenerationMixin(GenerationMixin):

    """
    Enable optimization for encoder-decoder text generation where the encoder outputs
    are cached on the Decoder device using buffers.
    """

    use_encoder_output_buffer = False

    @property
    def encoder_output_buffer_enabled(self) -> bool:
        return self.config.is_encoder_decoder and self.use_encoder_output_buffer

    def _pad_tensors_to_max_len(self, tensor: torch.Tensor, max_length: int, pad_token_id: int) -> torch.Tensor:
        return nn.functional.pad(tensor, (0, max_length - tensor.shape[1]), "constant", pad_token_id)

    def _ensure_generation_step_progression(self, generation_step):
        if not hasattr(self, "_previous_generation_step"):
            self._previous_generation_step = generation_step
            return
        if generation_step <= self._previous_generation_step and generation_step != 0:
            raise ValueError("`generation_step` must increase, or begin from 0.")
        self._previous_generation_step = generation_step

    def _call_generate(
        self,
        *args,
        generation_step: int,
        on_device_generation_model_ctr: Optional[Callable[[torch.nn.Module], torch.nn.Module]] = None,
        **kwargs,
    ):
        self._ensure_generation_step_progression(generation_step)
        t = self._get_generation_step_tensor(generation_step, ascending=on_device_generation_model_ctr is not None)
        if not hasattr(self, "poptorch_decoder"):
            generation_model = self
            if on_device_generation_model_ctr is not None:
                generation_model = on_device_generation_model_ctr(self)
            decoder_wrapper = DecoderWrapper(generation_model.eval())

            if os.getenv("DEBUG_RUN_DECODER_ON_CPU", False):
                self.poptorch_decoder = decoder_wrapper
            else:
                decoder_ipu_config = getattr(self, "decoder_ipu_config", self.ipu_config)
                decoder_options = decoder_ipu_config.to_options(for_inference=True)

                if self.encoder_output_buffer_enabled:
                    require_version(
                        "poptorch>=3.3", "Updatable encoder output buffer optimization only available in poptorch>=3.3"
                    )
                    named_buffers = {"encoder_last_hidden_state": kwargs["encoder_outputs"]["last_hidden_state"]}
                    if kwargs.get("attention_mask") is not None:
                        named_buffers["encoder_attention_mask"] = kwargs["attention_mask"].half()
                    decoder_wrapper.register_encoder_output_buffers(named_buffers)
                    decoder_options.updatableNamedBuffers(list(named_buffers.keys()))

                self.poptorch_decoder = poptorch.inferenceModel(decoder_wrapper, decoder_options)

        if self.encoder_output_buffer_enabled:
            kwargs.pop("encoder_outputs", None)
            kwargs.pop("attention_mask", None)

        # This will trigger a compile first time it's ran
        with graph_profile_dir_append("/decoder" if self.config.is_encoder_decoder else ""):
            return self.poptorch_decoder(*args, t=t, **kwargs)

    def _update_model_buffers_if_needed(self, model_kwargs):
        """
        If decoder model then we cache the encoder values inside pytorch buffers to reduce the IO cost
        """
        if not (self.encoder_output_buffer_enabled and hasattr(self, "poptorch_decoder")):
            return
        self.poptorch_decoder.encoder_last_hidden_state.copy_(model_kwargs["encoder_outputs"]["last_hidden_state"])
        if model_kwargs.get("attention_mask") is not None:
            self.poptorch_decoder.encoder_attention_mask.copy_(model_kwargs["attention_mask"].half())
        self.poptorch_decoder.copyNamedBuffersToDevice()

    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()

        # 2. prepare encoder args and encoder kwargs from model kwargs
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor

        if not hasattr(self, "poptorch_encoder"):
            # Use split encoder ipu_config for encoder/decoder models
            if os.getenv("DEBUG_RUN_ENCODER_ON_CPU", False):
                self.poptorch_encoder = encoder.eval()
            else:
                self.poptorch_encoder = poptorch.inferenceModel(
                    encoder.eval(), self.encoder_ipu_config.to_options(for_inference=True)
                )
        with graph_profile_dir_append("/encoder"):
            model_kwargs["encoder_outputs"]: ModelOutput = self.poptorch_encoder(**encoder_kwargs)

        return model_kwargs

    def detachFromDevice(self):
        if hasattr(self, "poptorch_encoder"):
            self.poptorch_encoder.detachFromDevice()
        if hasattr(self, "poptorch_decoder"):
            self.poptorch_decoder.detachFromDevice()

    def _get_generation_step_tensor(self, generation_step, ascending=False):
        # Returns a 1 dimensional tensor of the form [device_iterations * replication factor]
        # with all elements equal to generation_step.
        # This ensures the dimensions are as expected by any parallelism options.
        decoder_ipu_config = getattr(self, "decoder_ipu_config", self.ipu_config)
        per_replica = (
            torch.arange(decoder_ipu_config.inference_device_iterations) + generation_step
            if ascending
            else torch.ones(decoder_ipu_config.inference_device_iterations) * generation_step
        )
        return per_replica.repeat(decoder_ipu_config.inference_replication_factor)

    def change_lm_head_to_indexed_input_linear(self, restore: bool):
        """Changes the LM head with the faster _IndexedInputLinear layer.

        Args:
            restore: whether to restore the LM head to the original version or not.
        """
        if restore:
            lm_head = self.get_output_embeddings()
            if lm_head.__class__ == _IndexedInputLinear:
                self.set_output_embeddings(lm_head.wrapped_linear)
        else:
            self.set_output_embeddings(_IndexedInputLinear(self.get_output_embeddings()))

    # Modified from https://github.com/huggingface/transformers/blob/v4.20.1/src/transformers/generation_utils.py#L1532
    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_utils.GreedySearchDecoderOnlyOutput`], [`~generation_utils.GreedySearchEncoderDecoderOutput`]
            or `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        else:
            max_length = stopping_criteria.max_length
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        self._update_model_buffers_if_needed(model_kwargs)

        use_cache = model_kwargs.get("use_cache", False)
        if use_cache and self.__class__ not in MODELS_SUPPORTING_KV_CACHE:
            raise ValueError(
                f"{self.__class__} does not support KV caching. Pipelined models can be "
                "decorated using `supports_kv_cache` once they support static KV caching."
            )

        # Change: intercept to optionally run the entire generation loop on device
        if self.on_device_generation_steps > 0:
            return self._on_device_greedy_search(
                input_ids,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                max_length=max_length,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        while True:
            # Change: remove synced_gpu code
            # Change: add input max_length padding
            if not use_cache:
                input_ids = self._pad_tensors_to_max_len(input_ids, stopping_criteria.max_length, pad_token_id)

                # Change: For a seq2seq model such as BART, the "attention_mask" is the encoder/cross attention mask and it does not require padding.
                if not self.config.is_encoder_decoder:
                    model_kwargs["attention_mask"] = self._pad_tensors_to_max_len(
                        model_kwargs["attention_mask"], stopping_criteria.max_length, 0
                    )

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self._call_generate(
                generation_step=cur_len - 1,
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # Change: Remove padding and restore to actual length
            if not use_cache:
                input_ids = input_ids[:, :cur_len]
                if not self.config.is_encoder_decoder:
                    model_kwargs["attention_mask"] = model_kwargs["attention_mask"][:, :cur_len]

            # Change: remove synced_gpu code

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                # Change: remove synced_gpu code
                break
        # End of while True

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation_utilsBeamSearchDecoderOnlyOutput`], [`~generation_utils.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        else:
            max_length = stopping_criteria.max_length
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        use_cache = model_kwargs.get("use_cache", False)
        if use_cache and self.__class__ not in MODELS_SUPPORTING_KV_CACHE:
            raise ValueError(
                f"{self.__class__} does not support KV caching. Pipelined models can be "
                "decorated using `supports_kv_cache` once they support static KV caching."
            )

        self._update_model_buffers_if_needed(model_kwargs)

        # Change: intercept to optionally run the entire generation loop on device
        if self.on_device_generation_steps > 0:
            return self._on_device_beam_search(
                input_ids,
                beam_scorer=beam_scorer,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                max_length=max_length,
                return_dict_in_generate=return_dict_in_generate,
                **model_kwargs,
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            # Change: remove synced_gpu code
            # Change: add input max_length padding
            if not use_cache:
                input_ids = self._pad_tensors_to_max_len(input_ids, stopping_criteria.max_length, pad_token_id)

                # Change: For a seq2seq model such as BART, the "attention_mask" is the encoder/cross attention mask and it does not require padding.
                if not self.config.is_encoder_decoder:
                    model_kwargs["attention_mask"] = self._pad_tensors_to_max_len(
                        model_kwargs["attention_mask"], stopping_criteria.max_length, 0
                    )

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self._call_generate(
                generation_step=cur_len - 1,
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            # Change: Remove padding and restore to actual length
            if not use_cache:
                input_ids = input_ids[:, :cur_len]
                if not self.config.is_encoder_decoder:
                    model_kwargs["attention_mask"] = model_kwargs["attention_mask"][:, :cur_len]

            # Change: remove synced_gpu code

            # Change: cast to float on cpu
            next_token_logits = outputs.logits[:, -1, :].float()
            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)
            # Change: add beam_idx to model_kwargs so KV caching can be made aware of it on device
            model_kwargs["beam_idx"] = beam_idx

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_utils.SampleDecoderOnlyOutput`], [`~generation_utils.SampleEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.SampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(15, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.sample(
        ...     input_ids,
        ...     logits_processor=logits_processor,
        ...     logits_warper=logits_warper,
        ...     stopping_criteria=stopping_criteria,
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and a wonderful day.\n\nI was lucky enough to meet the']
        ```"""

        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        use_cache = model_kwargs.get("use_cache", False)
        if use_cache and self.__class__ not in MODELS_SUPPORTING_KV_CACHE:
            raise ValueError(
                f"{self.__class__} does not support KV caching. Pipelined models can be "
                "decorated using `supports_kv_cache` once they support static KV caching."
            )

        self._update_model_buffers_if_needed(model_kwargs)

        # Change: intercept to optionally run the entire generation loop on device
        if self.on_device_generation_steps > 0:
            return self._on_device_sample()

        # keep track of which sequences are already finished
        unfinished_sequences = input_ids.new(input_ids.shape[0]).fill_(1)
        cur_len = input_ids.shape[-1]

        # auto-regressive generation
        while True:
            # Change: remove synced_gpu code
            # Change: add input max_length padding
            if not use_cache:
                input_ids = self._pad_tensors_to_max_len(input_ids, stopping_criteria.max_length, pad_token_id)

                # Change: For a seq2seq model such as BART, the "attention_mask" is the encoder/cross attention mask and it does not require padding.
                if not self.config.is_encoder_decoder:
                    model_kwargs["attention_mask"] = self._pad_tensors_to_max_len(
                        model_kwargs["attention_mask"], stopping_criteria.max_length, 0
                    )

            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self._call_generate(
                generation_step=cur_len - 1,
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            # Change: Remove padding and restore to actual length
            if not use_cache:
                input_ids = input_ids[:, :cur_len]
                if not self.config.is_encoder_decoder:
                    model_kwargs["attention_mask"] = model_kwargs["attention_mask"][:, :cur_len]

            # Change: remove synced_gpu code

            # Change: cast to float on cpu
            next_token_logits = outputs.logits[:, -1, :].float()

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores.float(), dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            cur_len = cur_len + 1

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences.mul((next_tokens != eos_token_id).long())

            # stop when each sentence is finished, or if we exceed the maximum length
            if unfinished_sequences.max() == 0 or stopping_criteria(input_ids, scores):
                break

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        **model_kwargs,
    ) -> Union[BeamSampleOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search multinomial
        sampling** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:

            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`int`, *optional*):
                The id of the *end-of-sequence* token.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation_utils.BeamSampleDecoderOnlyOutput`], [`~generation_utils.BeamSampleEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation_utils.BeamSampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation_utils.BeamSampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     max_length=model.config.max_length,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> outputs = model.beam_sample(
        ...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        output_scores = output_scores if output_scores is not None else self.config.output_scores
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate if return_dict_in_generate is not None else self.config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        use_cache = model_kwargs.get("use_cache", False)
        if use_cache and self.__class__ not in MODELS_SUPPORTING_KV_CACHE:
            raise ValueError(
                f"{self.__class__} does not support KV caching. Pipelined models can be "
                "decorated using `supports_kv_cache` once they support static KV caching."
            )

        self._update_model_buffers_if_needed(model_kwargs)

        # Change: intercept to optionally run the entire generation loop on device
        if self.on_device_generation_steps > 0:
            return self._on_device_beam_sample()

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        while True:
            # Change: remove synced_gpu code
            # Change: add input max_length padding
            if not use_cache:
                input_ids = self._pad_tensors_to_max_len(input_ids, stopping_criteria.max_length, pad_token_id)

                # Change: For a seq2seq model such as BART, the "attention_mask" is the encoder/cross attention mask and it does not require padding.
                if not self.config.is_encoder_decoder:
                    model_kwargs["attention_mask"] = self._pad_tensors_to_max_len(
                        model_kwargs["attention_mask"], stopping_criteria.max_length, 0
                    )

            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self._call_generate(
                generation_step=cur_len - 1,
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            # Change: Remove padding and restore to actual length
            if not use_cache:
                input_ids = input_ids[:, :cur_len]
                if not self.config.is_encoder_decoder:
                    model_kwargs["attention_mask"] = model_kwargs["attention_mask"][:, :cur_len]

            # Change: remove synced_gpu code

            # Change: cast to float on cpu
            next_token_logits = outputs.logits[:, -1, :].float()

            # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
            # cannot be generated both before and after the `nn.functional.log_softmax` operation.
            next_token_logits = self.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (logits_warper(input_ids, next_token_scores_processed),)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = torch_int_div(next_tokens, vocab_size)
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            if model_kwargs["past"] is not None:
                model_kwargs["past"] = self._reorder_cache(model_kwargs["past"], beam_idx)
            # Change: add beam_idx to model_kwargs so KV caching can be made aware of it on device
            model_kwargs["beam_idx"] = beam_idx

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                break

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.config.is_encoder_decoder:
                return BeamSampleEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSampleDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

    on_device_generation_steps: int = 0

    def set_on_device_generation_steps(self, value: Optional[int] = 0):
        self.on_device_generation_steps = value
        if value == 0:
            del self.on_device_generation_steps

    def _adapt_logits_processor_for_on_device_generation(self, logits_processor: LogitsProcessorList, vocab_size: int):
        adapted_processors = []
        for processor in logits_processor:
            ipu_processor_cls = IPULogitsProcessors.get(processor.__class__, None)
            if ipu_processor_cls is None:
                raise NotImplementedError(f"{processor.__class__.__name__} is not supported yet to run on IPU.")

            try:
                ipu_processor = ipu_processor_cls.from_model(processor, vocab_size)
            except AttributeError:
                ipu_processor = copy.deepcopy(processor)
                ipu_processor.__class__ = ipu_processor_cls
            adapted_processors.append(ipu_processor)
        return LogitsProcessorList(adapted_processors)

    def _adapt_stopping_criteria_for_on_device_generation(
        self, stopping_criteria: StoppingCriteriaList, on_device_generation_steps: int
    ):
        adapted_stopping_criteria = []
        for stopping_criterion in stopping_criteria:
            if hasattr(stopping_criterion, "max_length"):
                max_length = stopping_criterion.max_length
                new_max_length = max_length - on_device_generation_steps
                logger.info(
                    f"Temporarily adapting `max_length` from {max_length} to {new_max_length} for on device generation."
                )
                stopping_criterion = copy.deepcopy(stopping_criterion)
                stopping_criterion.max_length = new_max_length
            adapted_stopping_criteria.append(stopping_criterion)
        return StoppingCriteriaList(adapted_stopping_criteria)

    def _prepare_inputs_for_on_device_generation(self, model_inputs, on_device_generation_steps, batch_size):
        """
        A model-agnostic version of `prepare_inputs_for_generation` whose main purpose is to duplicate
        decoder inputs by `on_device_generation_steps=inference_device_iterations` and perform additional input validation.
        Since we are duplicating tensors, we restrict duplication to `torch.Tensor` and the exceptional case of
        `encoder_outputs.last_hidden_state`.
        """
        adapted_model_inputs = {}
        for k, v in model_inputs.items():
            if k in ("attention_mask", "encoder_outputs") and self.encoder_output_buffer_enabled:
                # These inputs will copied onto device via buffers, so we don't need to duplicate them.
                adapted_model_inputs[k] = v
                continue
            if k == "beam_idx":
                # With on-device generation, beam_idx at each step is handled through buffers.
                continue

            if torch.is_tensor(v):
                if v.shape[0] != batch_size:
                    raise ValueError(f"Unexpected size in dim 0 for {k}, expected {batch_size}.")
                v = v.repeat(on_device_generation_steps, *(1 for _ in range(v.ndim - 1)))
            elif k == "encoder_outputs":
                v_type = type(v)
                if not isinstance(v, BaseModelOutput):
                    raise ValueError(
                        "Expected `encoder_outputs` to be an instance of `BaseModelOutput`, " f"received {v_type}."
                    )
                v = v.last_hidden_state
                v = v.repeat(on_device_generation_steps, *(1 for _ in range(v.ndim - 1)))
                v = v_type(last_hidden_state=v)
            elif v is None:
                pass
            elif isinstance(v, (int, float, str, bool)):
                pass
            else:
                raise TypeError(
                    f"Unexpected type {type(v)} received for decoder input {k}. On device generation enforces "
                    "stricter input validation to minimise unexpected errors. Improvements are always welcome."
                )
            adapted_model_inputs[k] = v
        return adapted_model_inputs

    def _on_device_greedy_search(
        self,
        input_ids: torch.Tensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        pad_token_id: int,
        eos_token_id: int,
        max_length: int,
        return_dict_in_generate: Optional[bool] = False,
        **model_kwargs,
    ):
        if not model_kwargs.get("use_cache", False):
            raise NotImplementedError("On device greedy search assumes `use_cache=True`.")

        if return_dict_in_generate:
            raise NotImplementedError("On device greedy search assumes `return_dict_in_generate=False`.")

        batch_size, context_length = input_ids.shape
        vocab_size = self.get_output_embeddings().out_features

        if context_length > 1:
            raise ValueError("Context length (input_ids.shape[-1]) > 1 is not supported yet.")

        if (max_length - context_length) % self.on_device_generation_steps != 0:
            logger.info(
                "`max_length - context_length` does not evenly divide `on_device_generation_steps` "
                f"({max_length - context_length} vs {self.on_device_generation_steps}). Generation will be done "
                f"{self.on_device_generation_steps} tokens at a time and stop short of `max_length` so as not to exceed it."
            )

        if self.ipu_config.inference_device_iterations != 1:
            raise ValueError(
                "On device generation expects `inference_device_iterations=1`, "
                f"received {self.ipu_config.inference_device_iterations}. "
                "For on device generation, `inference_device_iterations` will be set to "
                f"`on_device_generation_steps={self.on_device_generation_steps}`."
            )
        if hasattr(self, "decoder_ipu_config"):
            self.decoder_ipu_config.inference_device_iterations = self.on_device_generation_steps
        else:
            self.ipu_config.inference_device_iterations = self.on_device_generation_steps

        logits_processor = self._adapt_logits_processor_for_on_device_generation(logits_processor, vocab_size)
        stopping_criteria = self._adapt_stopping_criteria_for_on_device_generation(
            stopping_criteria, self.on_device_generation_steps
        )

        # This function only has to be called at the beginning of generation since
        # all necessary state should be stored in buffers on device.
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # A model-agnostic version of above mainly to duplicate inputs for device iterations.
        model_inputs = self._prepare_inputs_for_on_device_generation(
            model_inputs, self.on_device_generation_steps, batch_size
        )

        on_device_generation_model_ctr = lambda inst: OnDeviceGenerationModel(
            inst,
            batch_size=batch_size,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            logits_processor=logits_processor,
            num_beams=1,
            use_cache=True,
        )

        generation_step = 0
        while True:
            output = self._call_generate(
                generation_step=generation_step,  # NB: equal to `cur_len - 1` since context_length=1
                on_device_generation_model_ctr=on_device_generation_model_ctr,
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )
            next_tokens = output.generated_tokens.view(self.on_device_generation_steps, batch_size).T

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens], dim=-1)

            generation_step += self.on_device_generation_steps

            # stop when each sentence is finished, or if we exceed the maximum length
            if torch.any(output.done) or stopping_criteria(input_ids, ()):
                break

        return input_ids

    def _on_device_beam_search(
        self,
        input_ids: torch.Tensor,
        beam_scorer: BeamScorer,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        pad_token_id: int,
        eos_token_id: int,
        max_length: int,
        return_dict_in_generate: Optional[bool] = False,
        **model_kwargs,
    ):
        if not model_kwargs.get("use_cache", False):
            raise NotImplementedError("On device beam search assumes `use_cache=True`.")

        if return_dict_in_generate:
            raise NotImplementedError("On device beam search assumes `return_dict_in_generate=False`.")

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams
        batch_beam_size, context_length = input_ids.shape
        vocab_size = self.get_output_embeddings().out_features

        if context_length > 1:
            raise ValueError("Context length (input_ids.shape[-1]) > 1 is not supported yet.")

        if (max_length - context_length) % self.on_device_generation_steps != 0:
            logger.info(
                "`max_length - context_length` does not evenly divide `on_device_generation_steps` "
                f"({max_length - context_length} vs {self.on_device_generation_steps}). Generation will be done "
                f"{self.on_device_generation_steps} tokens at a time and stop short of `max_length` so as not to exceed it."
            )

        if self.ipu_config.inference_device_iterations != 1:
            raise ValueError(
                "On device generation expects `inference_device_iterations=1`, "
                f"received {self.ipu_config.inference_device_iterations}. "
                "For on device generation, `inference_device_iterations` will be set to "
                f"`on_device_generation_steps={self.on_device_generation_steps}`."
            )
        if hasattr(self, "decoder_ipu_config"):
            self.decoder_ipu_config.inference_device_iterations = self.on_device_generation_steps
        else:
            self.ipu_config.inference_device_iterations = self.on_device_generation_steps

        logits_processor = self._adapt_logits_processor_for_on_device_generation(logits_processor, vocab_size)
        stopping_criteria = self._adapt_stopping_criteria_for_on_device_generation(
            stopping_criteria, self.on_device_generation_steps
        )

        # This function only has to be called at the beginning of generation since
        # all necessary state should be stored in buffers on device.
        model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

        # A model-agnostic version of above mainly to duplicate inputs for device iterations.
        model_inputs = self._prepare_inputs_for_on_device_generation(
            model_inputs, self.on_device_generation_steps, batch_beam_size
        )

        on_device_generation_model_ctr = lambda inst: OnDeviceGenerationModel(
            inst,
            batch_size=batch_size,
            max_length=max_length,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            logits_processor=logits_processor,
            num_beams=num_beams,
            use_cache=True,
            length_penalty=beam_scorer.length_penalty,
            early_stopping=beam_scorer.do_early_stopping,
        )

        generation_step = 0
        while True:
            output = self._call_generate(
                generation_step=generation_step,  # NB: equal to `cur_len - 1` since context_length=1
                on_device_generation_model_ctr=on_device_generation_model_ctr,
                **model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
            )

            generation_step += self.on_device_generation_steps

            first_done = torch.argmax(output.done.int())

            input_ids = output.generated_tokens[
                first_done * batch_size : (first_done + 1) * batch_size, : context_length + generation_step
            ].to(input_ids.dtype)

            if torch.any(output.done) or stopping_criteria(input_ids, ()):
                break

        return input_ids

    def _on_device_sample(self):
        raise NotImplementedError("On device sampling is not supported.")

    def _on_device_beam_sample(self):
        raise NotImplementedError("On device beam sampling is not supported.")

import warnings

from transformers import SummarizationPipeline, Text2TextGenerationPipeline, TranslationPipeline
from transformers.pipelines.text2text_generation import ReturnType


class IPUText2TextGenerationPipeline(Text2TextGenerationPipeline):
    def _sanitize_parameters(
        self,
        return_tensors=None,
        return_text=None,
        return_type=None,
        clean_up_tokenization_spaces=None,
        truncation=None,
        stop_sequence=None,
        input_max_length=None,
        **generate_kwargs
    ):
        preprocess_params = {}
        if truncation is not None:
            preprocess_params["truncation"] = truncation

        if input_max_length is not None:
            preprocess_params["input_max_length"] = input_max_length

        forward_params = generate_kwargs

        postprocess_params = {}
        if return_tensors is not None and return_type is None:
            return_type = ReturnType.TENSORS if return_tensors else ReturnType.TEXT
        if return_type is not None:
            postprocess_params["return_type"] = return_type

        if clean_up_tokenization_spaces is not None:
            postprocess_params["clean_up_tokenization_spaces"] = clean_up_tokenization_spaces

        if stop_sequence is not None:
            stop_sequence_ids = self.tokenizer.encode(stop_sequence, add_special_tokens=False)
            if len(stop_sequence_ids) > 1:
                warnings.warn(
                    "Stopping on a multiple token sequence is not yet supported on transformers. The first token of"
                    " the stop sequence will be used as the stop sequence string in the interim."
                )
            generate_kwargs["eos_token_id"] = stop_sequence_ids[0]

        return preprocess_params, forward_params, postprocess_params

    def _parse_and_tokenize(self, *args, truncation, **kwargs):
        prefix = self.model.config.prefix if self.model.config.prefix is not None else ""
        if isinstance(args[0], list):
            if self.tokenizer.pad_token_id is None:
                raise ValueError("Please make sure that the tokenizer has a pad_token_id when using a batch input")
            args = ([prefix + arg for arg in args[0]],)
            padding = True

        elif isinstance(args[0], str):
            args = (prefix + args[0],)
            padding = False
        else:
            raise ValueError(
                f" `args[0]`: {args[0]} have the wrong format. The should be either of type `str` or type `list`"
            )
        padding = "max_length"
        inputs = self.tokenizer(
            *args,
            padding=padding,
            max_length=kwargs.get("input_max_length"),
            truncation=truncation,
            return_tensors=self.framework,
        )
        # This is produced by tokenizers but is an invalid generate kwargs
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
        return inputs


class IPUSummarizationPipeline(SummarizationPipeline, IPUText2TextGenerationPipeline):
    pass


class IPUTranslationPipeline(TranslationPipeline, IPUText2TextGenerationPipeline):
    pass

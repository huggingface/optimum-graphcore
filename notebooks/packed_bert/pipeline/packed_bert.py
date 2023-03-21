import logging
import time
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset

import poptorch
from models.modeling_bert_packed import (
    PipelinedPackedBertForQuestionAnswering,
    PipelinedPackedBertForSequenceClassification,
)
from optimum.graphcore import IPUConfig
from scipy.special import softmax
from transformers import AutoConfig, AutoTokenizer
from transformers.data.data_collator import default_data_collator
from utils.packing.dataset_creator import PackedDatasetCreator
from utils.packing.dataset_templates import PackedQuestionAnsweringDataset
from utils.packing.qa_utils import postprocess_packed_qa_predictions, preprocess_packed_qa


logger = logging.getLogger("")


def get_poplar_executor(model, ipu_config, batch, detach=False):
    ipu_options = ipu_config.to_options(for_inference=True)
    model.ipu_config = ipu_config

    if isinstance(model, poptorch.PoplarExecutor):
        print("Model already wrapped - nothing to do.")
        return model
    try:
        model.deparallelize()
    except:
        pass

    ipu_model = poptorch.inferenceModel(model.eval().parallelize(), ipu_options)

    ipu_model.compile(**batch)

    if detach:
        ipu_model.detachFromDevice()

    return ipu_model


def prepare_inference_dataloader(ipu_config, dataset, batch_size, mode="async_rebatched"):
    return poptorch.DataLoader(
        ipu_config.to_options(for_inference=True),
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Must be false, retained order important for batched inference
        drop_last=False,  # Must be false, we pad up to global batch size in inference pipeline to avoid any division error
        mode=mode,
        collate_fn=default_data_collator,
    )


class PackedBertTextClassificationPipeline:
    """
    Packed classification pipeline:

    Batched inference pipeline for packed BERT text classification with multi/single label. Wraps all preprocessing and model for inference, executes on text inputs in format `questions, contexts` of any size, proceeds to batch according to checkpoint or as per custom IPU configs, and packs data. Performs inference on PipelinedPackedBertForSequenceClassification. Returns postprocessed predictions in same order as input data.
    """

    def __init__(
        self,
        model,
        executable_cache_dir: str = "./exe_cache",
        problem_type: str = "single_label_classification",
        max_seq_per_pack: int = 12,
        max_seq_length: int = 384,
        ipu_config: IPUConfig = None,
        micro_batch_size: int = 1,
        dataloader_mode: str = "async_rebatched",
        detach_model_after_compile: bool = False,
        pretrained_tokenizer: str = "bert-base-uncased",
        label_categories: List = [],
    ) -> None:
        self.model_ckpt = model
        self.problem_type = problem_type
        self.max_seq_per_pack = max_seq_per_pack
        self.max_seq_length = max_seq_length

        self.pretrained_tokenizer = pretrained_tokenizer
        self.dataloader_mode = dataloader_mode
        self.detach_model_after_post_compile = detach_model_after_compile
        self.executable_cache_dir = executable_cache_dir

        self.micro_batch_size = micro_batch_size
        self.sentence_2_key = None
        self.label_categories = label_categories

        if not ipu_config:
            try:
                logger.info("Attempting loading IPUConfig from model checkpoint:")
                self.ipu_config = IPUConfig.from_pretrained(
                    self.model_ckpt, executable_cache_dir=self.executable_cache_dir
                )
            except:
                logger.warn(
                    "Loading default config: 'Graphcore/bert-base-uncased' - because no IPUConfig found in model folder."
                )
                self.ipu_config = IPUConfig.from_pretrained(
                    "Graphcore/bert-base-uncased", executable_cache_dir=self.executable_cache_dir
                )
        else:
            self.ipu_config = ipu_config

        self.gbs = (
            self.ipu_config.inference_device_iterations
            * self.ipu_config.inference_replication_factor
            * self.micro_batch_size
        )

        try:
            logger.info("Attempting loading tokenizer from model checkpoint")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt, use_fast=True)
        except:
            logger.warn("Loading tokenizer from defined because no pretrained tokenizer found in model folder.")
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_tokenizer, use_fast=True)

        config = AutoConfig.from_pretrained(self.model_ckpt)
        config.max_sequences_per_pack = self.max_seq_per_pack
        config.problem_type = self.problem_type

        self.model = (
            PipelinedPackedBertForSequenceClassification(config).from_pretrained(self.model_ckpt, config=config).half()
        )

        compile_data = Dataset.from_dict({"text": ["I am a dummy sentence for compilation."]})

        enc_compile_data = compile_data.map(self.preprocess_function, batched=True)

        pck_compile_data = PackedDatasetCreator(
            tokenized_dataset=enc_compile_data,
            max_sequence_length=self.max_seq_length,
            max_sequences_per_pack=self.max_seq_per_pack,
            inference=True,
            pad_to_global_batch_size=True,
            global_batch_size=self.gbs,
            problem_type=self.problem_type,
        ).create()

        c_dataloader = prepare_inference_dataloader(
            self.ipu_config, pck_compile_data, self.micro_batch_size, self.dataloader_mode
        )

        c_batch = next(iter(c_dataloader))

        # Remove custom column for compile - autoignored in optimum, manually ignored in predict
        c_batch.pop("example_ids", None)

        self.poplar_executor = get_poplar_executor(self.model, self.ipu_config, c_batch)

    def preprocess_function(self, examples):
        if self.sentence_2_key:
            return self.tokenizer(
                examples["text"], examples["text_2"], truncation=True, max_length=self.max_seq_length
            )
        else:
            return self.tokenizer(examples["text"], truncation=True, max_length=self.max_seq_length)

    def postprocess_preds(self, logits, ids):
        ids = torch.concat(ids)
        mask = ids != -100
        ids = ids[mask]

        if self.problem_type == "multi_label_classification":
            pred_scores = softmax(torch.concat(logits)[mask, :].numpy().astype("float32"), axis=1)
        if self.problem_type == "single_label_classification":
            pred_scores = softmax(torch.concat(logits)[mask, :].numpy().astype("float32"), axis=1)

        pred_scores = pred_scores[np.argsort(ids)]

        return pred_scores

    def predict(self, sentence_1, sentence_2=None):
        self.sentence_2_key = sentence_2

        prep_st = time.time()

        data_dict = {"text": sentence_1}
        if sentence_2:
            data_dict["text_2"] = sentence_2

        dataset = Dataset.from_dict(data_dict)
        enc_data = dataset.map(self.preprocess_function, batched=True)

        # Pack the inputs
        packed_data = PackedDatasetCreator(
            tokenized_dataset=enc_data,
            max_sequence_length=self.max_seq_length,
            max_sequences_per_pack=self.max_seq_per_pack,
            inference=True,
            pad_to_global_batch_size=True,
            global_batch_size=self.gbs,
            problem_type=self.problem_type,
        ).create()

        dataloader = prepare_inference_dataloader(
            self.ipu_config, packed_data, self.micro_batch_size, self.dataloader_mode
        )

        example_ids = []
        outputs = []

        # Process the model to return logits
        prep_time = time.time() - prep_st

        model_st = time.time()
        for batch in iter(dataloader):
            logits = self.poplar_executor(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
                position_ids=batch["position_ids"],
            )

            ids = batch["example_ids"]
            outputs.append(logits.view(ids.shape[0], self.max_seq_per_pack, -1))
            example_ids.append(ids)

        model_en = time.time()
        model_time = model_en - model_st
        tput = len(sentence_1) / (model_time)

        # Postprocess predictions to preserve order
        post_st = time.time()
        final_preds = self.postprocess_preds(outputs, example_ids)

        if len(self.label_categories) == final_preds.shape[-1]:
            final_preds = {k: dict(list(zip(self.label_categories, v))) for k, v in enumerate(final_preds)}
        else:
            final_preds = {{n: k[n] for n in k} for k in final_preds}

        post_proc_time = time.time() - post_st

        return {
            "predictions": final_preds,
            "throughput": tput,
            "inference_total_time": model_time,
            "preprocessing_time": prep_time,
            "postprocessing_time": post_proc_time,
        }


class PackedBertQuestionAnsweringPipeline:
    """
    Packed Question-answering pipeline:

    Batched inference pipeline for packed BERT question answering. Wraps all preprocessing and model for inference, executes on text inputs in format `questions, contexts` of any size, proceeds to batch according to checkpoint or as per custom IPU configs, and packs data. Performs inference on PipelinedPackedBertForQuestionAnswering. Returns postprocessed predictions in same order as input data.
    """

    def __init__(
        self,
        model,
        executable_cache_dir: str = "./exe_cache",
        problem_type: str = "question_answering",
        max_seq_per_pack: int = 12,
        max_seq_length: int = 384,
        pretrained_tokenizer: str = "bert-base-uncased",
        ipu_config: str = None,
        micro_batch_size: int = 1,
        dataloader_mode: str = "async_rebatched",
        detach_model_after_compile: bool = False,
    ) -> None:
        self.problem_type = problem_type
        self.max_seq_per_pack = max_seq_per_pack
        self.max_seq_length = max_seq_length

        self.model_ckpt = model
        self.pretrained_tokenizer = pretrained_tokenizer
        self.dataloader_mode = dataloader_mode
        self.detach_model_after_post_compile = detach_model_after_compile
        self.executable_cache_dir = executable_cache_dir
        self.micro_batch_size = micro_batch_size

        if not ipu_config:
            try:
                logger.info("Attempting loading IPUConfig from model checkpoint:")
                self.ipu_config = IPUConfig.from_pretrained(
                    self.model_ckpt, executable_cache_dir=self.executable_cache_dir
                )
            except:
                logger.warn(
                    "Loading default config: 'Graphcore/bert-base-uncased' - because no IPUConfig found in model folder."
                )
                self.ipu_config = IPUConfig.from_pretrained(
                    "Graphcore/bert-base-uncased", executable_cache_dir=self.executable_cache_dir
                )
        else:
            self.ipu_config = ipu_config

        self.gbs = (
            self.ipu_config.inference_device_iterations
            * self.ipu_config.inference_replication_factor
            * self.micro_batch_size
        )

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt, use_fast=True)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_tokenizer, use_fast=True)

        config = AutoConfig.from_pretrained(self.model_ckpt)
        config.max_sequences_per_pack = self.max_seq_per_pack
        config.problem_type = self.problem_type

        self.model = (
            PipelinedPackedBertForQuestionAnswering(config).from_pretrained(self.model_ckpt, config=config).half()
        )

        compile_data = Dataset.from_dict(
            {
                "id": np.array([str(i) for i in range(self.gbs)]).astype("<U32"),
                "question": ["Do trees have leaves in the wintertime?."] * self.gbs,
                "context": [
                    "Most trees leaves fall off after the autumn season. However, evergreen trees keep their leaves through winter."
                ]
                * self.gbs,
            }
        )

        enc_compile_data = preprocess_packed_qa(
            dataset=compile_data,
            tokenizer=self.tokenizer,
            question_key="question",
            context_key="context",
            answer_key="answer",
            sequence_length=self.max_seq_length,
            padding=True,  # only for compile, so we dont need to pack the dummy data
            train=False,
        )

        packed_compile_data_pre = PackedDatasetCreator(
            tokenized_dataset=enc_compile_data,
            max_sequence_length=self.max_seq_length,
            max_sequences_per_pack=self.max_seq_per_pack,
            inference=True,
            pad_to_global_batch_size=True,
            global_batch_size=self.gbs,
            problem_type=self.problem_type,
        ).create()

        packed_compile_data = Dataset.from_list(packed_compile_data_pre)
        packed_compile_data = packed_compile_data.remove_columns(["offset_mapping", "example_ids"])

        c_dataloader = prepare_inference_dataloader(
            self.ipu_config, packed_compile_data, self.micro_batch_size, self.dataloader_mode
        )

        c_batch = next(iter(c_dataloader))
        c_batch.pop("offset_mapping", None)
        c_batch.pop("example_id", None)

        self.poplar_executor = get_poplar_executor(self.model, self.ipu_config, c_batch)

    def predict(self, questions, contexts):
        prep_st = time.time()

        dataset = Dataset.from_dict(
            {
                "id": np.array([str(i) for i in range(len(questions))]).astype("<U32"),
                "question": questions,
                "context": contexts,
            }
        )

        enc_data = preprocess_packed_qa(
            dataset=dataset,
            tokenizer=self.tokenizer,
            question_key="question",
            context_key="context",
            answer_key="answer",
            sequence_length=self.max_seq_length,
            padding=False,
            train=False,
        )

        packed_data_pre = PackedDatasetCreator(
            tokenized_dataset=enc_data,
            max_sequence_length=self.max_seq_length,
            max_sequences_per_pack=self.max_seq_per_pack,
            inference=True,
            pad_to_global_batch_size=True,
            global_batch_size=self.gbs,
            problem_type=self.problem_type,
        ).create()

        # Not the most efficient way...
        packed_data = Dataset.from_list(packed_data_pre)
        packed_data = packed_data.remove_columns(["offset_mapping", "example_ids"])
        packed_data = PackedQuestionAnsweringDataset(
            input_ids=packed_data["input_ids"],
            attention_mask=packed_data["attention_mask"],
            token_type_ids=packed_data["token_type_ids"],
            position_ids=packed_data["position_ids"],
            start_positions=None,
            end_positions=None,
            offset_mapping=None,
            example_ids=None,
        )

        dataloader = prepare_inference_dataloader(
            self.ipu_config, packed_data, self.micro_batch_size, self.dataloader_mode
        )

        outputs = []
        prep_time = time.time() - prep_st

        model_st = time.time()
        for batch in iter(dataloader):
            logits = self.poplar_executor(**batch)
            outputs.append(torch.stack(logits))

        model_en = time.time()
        model_time = model_en - model_st
        tput = len(questions) / (model_time)

        post_st = time.time()
        outputs = torch.cat(outputs, dim=1).numpy()
        final_preds = postprocess_packed_qa_predictions(dataset, packed_data_pre, outputs)

        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_preds.items()]

        post_proc_time = time.time() - post_st

        return {
            "predictions": formatted_predictions,
            "throughput": tput,
            "inference_total_time": model_time,
            "preprocessing_time": prep_time,
            "postprocessing_time": post_proc_time,
        }

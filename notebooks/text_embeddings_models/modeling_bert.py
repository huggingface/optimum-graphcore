import poptorch
import logging

from transformers.models.bert.modeling_bert import BertModel, BertSelfAttention
from optimum.graphcore.models.bert.bert_fused_attention import BertFusedSelfAttention
from optimum.graphcore.modeling_utils import (
    SerializedEmbedding,
    get_layer_ipu,
    outline_attribute,
    recomputation_checkpoint,
    PipelineMixin,
    register
)

logger = logging.getLogger("")

@register(BertModel)
class PipelinedBertModel(BertModel, PipelineMixin):
    def __init__(self, config):
        super().__init__(config)

    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline.
        - Adds pipeline stages to the model
        - Replaces self-attention layers with fused-qkv self-attention layers
        - (If enabled) Replaces the word embedding with a SerializedEmbedding
        - Adds recomputation checkpoints
        """
        super().parallelize()

        # Use faster fused-qkv self-attention
        for layer in self.encoder.layer:
            layer.attention.self.__class__ = BertFusedSelfAttention

        logger.info("-------------------- Device Allocation --------------------")
        logger.info("Embedding --> IPU 0")
        if self.ipu_config.embedding_serialization_factor > 1:
            self.embeddings.word_embeddings = SerializedEmbedding.from_model(
                self.embeddings.word_embeddings, self.ipu_config.embedding_serialization_factor
            )
        self.embeddings = poptorch.BeginBlock(self.embeddings, "Embedding", ipu_id=0)
        hs = outline_attribute(self.embeddings.LayerNorm, "embedding")
        self._hooks.extend(hs)

        layer_ipu = get_layer_ipu(self.ipu_config, self.encoder.layer)
        for index, layer in enumerate(self.encoder.layer):
            ipu = layer_ipu[index]
            if self.ipu_config.recompute_checkpoint_every_layer and index != self.config.num_hidden_layers - 1:
                h = recomputation_checkpoint(layer)
                self._hooks.append(h)
            self.encoder.layer[index] = poptorch.BeginBlock(layer, f"Encoder{index}", ipu_id=ipu)
            logger.info(f"Encoder {index:<2} --> IPU {ipu}")

        logger.info("Pooler --> IPU 0")
        self.pooler = poptorch.BeginBlock(self.pooler, "Pooler", ipu_id=0)

        return self

    def deparallelize(self):
        """
        Undo the changes to the model done by `parallelize`.
        You should call this before doing `save_pretrained` so that the `model.state_dict` is
        compatible with the original model.
        """
        super().deparallelize()

        for layer in self.encoder.layer:
            layer.attention.self.__class__ = BertSelfAttention

        # Deserialize the serialized word embedding
        if self.ipu_config.embedding_serialization_factor > 1:
            self.embeddings.word_embeddings = self.embeddings.word_embeddings.to_model()
        return self

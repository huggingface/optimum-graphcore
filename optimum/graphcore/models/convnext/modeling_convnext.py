import poptorch
import transformers
from optimum.utils import logging
from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register

logger = logging.get_logger(__name__)

class ConvNextPipelineMixin(PipelineMixin):
    def parallelize(self):
        """Transform the model into an IPU pipeline"""
        super().parallelize()
        logger.info("---------- Device Allocation -----------")
        # Set embedding pipeline mapping
        logger.info(f"Embedding  --> IPU 0")
        self.convnext.embeddings = poptorch.BeginBlock(self.convnext.embeddings, "Embedding", ipu_id=0)
        
        #Set encoder pipeline mappings
        encoder_layer_ipu = get_layer_ipu(self.ipu_config.layers_per_ipu)
        for idx, layer in enumerate(self.convnext.encoder.stages):
            ipu_id = encoder_layer_ipu[idx]
            logger.info(f"Encoder stage {idx} --> IPU {ipu_id}")
            self.convnext.encoder.stages[idx] = poptorch.BeginBlock(layer, f"Encoder_stage_{idx}", ipu_id)
        
        return self

@register(transformers.ConvNextForImageClassification)
class PipelinedConvNextForImageClassification(transformers.ConvNextForImageClassification, ConvNextPipelineMixin):
    def parallelize(self):
        """Set pipeline mapping for the head (layernorm + classifier layers)"""
        super().parallelize()
        last_ipu = self.ipu_config.ipus_per_replica - 1
        logger.info(f"Head --> IPU {last_ipu}")
        self.convnext.layernorm = poptorch.BeginBlock(self.convnext.layernorm, "LayerNorm", last_ipu)
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", last_ipu)
        
        return self  
    
    def forward(self, pixel_values=None, labels=None, output_hidden_states=None, return_dict=None):
        return super().forward(pixel_values=pixel_values, labels=labels, output_hidden_states=output_hidden_states, return_dict=False)
        
    
    

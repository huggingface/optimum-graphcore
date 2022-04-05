import poptorch
import transformers
from optimum.utils import logging
from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register

logger = logging.get_logger(__name__)



class ConvNextPipelineMixin(PipelineMixin):
    def parallelize(self):
        """Transform the model into an IPU pipeline"""
        super().parallelize()
        #TODO: Use recomputation checkpoints? 
        # ConvNext has 6 'high-level' layers/stages: Embedding -> Encoder stage 0 -> ... -> Encoder stage 3 -> Head (layernorm + classifier)
        # ipu_config.layers_per_ipu sets how many of these high-level layers belong to each IPU
        # e.g. layers_per_ipu = [6] --> All layers on IPU0
        # e.g. layers_per_ipu = [3,3] ---> Embedding + encoder stage 0 & 1 on IPU0, encoder stage 2 & 3 + head on IPU1
        stages_per_ipu = self.ipu_config.layers_per_ipu
        stages_on_ipu = get_layer_ipu(stages_per_ipu)
        #self.stages_on_ipu = stages_on_ipu
        logger.info("---------- Device Allocation -----------")
        # Set embedding pipeline mapping
        embedding_ipu_id = stages_on_ipu[0]
        logger.info(f"Embedding  --> IPU {embedding_ipu_id}")
        self.convnext.embeddings = poptorch.BeginBlock(self.convnext.embeddings, "Embedding", embedding_ipu_id)
        
        #Set encoder pipeline mappings
        encoder_stages_ipu_ids = stages_on_ipu[1:5]
        for idx, layer in enumerate(self.convnext.encoder.stages):
            ipu_id = encoder_stages_ipu_ids[idx]
            logger.info(f"Encoder stage {idx} --> IPU {ipu_id}")
            self.convnext.encoder.stages[idx] = poptorch.BeginBlock(layer, f"Encoder_stage_{idx}", ipu_id)
        
        return self
        
@register(transformers.ConvNextModel)
class PipelinedConvNextModel(transformers.ConvNextModel, ConvNextPipelineMixin):        
    def parallelize(self):
        #TODO: Do we need this method?
        return super().parallelize()
        
    def deparallelize(self):
        #TODO: Do we need this method?
        return super().deparallelize()
    
    def forward(self, pixel_values=None, labels=None, output_hidden_states=None, return_dict=None):
        return super().forward(pixel_values=pixel_values, labels=labels, output_hidden_states=output_hidden_states, return_dict=False)
    
@register(transformers.ConvNextForImageClassification)
class PipelinedConvNextForImageClassification(transformers.ConvNextForImageClassification, ConvNextPipelineMixin):
    def parallelize(self):
        #TODO: Use recomputation checkpoints?
        super().parallelize()
        """Set pipeline mapping for the head (layernorm + classifier layers)"""
        head_ipu_id = get_layer_ipu(self.ipu_config.layers_per_ipu)[-1]
        logger.info(f"Head --> IPU {head_ipu_id}")
        self.convnext.layernorm = poptorch.BeginBlock(self.convnext.layernorm, "LayerNorm", head_ipu_id)
        self.classifier = poptorch.BeginBlock(self.classifier, "Classifier", head_ipu_id)
        
        return self  
    
    def deparallelize(self):
        #TODO: Do we need this method?
        return super().deparallelize()
    
    def forward(self, pixel_values=None, labels=None, output_hidden_states=None, return_dict=None):
        #TODO: Do we need this method?
        return super().forward(pixel_values=pixel_values, labels=labels, output_hidden_states=output_hidden_states, return_dict=False)
        
    
    

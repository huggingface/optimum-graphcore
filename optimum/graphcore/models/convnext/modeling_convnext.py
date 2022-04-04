import poptorch
import transformers
from optimum.utils import logging
from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register

logger = logging.get_logger(__name__)

@register(transformers.ConvNextModel)
class PipelinedConvNextModel(transformers.ConvNextModel, PipelineMixin):
    def _add_begin_block(self, module, name, ipu_id):
        module = poptorch.BeginBlock(module, name, ipu_id)
        
    def parallelize(self):
        return super().parallelize()
    
    def deparallelize(self):
        return super().deparallelize()
    
    def forward(self, pixel_values=None, output_hidden_states=None, return_dict=None):
        return super.forward(pixel_values=pixel_values, output_hidden_states=output_hidden_states, return_dict=False)


@register(transformers.ConvNextForImageClassification)
class PipelinedConvNextForImageClassification(transformers.ConvNextForImageClassification, PipelineMixin):
    
    def _add_begin_block(self, module, name, ipu_id):
        module = poptorch.BeginBlock(module, name, ipu_id)
        
    def parallelize(self):
        """
        Transform the model to run in an IPU pipeline
        """
        stages_per_ipu = self.ipu_config.layers_per_ipu
        num_ipus = len(stages_per_ipu)
        encoder_stage_on_ipu = get_layer_ipu(stages_per_ipu) 
        super().parallelize()
        logger.info("---------- Device Allocation -----------")
        #TODO: Use recomputation checkpoints? 
        
        # ConvNext has 4 encoder stages, and ipu_config.layers_per_ipu sets how many of these stages belong to each IPU
        # e.g. layers_per_ipu = [4] -> all 4 stages on IPU 0. 
        # e.g. layers_per_ipu = [2,2] -> stages 0 and 1 on IPU0, stages 2 and 3 on IPU1.
        # The layers before the first encoder stage will belong to the same IPU as the first encoder stage automatically 
        # The layers after the 4th encoder stage will belong to the same IPU as the 4th encoder stage automatically 
        for idx, stage_layer in enumerate(self.convnext.encoder.stages):
            ipu_id = encoder_stage_on_ipu[idx]
            logger.info(f"Encoder stage {idx} --> IPU {ipu_id}")
            self._add_begin_block(self.convnext.encoder.stages[idx], name=f"Encoder_stage_{idx}", ipu_id=ipu_id)
        
        return self
    
    def deparallelize(self):
        #TODO: Do we need this method?
        return super().deparallelize()
    
    def forward(self, pixel_values=None, labels=None, output_hidden_states=None, return_dict=None):
        return super().forward(pixel_values=pixel_values, labels=labels, output_hidden_states=output_hidden_states, return_dict=False)

    
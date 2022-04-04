from turtle import forward
import poptorch
import transformers
from ...modeling_utils import PipelineMixin, get_layer_ipu, recomputation_checkpoint, register


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
        super().parallelize()
        #TODO: Pipeline the model over multiple IPUs in an efficient way
        #TODO: Add logging (see other models as an example)
        
        self._add_begin_block(self.classifier, name="Classifier", ipu_id=1)
        
        return self
    
    def deparallelize(self):
        #TODO: Do we need this method?
        return super().deparallelize()
    
    def forward(self, pixel_values=None, labels=None, output_hidden_states=None, return_dict=None):
        return super().forward(pixel_values=pixel_values, labels=labels, output_hidden_states=output_hidden_states, return_dict=False)
    


        


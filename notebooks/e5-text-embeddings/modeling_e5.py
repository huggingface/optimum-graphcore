import torch
import torch.nn.functional as F
import logging
from typing import Optional, List

from optimum.graphcore.models.bert.modeling_bert import BertPipelineMixin
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel

logger = logging.getLogger("e5")

class PipelinedE5Model(BertPreTrainedModel, BertPipelineMixin):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.pool_type = config.pool_type
    
    def pool(
        self, 
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        ) -> torch.Tensor:
             
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    
        if self.pool_type == "avg":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pool_type == "cls":
            emb = last_hidden[:, 0]
        else:
            raise ValueError(f"pool_type {pool_type} not supported")
            
        return emb
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None
        ) -> torch.Tensor:

        outputs = self.bert(
            input_ids, 
            attention_mask, 
            token_type_ids, 
            position_ids, 
            head_mask, 
            inputs_embeds, 
            encoder_hidden_states, 
            encoder_attention_mask, 
            past_key_values, 
            use_cache, 
            output_attentions, 
            output_hidden_states, 
            return_dict
        )

        embeds = self.pool(outputs.last_hidden_state, attention_mask)
        embeds = F.normalize(embeds, p=2, dim=-1)

        return embeds
        

        




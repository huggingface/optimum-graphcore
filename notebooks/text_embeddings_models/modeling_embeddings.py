import torch
import torch.nn.functional as F
import logging
from typing import Optional, List

from transformers import AutoModel
from optimum.graphcore.modeling_utils import to_pipelined

logger = logging.getLogger("e5")

class IPUEmbeddingsModel:
    def __init__(self, model_name, model_config, ipu_config, fp16=True):
        self.model = AutoModel.from_pretrained(model_name, config=model_config)
        self.model = to_pipelined(self.model, ipu_config)
        self.model = self.model.parallelize()
        if fp16: self.model = self.model.half()

        self.pool_type = self.model_config.pool_type()
    
    def pool(
        self, 
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
        ) -> torch.Tensor:
             
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    
        if self.pool_type == "avg":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pool_type == "cls":
            emb = last_hidden[:, 0]
        else:
            raise ValueError(f"pool_type {self.pool_type} not supported")

        return emb
    
    def forward(self, **kwargs) -> torch.Tensor:

        outputs = self.model(**kwargs)
        embeds = self.pool(outputs.last_hidden_state, kwargs["attention_mask"], pool_type='avg')
        embeds = F.normalize(embeds, p=2, dim=-1)

        return embeds


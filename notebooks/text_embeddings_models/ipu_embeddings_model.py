import logging

import torch

from optimum.graphcore import IPUConfig
from optimum.graphcore.modeling_utils import to_pipelined


logger = logging.getLogger("e5")

class IPUEmbeddingsModel(torch.nn.Module):
    def __init__(self, model, ipu_config: IPUConfig, fp16=True):
        super().__init__()

        self.encoder = to_pipelined(model, ipu_config)
        self.encoder = self.encoder.parallelize()
        if fp16: self.encoder = self.encoder.half()

    def pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor, pool_type: str) -> torch.Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if pool_type == "avg":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif pool_type == "cls":
            emb = last_hidden[:, 0]
        else:
            raise ValueError(f"pool_type {pool_type} not supported")

        return emb

    def forward(self, pool_type: str ='avg', **kwargs) -> torch.Tensor:
        outputs = self.encoder(**kwargs)

        embeds = self.pool(outputs.last_hidden_state, kwargs["attention_mask"], pool_type=pool_type)
        embeds = torch.nn.functional.normalize(embeds, p=2, dim=-1)

        return embeds

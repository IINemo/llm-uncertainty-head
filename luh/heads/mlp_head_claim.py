import torch
import torch.nn as nn
import torch.nn.functional as F

from .uncertainty_head_base import UncertaintyHeadBase

import logging

log = logging.getLogger()



class MLPClaimHead(UncertaintyHeadBase):
    def __init__(
        self,
        feature_extractor,
        internal_dim1=256,
        internal_dim2=128,
        internal_dim3=64,
        cfg = None,  # Temporary we save initializing cfg in the head itself
    ):
        super().__init__(feature_extractor, cfg=cfg, model_type="claim")

        log.info(f"Feature dim: {feature_extractor.feature_dim()}")
        self.layers = nn.ModuleList([nn.Linear(feature_extractor.feature_dim(), internal_dim1),
                                     nn.ReLU(),
                                     nn.Linear(internal_dim1, internal_dim2),
                                     nn.ReLU(),
                                     nn.Linear(internal_dim2, internal_dim3),
                                     nn.ReLU(),
                                     nn.Linear(internal_dim3, 1)])

    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        claims = llm_inputs["claims"]
        features = X.to(torch.float32)

        results = []
        batch_size = len(claims)
        for i in range(batch_size):
            entity_mask = claims[i]
            if len(entity_mask) == 0:
                continue
            
            # First process features through MLP
            batch_features = features[i, :]
            processed_features = batch_features
            for layer in self.layers[:-1]:  # Apply all layers except final linear
                processed_features = layer(processed_features)
            
            # Then average across tokens for each claim
            sum_entity_embeds = (processed_features * entity_mask.unsqueeze(-1)).sum(dim=1)  
            count_entity_tokens = entity_mask.sum(dim=1).clamp(min=1)
            entity_representation = sum_entity_embeds / count_entity_tokens.unsqueeze(-1)
            
            # Final linear layer
            out = self.layers[-1](entity_representation)
            results.append(out)
        
        # Padding to ensure uniform output shape
        max_entities_per_batch = max([o.shape[0] for o in results], default=1)
        padded_results = [F.pad(o, (0, 0, 0, max_entities_per_batch - o.shape[0]), value=-100) for o in results]
        
        return torch.stack(padded_results) if len(padded_results) else torch.zeros(0)

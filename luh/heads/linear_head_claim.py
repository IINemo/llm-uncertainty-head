import torch
import torch.nn as nn
import torch.nn.functional as F

from .uncertainty_head_base import UncertaintyHeadBase

import logging

log = logging.getLogger()


class LinearHeadClaim(UncertaintyHeadBase):
    def __init__(
        self,
        feature_extractor,
        cfg = None,  # Temporary we save initializing cfg in the head itself
    ):
        super().__init__(feature_extractor, cfg=cfg, model_type="claim")
        self.classifier = nn.Linear(feature_extractor.feature_dim(), 1)

    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        claims = llm_inputs["claims"]
        features = X.to(torch.float32)

        results = []
        batch_size = len(claims)
        for i in range(batch_size):
            entity_mask = claims[i]
            
            sum_entity_embeds = (features[i, :] * entity_mask.unsqueeze(-1)).sum(dim=1)  
            count_entity_tokens = entity_mask.sum(dim=1).clamp(min=1)
            entity_representation = sum_entity_embeds / count_entity_tokens.unsqueeze(-1)
            
            out = self.classifier(entity_representation)
            results.append(out)
        
        # Padding to ensure uniform output shape
        max_entities_per_batch = max(o.shape[0] for o in results)
        padded_results = [F.pad(o, (0, 0, 0, max_entities_per_batch - o.shape[0]), value=-100) for o in results]
        
        return torch.stack(padded_results)

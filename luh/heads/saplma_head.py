import torch
import torch.nn as nn

from .uncertainty_head_base import UncertaintyHeadBase

import logging

log = logging.getLogger()


class SaplmaHead(UncertaintyHeadBase):
    def __init__(
        self,
        feature_extractor,
        internal_dim1=256,
        internal_dim2=128,
        internal_dim3=64,
        cfg = None,  # Temporary we save initializing cfg in the head itself
    ):
        super().__init__(feature_extractor, cfg=cfg)

        log.info(f"Feature dim: {feature_extractor.feature_dim()}")
        self.layers = nn.ModuleList([nn.Linear(feature_extractor.feature_dim(), internal_dim1),
                                     nn.ReLU(),
                                     nn.Linear(internal_dim1, internal_dim2),
                                     nn.ReLU(),
                                     nn.Linear(internal_dim2, internal_dim3),
                                     nn.ReLU(),
                                     nn.Linear(internal_dim3, 1)])

    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        # No transformer modules, so X_attn_mask is not needed
        out = X.to(torch.float32)
        for layer in self.layers:
            out = layer(out)

        return out

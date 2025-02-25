import torch
import torch.nn as nn

from .uncertainty_head_base import UncertaintyHeadBase

import logging

log = logging.getLogger()


class LinearHead(UncertaintyHeadBase):
    def __init__(
        self,
        feature_extractor,
        cfg = None,  # Temporary we save initializing cfg in the head itself
    ):
        super().__init__(feature_extractor, cfg=cfg, model_type="token")
        self.classifier = nn.Linear(feature_extractor.feature_dim(), 1)

    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        out = X.to(torch.float32)
        out = self.classifier(out)
        return out

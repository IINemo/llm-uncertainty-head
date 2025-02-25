import torch
import torch.nn as nn

from .uncertainty_head_base import UncertaintyHeadBase

import logging

log = logging.getLogger()


class UncertaintyHead(UncertaintyHeadBase):
    def __init__(
        self,
        feature_extractor,
        head_dim: int = 4096,
        interim_dim: int = 0,
        n_layers: int = 2,
        n_heads: int = 32,
        dropout: float = 0.1,
        enable_feature_projection_layer: bool = True,
        use_transformer_encoder: bool = True,
        cfg = None,  # Temporary we save initializing cfg in the head itself
    ):
        super().__init__(feature_extractor, cfg=cfg)

        self.feature_extractor = feature_extractor
        self.use_transformer_encoder = use_transformer_encoder

        log.info(f"Feature size: {feature_extractor.feature_dim()}")
        if enable_feature_projection_layer:
               self.proj = nn.Sequential(
                nn.Linear(feature_extractor.feature_dim(), head_dim * 2),
                nn.LayerNorm(head_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim * 2, head_dim),
                nn.LayerNorm(head_dim),
                nn.GELU(),
            )
        else:
            head_dim = feature_extractor.feature_dim()

        if self.use_transformer_encoder: 
            #self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, head_dim))
            #self.positional_encoding = nn.Embedding(5000, head_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=head_dim, nhead=n_heads, dropout=dropout, activation="gelu", batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )
        
        if interim_dim:
            self.classifier = nn.Sequential(
                nn.Linear(head_dim, interim_dim),
                nn.LayerNorm(interim_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(interim_dim, 1)
            )
        else:
            self.classifier = nn.Linear(head_dim, 1)

        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"Total number of parameters {total_params}")

    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        out = X.to(torch.float32)

        if hasattr(self, "proj"):
            out = self.proj(out)

        if self.use_transformer_encoder: 
            src_key_padding_mask = (X_attn_mask == 0)
            # positions = torch.arange(X.size(1), device=X.device).unsqueeze(0).expand(X.shape[0], X.size(1))
            # pos_embeds = self.positional_encoding(positions)
            # out = out + pos_embeds
            out = self.transformer_encoder(out, src_key_padding_mask=src_key_padding_mask)

        out = self.classifier(out)

        return out

import torch
import torch.nn as nn
import torch.nn.functional as F

from .uncertainty_head_base import UncertaintyHeadBase

import logging

log = logging.getLogger()


class UncertaintyHeadClaim(UncertaintyHeadBase):
    def __init__(
        self,
        feature_extractor,
        head_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        cfg = None,  # Temporary we save initializing cfg in the head itself
        mask_future_tokens: bool = False,
    ):
        super().__init__(feature_extractor, cfg=cfg, model_type="claim")

        self.mask_future_tokens = mask_future_tokens

        self.feature_extractor = feature_extractor
        log.info(f"Feature size: {feature_extractor.feature_dim()}")

        self.proj = nn.Sequential(
                nn.Linear(feature_extractor.feature_dim(), head_dim * 2),
                nn.LayerNorm(head_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_dim * 2, head_dim),
                nn.LayerNorm(head_dim),
                nn.GELU(),
            )

        #self.position_embedding = nn.Embedding(5000, head_dim)
        self.entity_embedding = nn.Embedding(2, head_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
                d_model=head_dim, nhead=n_heads, dropout=dropout, activation="gelu", batch_first=True
            )
        self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=n_layers
            )
        
        self.classifier = nn.Sequential(
                nn.Linear(head_dim, head_dim),
                nn.LayerNorm(head_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(head_dim, 1)
            )

        total_params = sum(p.numel() for p in self.parameters())
        log.info(f"Total number of parameters {total_params}")

    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        claims = llm_inputs["claims"]
        features = X.to(torch.float32)
        features = self.proj(features)

        src_key_padding_mask = (X_attn_mask == 0)
        results = []
        batch_size = len(claims)
        #max_tokens = X.size(1)

        for i in range(batch_size):
            entity_mask = claims[i]
            ent_embeds = self.entity_embedding(entity_mask)
            #positions = torch.arange(max_tokens, device=X.device).unsqueeze(0)
            #pos_embeds = self.position_embedding(positions)
            
            out = features[i].unsqueeze(0).repeat(ent_embeds.shape[0], 1, 1) + ent_embeds # + pos_embeds.repeat(ent_embeds.shape[0], 1, 1)
            src_key_pd = src_key_padding_mask[i].unsqueeze(0).repeat(ent_embeds.shape[0], 1)

            assert entity_mask.shape == src_key_pd.shape
            if self.mask_future_tokens:
                cumulative_mask = torch.flip(torch.cummax(torch.flip(entity_mask.int(), dims=[1]), dim=1)[0], dims=[1]).bool()
                # log.debug(f'Masking future tokens in: {src_key_pd} by {entity_mask} entity mask: {src_key_pd & cumulative_mask}')
                src_key_pd &= cumulative_mask
            
            out = self.transformer_encoder(out, src_key_padding_mask=src_key_pd)
            
            sum_entity_embeds = (out * entity_mask.unsqueeze(-1)).sum(dim=1)  
            count_entity_tokens = entity_mask.sum(dim=1).clamp(min=1)
            entity_representation = sum_entity_embeds / count_entity_tokens.unsqueeze(-1)
            
            out = self.classifier(entity_representation)
            results.append(out)
        
        # Padding to ensure uniform output shape
        max_entities_per_batch = max(o.shape[0] for o in results)
        padded_results = [F.pad(o, (0, 0, 0, max_entities_per_batch - o.shape[0]), value=-100) for o in results]
        
        return torch.stack(padded_results)

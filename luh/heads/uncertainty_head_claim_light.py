import torch
import torch.nn as nn
import torch.nn.functional as F

from .uncertainty_head_base import UncertaintyHeadBase

import logging

log = logging.getLogger()


class UncertaintyHeadClaimLight(UncertaintyHeadBase):
    def __init__(
        self,
        feature_extractor,
        head_dim: int = 256,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        cfg = None,  # Temporary we save initializing cfg in the head itself
    ):
        super().__init__(feature_extractor, cfg=cfg, model_type="claim")

        self.feature_extractor = feature_extractor
        log.info(f"Feature size: {feature_extractor.feature_dim()}")

        self.proj = nn.Sequential(
                nn.Linear(feature_extractor.feature_dim(), head_dim),
                nn.LayerNorm(head_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.entity_embedding = nn.Embedding(2, head_dim)
        self.cls_token_embedding = nn.Parameter(torch.randn(1, 1, head_dim)) # CLS token
        # Max positions should ideally be 1 (for CLS) + cfg.max_sequence_length_from_feature_extractor
        # Using 512 as a placeholder for (max_tokens_from_X + 1)
        self.max_transformer_seq_len = 512 
        self.position_embeddings = nn.Embedding(self.max_transformer_seq_len, head_dim)
        
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

        src_key_padding_mask = (X_attn_mask == 0) # True for padded tokens, False for non-padded
        results = []
        batch_size = len(claims)

        for i in range(batch_size):
            entity_mask_for_item = claims[i] # Shape: (num_entities, max_tokens)
            if len(entity_mask_for_item) == 0:
                # results.append(torch.empty(0, 1, device=features.device)) # Or handle as needed
                continue

            num_entities = entity_mask_for_item.shape[0]
            current_features = features[i] # Shape: (max_tokens, head_dim)
            
            # Create entity-specific token embeddings
            # entity_mask_for_item needs to be long for nn.Embedding
            ent_embeds = self.entity_embedding(entity_mask_for_item.long()) # Shape: (num_entities, max_tokens, head_dim)
            base_token_embeddings_repeated = current_features.unsqueeze(0).expand(num_entities, -1, -1) # Shape: (num_entities, max_tokens, head_dim)
            entity_specific_token_embeddings = base_token_embeddings_repeated + ent_embeds # Shape: (num_entities, max_tokens, head_dim)

            # Prepare CLS token and combine with entity-specific token embeddings
            cls_embedding_expanded = self.cls_token_embedding.expand(num_entities, -1, -1) # Shape: (num_entities, 1, head_dim)
            transformer_input_no_pos = torch.cat([cls_embedding_expanded, entity_specific_token_embeddings], dim=1) # Shape: (num_entities, 1 + max_tokens, head_dim)

            # Add positional embeddings
            current_seq_len = transformer_input_no_pos.size(1)
            if current_seq_len > self.max_transformer_seq_len:
                # This should not happen if inputs are correctly truncated/padded to feature_extractor's max_len
                # Or if self.max_transformer_seq_len is set correctly based on true max possible length
                log.warning(f"Current sequence length {current_seq_len} exceeds max configured position embeddings {self.max_transformer_seq_len}. Truncating.")
                transformer_input_no_pos = transformer_input_no_pos[:, :self.max_transformer_seq_len, :]                
                current_seq_len = self.max_transformer_seq_len

            position_ids = torch.arange(current_seq_len, device=X.device).unsqueeze(0).expand(num_entities, -1) # Shape: (num_entities, current_seq_len)
            pos_embeds = self.position_embeddings(position_ids) # Shape: (num_entities, current_seq_len, head_dim)
            transformer_input = transformer_input_no_pos + pos_embeds

            # Adjust padding mask for the CLS token
            original_padding_mask_for_item = src_key_padding_mask[i].unsqueeze(0).expand(num_entities, -1) # Shape: (num_entities, max_tokens)
            # CLS token is never padded, so add a column of False (or 0 if mask uses 0 for non-padded)
            cls_padding = torch.full((num_entities, 1), False, dtype=original_padding_mask_for_item.dtype, device=X.device)
            final_padding_mask = torch.cat([cls_padding, original_padding_mask_for_item], dim=1) # Shape: (num_entities, 1 + max_tokens)
            
            transformer_output = self.transformer_encoder(transformer_input, src_key_padding_mask=final_padding_mask) # Shape: (num_entities, 1 + max_tokens, head_dim)
            
            # Use the CLS token's output (at index 0) as the entity representation
            entity_representation = transformer_output[:, 0, :] # Shape: (num_entities, head_dim)
            
            classified_output = self.classifier(entity_representation)
            results.append(classified_output)
        
        # Padding to ensure uniform output shape
        max_entities_per_batch = max([o.shape[0] for o in results], default=1)
        padded_results = [F.pad(o, (0, 0, 0, max_entities_per_batch - o.shape[0]), value=-100) for o in results]
        
        return torch.stack(padded_results) if len(padded_results) else torch.zeros(0)

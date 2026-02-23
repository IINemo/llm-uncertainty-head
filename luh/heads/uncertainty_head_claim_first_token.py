import torch

from .uncertainty_head_claim import UncertaintyHeadClaim


class UncertaintyHeadClaimFirstToken(UncertaintyHeadClaim):
    @staticmethod
    def _keep_first_token_only(entity_mask: torch.Tensor) -> torch.Tensor:
        if entity_mask.numel() == 0:
            return entity_mask

        # Keep only the first positive token for each claim row.
        bool_mask = entity_mask.to(torch.bool)
        first_only = bool_mask & (bool_mask.cumsum(dim=1) == 1)
        return first_only.to(entity_mask.dtype)

    def _compute_tensors(self, llm_inputs, X, X_attn_mask):
        claims = llm_inputs["claims"]
        truncated_claims = [self._keep_first_token_only(mask) for mask in claims]

        patched_inputs = dict(llm_inputs)
        patched_inputs["claims"] = truncated_claims
        return super()._compute_tensors(patched_inputs, X, X_attn_mask)


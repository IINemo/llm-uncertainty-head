from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
from lm_polygraph.stat_calculators.extract_claims import Claim

from .utils import recursive_to

from typing import Dict, Tuple, List
import torch
import numpy as np
import logging

log = logging.getLogger()


class CalculatorApplyUQHead(StatCalculator):
    def __init__(self, uncertainty_head, device = None):
        super().__init__()
        self.uncertainty_head = uncertainty_head
        self.device = device

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "uncertainty_claim_logits",
        ], ["uhead_features", "claims"]

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        batch = dependencies["llm_inputs"]

        if "claims" in dependencies:
            claims = dependencies["claims"]
        else:
            log.warning("No claims detected in dependencies. Falling back to whole generation as single claim")
            claims = [None]
        if "num_return_sequences" in kwargs:
            num_return_sequences = kwargs["num_return_sequences"]
            claims = [c for c in claims for _ in range(num_return_sequences)]
        
        batch["claims"] = self.prepare_claims(batch, claims, dependencies["full_attention_mask"].shape[1])

        device = self.device
        if device is None:
            device = model.device()
        self.uncertainty_head = self.uncertainty_head.to(device)
        with torch.no_grad():
            uncertainty_logits = self.uncertainty_head._compute_tensors(
                recursive_to(batch, device),
                dependencies["uhead_features"].to(device),
                dependencies["full_attention_mask"][:, :-1].to(device), # Ignoring last token
            )

        final_uncertainty_claims = [np.asarray([e.item() for e in claim if e != -100]) for claim in uncertainty_logits.cpu().numpy()]
        results = {"uncertainty_claim_logits": final_uncertainty_claims}
        return results

    def prepare_claims(self, batch, claims, full_len):
        batch_size = len(batch["input_ids"])
        context_lengths = batch["context_lengths"]
        all_claim_tensors = []
        for i in range(batch_size):
            instance_claims = []
            if claims[i] is None:
                claims[i] = [Claim(None, None, list(range(full_len - context_lengths[i])))]
            for claim in claims[i]:
                mask = torch.zeros(full_len, dtype=int)
                mask[context_lengths
                [i] + torch.as_tensor(claim.aligned_token_ids).long()] = 1
                instance_claims.append(mask[1:]) # ignoring <s>

            all_claim_tensors.append(torch.stack(instance_claims) if len(instance_claims) > 0 else torch.zeros(0, full_len - 1, dtype=int))
        
        return all_claim_tensors

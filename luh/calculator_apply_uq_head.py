from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model

from .utils import recursive_to

from typing import Dict, Tuple, List
import torch
import numpy as np


class CalculatorApplyUQHead(StatCalculator):
    def __init__(self, uncertainty_head):
        super().__init__()
        self.uncertainty_head = uncertainty_head

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
        max_new_tokens: int,
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        batch = dependencies["llm_inputs"]

        claims = dependencies["claims"]
        if "num_return_sequences" in kwargs:
            num_return_sequences = kwargs["num_return_sequences"]
            claims = [c for c in claims for _ in range(num_return_sequences)]
        
        batch["claims"] = self.prepare_claims(batch, claims, dependencies["full_attention_mask"].shape[1])

        with torch.no_grad():
            uncertainty_logits = self.uncertainty_head._compute_tensors(
                recursive_to(batch, model.device()),
                dependencies["uhead_features"].to(model.device()),
                dependencies["full_attention_mask"][:, :-1].to(model.device()), # Ignoring last token
            )

        final_uncertainty_claims = [np.asarray([e.item() for e in claim if e != -100]) for claim in uncertainty_logits.cpu().numpy()]
        results = {"uncertainty_claim_logits": final_uncertainty_claims}
        return results

    def prepare_claims(self, batch, claims, full_len):
        batch_size = len(batch["input_ids"])
        context_lenghts = batch["context_lenghts"]
        all_claim_tensors = []
        for i in range(batch_size):
            instance_claims = []
            for claim in claims[i]:
                mask = torch.zeros(full_len, dtype=int)
                mask[context_lenghts[i] + torch.as_tensor(claim.aligned_token_ids)] = 1
                instance_claims.append(mask[1:]) # ignoring <s>

            all_claim_tensors.append(torch.stack(instance_claims) if len(instance_claims) > 0 else torch.zeros(0, full_len - 1, dtype=int))
        
        return all_claim_tensors

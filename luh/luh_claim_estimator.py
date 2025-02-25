from lm_polygraph.estimators.estimator import Estimator

import numpy as np
import torch
from typing import Dict

from scipy.special import expit

import logging
log = logging.getLogger(__name__)


class LuhClaimEstimator(Estimator):
    def __init__(
        self,
        reduce_type: str,  # either mean or min
        reverse_predictions: bool = False
    ):
        super().__init__(
            ["uncertainty_logits", "claims"],
            "claim",
        )

        self._reduce_type = reduce_type
        self._reverse_predictions = reverse_predictions

    def _reduce(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()

        if self._reduce_type == "mean_logits":
            scores = x.mean(-1)

        # elif self._reduce_type == "mean":
        #     scores = x.exp().mean(-1) # TODO: change to sigmoid
        # elif self._reduce_type == "mean_geom":
        #     scores = x.exp().prod(-1) / len(x)
        # elif self._reduce_type == "prod_geom":
        #     scores = x.exp().prod(-1)

        elif self._reduce_type == "mean":
            scores = expit(x).mean(-1) # TODO: change to sigmoid
        elif self._reduce_type == "mean_geom":
            scores = expit(x).prod(-1) / len(x)
        elif self._reduce_type == "prod_geom":
            scores = expit(x).prod(-1)
        elif self._reduce_type == "max":
            scores = x.max(-1)

        elif self._reduce_type == "min":
            scores = x.min(-1).values
        # elif self._reduce_type == "top3":
        #     topk = min(3, x.shape[-1])
        #     scores = x.exp().topk(topk, largest=False).values.prod(-1) / topk
        else:
            raise ValueError(f"Unrecognized reduce type {self._reduce_type}")

        if self._reverse_predictions:
            log.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Here we are returning negative scores &%%%%%%%%%%%%%!!!!!!!!!')
            return (-scores).tolist()
        else:
            log.debug('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Here we are returning positive scores &%%%%%%%%%%%%%!!!!!!!!!')
            return scores.tolist()

    def __str__(self):
        return f"LuqClaimEstimator_{self._reduce_type}_reverse={self._reverse_predictions}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        claims = stats["claims"]

        luq_scores = stats["uncertainty_logits"]
        claim_ue = []
        for sample_ls, sample_claims in zip(luq_scores, claims):
            claim_ue.append([])
            for claim in sample_claims:
                tokens = np.array(claim.aligned_token_ids)
                claim_luq_score = sample_ls[tokens]
                claim_ue[-1].append(self._reduce(claim_luq_score))

        return claim_ue

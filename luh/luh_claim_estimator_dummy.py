from lm_polygraph.estimators.estimator import Estimator

import numpy as np
from typing import Dict
from scipy.special import expit

import logging
log = logging.getLogger(__name__)


class LuhClaimEstimatorDummy(Estimator):
    def __init__(
        self
    ):
        super().__init__(
            ["uncertainty_claim_logits", "claims"],
            "claim",
        )

    def __str__(self):
        return f"LuqClaimEstimatorDummy_claim"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        luq_scores = stats["uncertainty_claim_logits"]

        claim_ue = []
        for sample_ls in luq_scores:
            claim_ue.append(expit(sample_ls).tolist())

        return claim_ue

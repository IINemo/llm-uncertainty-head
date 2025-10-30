from lm_polygraph.estimators.estimator import Estimator

import numpy as np
from typing import Dict
from scipy.special import expit

import logging
log = logging.getLogger(__name__)


class LuhEstimatorDummy(Estimator):
    def __init__(
        self
    ):
        super().__init__(
            ["uncertainty_logits", "greedy_tokens"],
            "token",
        )

    def __str__(self):
        return f"LuqEstimatorDummy_token"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        luq_scores = stats["uncertainty_logits"]

        claim_ue = []
        for sample_ls in luq_scores:
            claim_ue.append(expit(sample_ls).tolist())

        return claim_ue

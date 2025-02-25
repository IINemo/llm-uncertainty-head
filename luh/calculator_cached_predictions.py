from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
from lm_polygraph.stat_calculators.extract_claims import Claim

from typing import Dict, List, Tuple
from datasets import load_from_disk, load_dataset
import torch
import numpy as np

import logging
log = logging.getLogger()



class CalculatorCachedPredictions(StatCalculator):
    def __init__(
        self,
        cached_predictions_path: str,
        dataset_path: str,
    ):
        super().__init__()

        self.cached_predictions = load_from_disk(cached_predictions_path)
        self.dataset = load_dataset(dataset_path)["test"]
        self.search_index = {self.dataset[i]["question"]: i for i in range(len(self.dataset))}

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "greedy_log_probs",
            "greedy_tokens",
            "greedy_texts",
            "greedy_log_likelihoods",
            "uncertainty_logits",
            "claims"
        ], []

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,  # TODO: move to args_generate
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        batch: Dict[str, torch.Tensor] = model.tokenize(texts)

        batch_size = batch["input_ids"].shape[0]
        input_tokens = model.tokenize(texts)["input_ids"]

        all_logits = []
        all_predicted_tokens = []
        all_uncertainty_logits = []
        all_claims = []
        ll = []
        for i in range(batch_size):
            ind = self.search_index[texts[i]]
            input_len = input_tokens[i].shape[0]
            logits = self.cached_predictions[ind]["logits"][input_len:]
            predicted_tokens = self.dataset[ind]["input_ids"][input_len:]
            uncertainty_logits = np.asarray(self.cached_predictions[ind]["uncertainty_logits"][input_len - 1:]).reshape(-1)
            claims = [Claim(**e) for e in self.dataset[ind]["claims"]]
            
            ll.append(np.asarray([logits[j][predicted_tokens[j]] for j in range(len(predicted_tokens))]))
            all_logits.append(logits)
            all_predicted_tokens.append(predicted_tokens)
            all_uncertainty_logits.append(uncertainty_logits)
            all_claims.append(claims)

        result_dict = {
            "input_tokens": input_tokens.tolist(),
            "greedy_log_probs": all_logits,
            "greedy_tokens": all_predicted_tokens,
            "greedy_texts": model.tokenizer.batch_decode(all_predicted_tokens),
            "uncertainty_logits": all_uncertainty_logits,
            "claims": all_claims,
            "greedy_log_likelihoods": ll,
        }

        return result_dict

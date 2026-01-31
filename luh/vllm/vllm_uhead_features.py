import torch
from lm_polygraph.stat_calculators import StatCalculator
from luh import AutoUncertaintyHead
from lm_polygraph.stat_calculators.extract_claims import Claim
from .vllm_features_token_probs import VLLMTokenProbabilities
from .vllm_features_hidden_states import VLLMHiddenStates
from luh.feature_extractors.combined import FeatureExtractorCombined
from luh.feature_extractors.token_probabilities import FeatureExtractorTokenProbabilities
from luh.feature_extractors.basic_hidden_states import FeatureExtractorBasicHiddenStates


class VLLMUncertaintyHeadFeatures(StatCalculator):
    def __init__(self, uhead: AutoUncertaintyHead, **hs_generator_kwargs):
        super().__init__()
        self.hs_generator_kwargs = hs_generator_kwargs
        feature_extractor = getattr(uhead, "feature_extractor")
        if isinstance(feature_extractor, FeatureExtractorCombined):
            feature_extractors = getattr(feature_extractor, "_feature_extractors")
        else:
            feature_extractors = [feature_extractor]

        self.vllm_feature_extractors: list[tuple[str, StatCalculator]] = []

        # NEW: keep generator layer ids separately
        self._hs_layer_ids: list[int] | None = None

        self.n_logprobs = 0

        for fe in feature_extractors:
            if isinstance(fe, FeatureExtractorTokenProbabilities):
                ext = VLLMTokenProbabilities(top_n=fe.top_n)
                self.vllm_feature_extractors.append(("vllm_token_probs", ext))
                self.n_logprobs = max(self.n_logprobs, ext.top_n)

            elif isinstance(fe, FeatureExtractorBasicHiddenStates):
                # fe._layer_nums are actual model layers (e.g., [2,10,20] or [-1] resolved)
                self._hs_layer_ids = list(fe._layer_nums)

                # Stat calculator selects among RETURNED layers.
                # If you want ALL returned layers in the same order:
                layer_selection = list(range(len(self._hs_layer_ids)))
                ext = VLLMHiddenStates(layer_nums=layer_selection)

                self.vllm_feature_extractors.append(("vllm_hidden_states", ext))

            else:
                raise Exception(f"Feature extractor {fe.__class__.__name__} is not supported with vllm")

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        return (["uhead_features", "full_attention_mask", "llm_inputs"], [])

    def vllm_with_uncertainty_arguments(self) -> dict:
        return {
            "n_logprobs": self.n_logprobs,
            "output_hidden_states": self._hs_layer_ids is not None,
            "hs_layer_ids": self._hs_layer_ids,
            "hs_generator_kwargs": self.hs_generator_kwargs,
        }

    def _completion_samples_from_output(self, output):
        # output may be RequestOutputWithUncertainty, RequestOutput, list of them,
        # or a completion object already.
        if isinstance(output, (list, tuple)):
            samples = []
            for o in output:
                if hasattr(o, "outputs"):  # request output
                    samples.extend(list(o.outputs))
                else:
                    samples.append(o)
            return samples

        if hasattr(output, "outputs"):
            return list(output.outputs)

        return [output]


    def __call__(self, dependencies, texts, model, max_new_tokens: int = 100, **kwargs):
        import numpy as np

        all_feats = []
        prev_mask = None

        for feature_key, calc in self.vllm_feature_extractors:
            dependencies.update(calc(dependencies, texts, model, max_new_tokens=max_new_tokens, **kwargs))

            feats = dependencies[feature_key]
            all_feats.append(feats)

            cur_mask = dependencies.get("full_attention_mask")
            if prev_mask is None:
                prev_mask = cur_mask
            else:
                if not np.array_equal(prev_mask, cur_mask):
                    raise Exception("Got different padding masks across vLLM feature extractors.")

        # Combine features: concatenate on last dim (B, T, *)
        if len(all_feats) == 1:
            combined = torch.FloatTensor(all_feats[0])
        else:
            combined = torch.cat(all_feats, dim=-1)

        dependencies["uhead_features"] = combined
        dependencies["llm_inputs"] = {
            "input_ids": [None for _ in range(len(combined))],
            "context_lenghts": dependencies["context_lenghts"],
        }
        dependencies["claims"] = [[
            Claim(claim_text=None, sentence=None, aligned_token_ids=list(range(len(m.token_ids) - 1)))
        ] for m in self._completion_samples_from_output(dependencies["vllm_output"])]
        return dependencies

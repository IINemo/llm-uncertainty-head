import torch

from .utils import get_hidden_states, get_layer_nums
from .feature_extractor_base import FeatureExtractorBase


class FeatureExtractorTokenProbabilitiesFromLayers(FeatureExtractorBase):
    def __init__(self, orig_base_model, top_n, layer_nums='all', **kwargs):
        self.lm_head = orig_base_model.lm_head
        self.top_n = top_n
        self._layer_nums = get_layer_nums(layer_nums, orig_base_model)

    def feature_dim(self):
        return len(self._layer_nums) * self.top_n

    def __call__(self, llm_inputs, llm_outputs):
        # batch_size x seq_len x layers x hidden_state
        hidden_states = get_hidden_states(llm_outputs)

        # For loop on purpose: otherwise receiving CUDA OOM for Llama3/Gemma2. Should be quite fast anyway
        prob_features = []
        for layer in self._layer_nums:
            with torch.no_grad():
                all_logits = self.lm_head(hidden_states[:, :, layer, :])  # batch_sz x seq_len x vocabulary
            top_probs = torch.sort(all_logits, dim=-1, descending=True)[0][:, :, :self.top_n]
            prob_features.append(top_probs)
        prob_features = torch.cat(prob_features, dim=-1)  # batch_sz x seq_len x (layers * top_n)

        return prob_features


def load_extractor(config, base_model, *args, **kwargs):
    return FeatureExtractorTokenProbabilitiesFromLayers(base_model, **config)

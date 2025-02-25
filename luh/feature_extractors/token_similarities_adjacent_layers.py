import torch

from .utils import get_hidden_states, get_layer_nums
from .feature_extractor_base import FeatureExtractorBase


class FeatureExtractorTokenSimilaritiesAdjacentLayers(FeatureExtractorBase):
    def __init__(self, orig_base_model, top_n, layer_nums='all', **kwargs):
        self.lm_head = orig_base_model.lm_head
        self.top_n = top_n
        self._layer_nums = get_layer_nums(layer_nums, orig_base_model)
        assert len(self._layer_nums) > 0

    def feature_dim(self):
        return (len(self._layer_nums) - 1) * self.top_n

    def get_embeddings(self, indices):
        return self.lm_head.weight[indices]

    def __call__(self, llm_inputs, llm_outputs):
        # batch_size x seq_len x layers x hidden_state
        hidden_states = get_hidden_states(llm_outputs)

        # For loop on purpose: otherwise receiving CUDA OOM for Llama3/Gemma2. Should be quite fast anyway
        prev_embeddings = None
        sim_features = []
        for layer in self._layer_nums:
            with torch.no_grad():
                all_logits = self.lm_head(hidden_states[:, :, layer, :])  # batch_sz x seq_len x vocabulary
            top_indices = torch.argsort(all_logits, dim=-1, descending=True)[:, :, :self.top_n]
            top_embeddings = self.get_embeddings(top_indices)  # batch_sz x seq_len x top_n x embeddings
            if prev_embeddings is not None:
                sims = torch.cosine_similarity(top_embeddings, prev_embeddings, dim=-1)  # batch_sz x seq_len x top_n
                sim_features.append(sims)
            prev_embeddings = top_embeddings
        prob_features = torch.cat(sim_features, dim=-1)  # batch_sz x seq_len x ((layers - 1) * top_n)

        return prob_features


def load_extractor(config, base_model, *args, **kwargs):
    return FeatureExtractorTokenSimilaritiesAdjacentLayers(base_model, **config)

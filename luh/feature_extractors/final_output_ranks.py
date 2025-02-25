import torch

from .feature_extractor_base import FeatureExtractorBase
from .utils import get_hidden_states, get_layer_nums


class FinalOutputRanks(FeatureExtractorBase):
    def __init__(self, orig_base_model, layer_nums='all', **kwargs):
        self.lm_head = orig_base_model.lm_head
        self._layer_nums = get_layer_nums(layer_nums, orig_base_model)

    def feature_dim(self):
        return len(self._layer_nums)

    def rankings(self, llm_inputs, llm_outputs, **kwargs):
        # batch_size x seq_len x layers x hidden_state
        hidden_states = get_hidden_states(llm_outputs)
        if hasattr(llm_outputs, "sequences"):
            token_ids = llm_outputs.sequences
        else:
            token_ids = llm_inputs['input_ids']

        rankings = torch.ones((
            token_ids.shape[0], token_ids.shape[1] - 1, len(self._layer_nums)
        )).int().to(token_ids.device)  # batch_size x seq_len x layers

        # For loop on purpose: otherwise receiving CUDA OOM for Llama3/Gemma2. Should be quite fast anyway
        for layer in self._layer_nums:
            with torch.no_grad():
                all_logits = self.lm_head(hidden_states[:, :, layer, :])  # batch_sz x seq_len x vocabulary
            sorted_idx = torch.argsort(all_logits, dim=-1, descending=True)
            expanded_token_ids = token_ids[:, :-1].unsqueeze(-1).expand(*sorted_idx.shape)
            matches = (expanded_token_ids[:, 1:, :] == sorted_idx[:, :-1, :])  # batch_sz x seq_len x vocabulary
            rankings[:, 1:, layer] = matches.int().argmax(dim=-1) + 1

        return rankings  # batch_size x sequence_length x feature_vector

    def __call__(self, llm_inputs, llm_outputs):
        all_rank = self.rankings(llm_inputs, llm_outputs)  # all ranks are in [1, |V|]
        return 1 / all_rank  # (0, 1]
        # This is a strange formula from FactoScope paper&github. Probably should not use it, instead just 1/rank.
        # return 1 / ((1 - all_rank) + 1 + 1e-7)


def load_extractor(config, base_model, *args, **kwargs):
    return FinalOutputRanks(base_model)

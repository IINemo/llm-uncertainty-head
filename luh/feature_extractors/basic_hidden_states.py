import torch

from .feature_extractor_base import FeatureExtractorBase
from .utils import get_layer_nums


class FeatureExtractorBasicHiddenStates(FeatureExtractorBase):
    def __init__(self, orig_base_model, layer_nums=[-1], **kwargs):
        self._layer_nums = get_layer_nums(layer_nums, orig_base_model)
        self._feature_dim = orig_base_model.config.hidden_size * len(self._layer_nums)

    def __call__(self, llm_inputs, llm_outputs):
        """ output = (batch_size x output.sequences.shape[0] x hidden_state) """

        is_training = type(llm_outputs["hidden_states"][-1]) == torch.Tensor
        #print("Is training", is_training)
        if is_training:
            # hidden states during training:  (n_layers x tensor(batch_size x n_tokens x hidden_state))
            res = torch.cat(
                [llm_outputs["hidden_states"][layer][:, :-1, :] for layer in self._layer_nums], dim=-1 # Ignoring the output for the last token
            )
            #print("Shape of the hidden states features", res.shape)
            return res
        else:
            # hidden states during generation: (n tokens x (n_layers x tensor(batch_size x 1 x hidden_state)))
            return torch.cat(
                [
                    torch.cat([t[layer] for layer in self._layer_nums], dim=-1)
                    #for t in outputs["hidden_states"][1:] # We are ignoring tokens from the prompt, super crucial, gives 10%
                    for t in llm_outputs["hidden_states"]
                ],
                dim=1,
            )

    def feature_dim(self):
        return self._feature_dim


def load_extractor(config, base_model):
    return FeatureExtractorBasicHiddenStates(base_model, **config)
# TODO: fix problem with name paramter in the config
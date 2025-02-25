import torch
from collections.abc import Iterable


def get_layer_nums(layer_nums, orig_base_model):
    if layer_nums == 'all':
        return list(range(orig_base_model.config.num_hidden_layers))
    elif isinstance(layer_nums, Iterable):
        return list(layer_nums)
    return (layer_nums,)


def get_hidden_states(llm_outputs):
    hs = llm_outputs["hidden_states"]
    is_training = type(hs[-1]) == torch.Tensor
    if is_training:
        return torch.stack([
            hs[layer][:, :-1, :]
            for layer in range(len(hs))
        ], dim=-2)
    else:
        return torch.cat([
            torch.stack([
                t[layer]
                for layer in range(len(t))
            ], dim=-2)
            for t in hs
        ], dim=1)

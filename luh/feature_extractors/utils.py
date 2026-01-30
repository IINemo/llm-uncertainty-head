import torch
from collections.abc import Iterable


def get_layer_nums(layer_nums, orig_base_model):
    if layer_nums == 'all':
        # Handle VLM configs with text_config
        config = orig_base_model.config
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_hidden_layers'):
            num_layers = config.text_config.num_hidden_layers
        elif hasattr(config, 'num_hidden_layers'):
            num_layers = config.num_hidden_layers
        else:
            raise AttributeError(f"Cannot find num_hidden_layers in config. Available: {dir(config)}")
        return list(range(num_layers))
    elif isinstance(layer_nums, Iterable):
        return list(layer_nums)
    return (layer_nums,)


def get_head_nums(head_nums, layer_nums, orig_base_model):
    if head_nums == 'all':
        # Handle VLM configs with text_config
        config = orig_base_model.config
        if hasattr(config, 'text_config') and hasattr(config.text_config, 'num_attention_heads'):
            num_heads = config.text_config.num_attention_heads
        elif hasattr(config, 'num_attention_heads'):
            num_heads = config.num_attention_heads
        else:
            raise AttributeError(f"Cannot find num_attention_heads in config. Available: {dir(config)}")
        all_heads = list(range(num_heads))
        return {l: all_heads for l in layer_nums}
    elif isinstance(head_nums, dict):
        heads: dict[int, list[int]] = {}  # list of heads for each layer
        for key, val in head_nums.items():
            for l in get_layer_nums(key, orig_base_model):
                heads.update(get_head_nums(val, [l], orig_base_model))
        assert all(l in heads.keys() for l in layer_nums)
        return heads
    elif isinstance(head_nums, Iterable):
        return {l: list(head_nums) for l in layer_nums}
    return {l: (head_nums,) for l in layer_nums}


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

import torch

from .feature_extractor_base import FeatureExtractorBase
from .utils import get_layer_nums


def move_to_cpu(obj):
    """Recursively move all tensors in a nested structure to CPU."""
    # Case 1: If it's a tensor, move to CPU
    if torch.is_tensor(obj):
        return obj.cpu()

    # Case 2: If it's a list or tuple, recursively process each element
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_cpu(x) for x in obj)

    # Case 3: If it's a dict, recursively process each value
    elif isinstance(obj, dict):
        return {k: move_to_cpu(v) for k, v in obj.items()}

    # Case 4: If it's anything else (e.g., int, float, string), just return as is
    else:
        return obj


class FeatureExtractorBasicAttention(FeatureExtractorBase):
    def __init__(self, orig_base_model, layer_nums, attn_history_sz, pool, **kwargs):
        """layer_nums = {all, [-1, -2, -5, ...]}"""

        self.pool = pool
        self._layer_nums = get_layer_nums(layer_nums, orig_base_model)
        self._input_size = (
            orig_base_model.config.num_attention_heads * len(self._layer_nums)
            if not pool
            else orig_base_model.config.num_attention_heads
        )
        self._attn_history_sz = attn_history_sz

    def _instance_feature(self, all_layers_instance, curr_pos, att_mask):
        #print(att_mask)
        # all_layers_instance = (n_layers x (num_heads, curr_seq_length, prev_seq_length))

        # def process_layer(instance):
        #     att_head_size = instance.shape[0]

        #     seq_len = instance.shape[-2]
        #     res = []
        #     for i in range(seq_len):
        #         token_features = torch.zeros(size=(self._attn_history_sz, att_head_size), device=instance.device)
        #         for j in range(self._attn_history_sz):
        #             att_index = curr_pos + i - j - 1
        #             if att_index < 0:
        #                 continue

        #             token_features[j] = instance[:, i, att_index]

        #         res.append(token_features.view(-1))

        #     # Output: curr_seq_length x (attn_history_sz * num_heads)
        #     return torch.vstack(res)

        attn_history_sz = self._attn_history_sz

        def process_layer(instance):
            """
            instance: (num_heads, seq_len, prev_seq_len)
            curr_pos: int
            attn_history_sz: int
            Returns: (seq_len, attn_history_sz * num_heads)
            """
            num_heads, seq_len, prev_seq_len = instance.shape

            # i in [0..seq_len-1], j in [0..attn_history_sz-1]
            i_grid = torch.arange(seq_len, device=instance.device)  # shape: (seq_len)
            j_grid = torch.arange(
                attn_history_sz, device=instance.device
            )  # shape: (attn_history_sz)

            # meshgrid => both (seq_len, attn_history_sz)
            i_grid, j_grid = torch.meshgrid(i_grid, j_grid, indexing="ij")

            # att_index = curr_pos + i - j - 1
            att_index_grid = (
                curr_pos + i_grid - j_grid - 1
            )  # (seq_len, attn_history_sz)

            # valid_mask: True where att_index >= 0, False otherwise
            valid_mask = att_index_grid >= 0  # (seq_len, attn_history_sz)

            # clamp negatives to 0 (or you could skip them, but that forces us to do partial indexing)
            # then multiply by valid_mask to zero them out
            att_index_grid_clamped = att_index_grid.clamp(min=0)
            #print(att_index_grid_clamped)

            #  -- Advanced indexing --
            # instance has shape (num_heads, seq_len, prev_seq_len)
            # We want: instance[:, i_grid, att_index_grid_clamped]
            # which yields (num_heads, seq_len, attn_history_sz)
            #print(att_index_grid_clamped.shape)
            out_raw = instance[:, i_grid, att_index_grid_clamped] * att_mask[att_index_grid_clamped]

            # Zero out invalid positions.  valid_mask is (seq_len, attn_history_sz),
            # so unsqueeze(0) => (1, seq_len, attn_history_sz) to broadcast over num_heads.
            out_raw = out_raw * valid_mask.unsqueeze(0)

            # Reorder from (num_heads, seq_len, attn_history_sz) to (seq_len, attn_history_sz, num_heads)
            out_raw = out_raw.permute(1, 2, 0)

            # Flatten last 2 dims => (seq_len, attn_history_sz * num_heads)
            out_final = out_raw.reshape(seq_len, -1)

            return out_final

        # TODO: add pooling instead of concatenation
        # List to store processed features for each layer
        processed_layers = [process_layer(instance) for instance in all_layers_instance]

        if self.pool:
            # Stack processed results along a new dimension (layers at dim=0)
            stacked_layers = torch.stack(processed_layers, dim=0)
            # Shape: (n_layers, seq_len, attn_history_sz * num_heads)

            # Max pool over the first dimension (n_layers)
            final_res = torch.amax(stacked_layers, dim=0)
            # Shape: (seq_len, attn_history_sz * num_heads)
        else:
            final_res = torch.cat(
                [process_layer(instance) for instance in all_layers_instance], dim=-1
            )

        # Output: curr_seq_length x (attn_history_sz * num_heads * num_layers)
        return final_res

    def feature_dim(self):
        return self._input_size * self._attn_history_sz

    def __call__(self, llm_inputs, llm_outputs):
        # TODO: take into account the attention mask actually it is needed only for the first token, if since we use prev token
        attentions_all = llm_outputs.attentions
        #print(llm_outputs.keys())

        is_training = type(attentions_all[-1]) == torch.Tensor
        if is_training:
            attentions_all = [[e[:, :, :-1, :-1] for e in attentions_all]]
            attention_mask = [llm_inputs['attention_mask']]
        else:
            attentions_all = attentions_all
            #attention_mask = llm_outputs['full_attention_mask']
            # print('full att mask', llm_outputs['full_attention_mask'].shape)
            # print(llm_outputs.sequences.shape)
            # print(len(attentions_all))
            attention_mask = [llm_outputs['full_attention_mask']  ] * len(attentions_all)

        # attentions_all = (n_blocks x n_layers x batch_size x block_seq_length x prev_seq_length)

        # attentions_all = move_to_cpu(attentions_all)

        # Iterating through blocks of tokens
        all_token_features = []
        curr_pos = 0
        for attentions, att_masks in zip(attentions_all, attention_mask):
            batch_size = attentions[0].shape[0]

            # Iterating through the batch
            all_features = []
            for i in range(batch_size):
                instance = [attentions[layer_num][i] for layer_num in self._layer_nums]
                all_features.append(self._instance_feature(instance, curr_pos, att_masks[i]))

            all_token_features.append(
                torch.vstack([e.unsqueeze(0) for e in all_features])
            )
            curr_pos += instance[0].shape[1]

        # Output: batch_size x sequence_length x feature_vector
        return torch.cat(all_token_features, dim=1)  # .to(output.attentions[0].device)

    def input_size(self):
        return self._input_size * self._attn_history_sz

    def output_attention(self):
        return True


def load_extractor(config, base_model):
    return FeatureExtractorBasicAttention(base_model, **config)

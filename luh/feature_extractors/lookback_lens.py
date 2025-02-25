import torch

from .feature_extractor_base import FeatureExtractorBase


class FeatureExtractorLookbackLens(FeatureExtractorBase):
    def __init__(self, orig_base_model, **kwargs):
        self._n_layers = orig_base_model.config.num_hidden_layers
        self._n_heads = orig_base_model.config.num_attention_heads
        self._input_size = self._n_layers * self._n_heads

    def feature_dim(self):
        return self._input_size

    def __call__(self, llm_inputs, llm_outputs):
        attentions_all = llm_outputs.attentions

        is_training = not hasattr(llm_outputs, "sequences")
        layer = 0  # only for tensor shapes
        batch_sz = llm_inputs['attention_mask'].shape[0]

        if is_training:
            context_bounds = torch.as_tensor(llm_outputs.context_lengths, device=attentions_all[0].device)
            attentions_all = [
                tuple(
                    attentions_all[l][:, :, i:i + 1, :i + 1]
                    for l in range(len(attentions_all))
                )
                for i in range(attentions_all[layer].shape[-1] - 1)
            ]
        else:
            context_bounds = torch.tensor(
                [attentions_all[0][layer].shape[-1] for _ in range(batch_sz)], device=attentions_all[0][0].device)
            attn_inp = [
                tuple(
                    attentions_all[0][l][:, :, i:i + 1, :i + 1]
                    for l in range(len(attentions_all[0]))
                )
                for i in range(attentions_all[0][layer].shape[-2])
            ]
            inp_len = attentions_all[0][layer].shape[-2]
            outp_len = len(attentions_all[1:])
            attn_outp = [
                tuple(
                    a[l][:, :, :, :i + 1]
                    for l in range(len(attentions_all[0]))
                )
                for i, a in zip(range(inp_len, inp_len + outp_len), attentions_all[1:])
            ]
            attentions_all = attn_inp + attn_outp

        context_lengths = [
            llm_inputs['attention_mask'][i][:context_bounds[i]].sum().item()
            for i in range(batch_sz)
        ]

        all_features = []
        for seq_idx, attentions in enumerate(attentions_all):
            features = []
            assert attentions[0].shape[2] == 1

            attn_ctx, attn_new = [], []
            for l in range(self._n_layers):
                a = attentions[l]  # shape: (batch_sz, H, 1, seq_len)
                ctx_bounds = context_bounds  # shape: (batch_sz)
                seq_range = torch.arange(a.shape[-1], device=a.device).unsqueeze(0)  # shape: (1, seq_len)
                ctx_mask = seq_range < ctx_bounds.unsqueeze(1)  # shape: (batch_sz, seq_len)
                new_mask = ~ctx_mask  # Complement of the context mask
                attn_ctx_layer = (a[:, :, 0, :] * ctx_mask.unsqueeze(1)).sum(-1)
                attn_new_layer = (a[:, :, 0, :] * new_mask.unsqueeze(1)).sum(-1)
                attn_ctx.append(attn_ctx_layer)
                attn_new.append(attn_new_layer)
            attn_ctx = torch.stack(attn_ctx)  # shape: (L, batch_sz, H)
            attn_new = torch.stack(attn_new)  # shape: (L, batch_sz, H)

            for batch_i in range(batch_sz):
                # calculate input length
                ctx_len = context_lengths[batch_i]
                ctx_bound = context_bounds[batch_i]
                if seq_idx > ctx_bound:  # in the new tokens
                    mean_attn_ctx = attn_ctx[:, batch_i, :] / ctx_len
                    mean_attn_new = attn_new[:, batch_i, :] / (seq_idx - ctx_bound)
                    lb_ratio = mean_attn_new / (mean_attn_ctx + mean_attn_new)
                else:  # in the padding / context
                    lb_ratio = torch.ones_like(attn_ctx[:, batch_i, :])
                features.append(lb_ratio.reshape(-1))
            features = torch.stack(features)  # batch_size x feature_vector
            all_features.append(features)

        # Output: batch_size x sequence_length x feature_vector
        result = torch.stack(all_features, dim=1)
        return result

    def input_size(self):
        return self._input_size

    def output_attention(self):
        return True


def load_extractor(config, base_model, *args, **kwargs):
    return FeatureExtractorLookbackLens(base_model)

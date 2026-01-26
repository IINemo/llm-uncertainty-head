import numpy as np
from lm_polygraph.stat_calculators import StatCalculator
from lm_polygraph.model_adapters import WhiteboxModelvLLM


class VLLMTokenProbabilities(StatCalculator):
    def __init__(self, top_n: int):
        super().__init__()
        self.top_n = top_n

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        return (
            ["vllm_token_probs", "full_attention_mask"],
            ["vllm_output"],
        )

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

    def __call__(
            self,
            dependencies: dict[str, np.array],
            texts: list[str],
            model: WhiteboxModelvLLM,
            max_new_tokens: int = 100,
            **kwargs,
    ) -> dict[str, np.ndarray]:
        output = dependencies["vllm_output"]

        samples = self._completion_samples_from_output(output)

        per_sample_feats = []
        per_sample_masks = []
        max_len = 0

        # ---- Per-sample extraction ----
        for s in samples:
            logprobs = getattr(s, "logprobs", None)

            if not logprobs:
                feats = np.zeros((0, self.top_n), dtype=np.float32)
                mask = np.zeros((0,), dtype=np.int64)
                per_sample_feats.append(feats)
                per_sample_masks.append(mask)
                continue

            T = len(logprobs)
            max_len = max(max_len, T)

            feats = np.zeros((T, self.top_n), dtype=np.float32)
            mask = np.zeros((T,), dtype=np.int64)

            for t, lp_dict in enumerate(logprobs):
                if not lp_dict:
                    continue

                # Valid generated token at timestep t
                mask[t] = 1

                # Collect logprobs
                items = [
                    float(info.logprob)
                    for info in lp_dict.values()
                ]
                items.sort(reverse=True)

                k = min(self.top_n, len(items))
                if k > 0:
                    top_ps = np.exp(np.array(items[:k], dtype=np.float32))
                    feats[t, :k] = top_ps

            per_sample_feats.append(feats)
            per_sample_masks.append(mask)

        # ---- Pad to rectangular tensors ----
        B = len(per_sample_feats)

        token_probs = np.zeros((B, max_len, self.top_n), dtype=np.float32)
        full_attention_mask = np.zeros((B, max_len), dtype=np.int64)

        for i in range(B):
            T = per_sample_feats[i].shape[0]
            if T > 0:
                token_probs[i, :T, :] = per_sample_feats[i]
                full_attention_mask[i, :T] = per_sample_masks[i]

        return {
            "vllm_token_probs": token_probs,
            "full_attention_mask": full_attention_mask,
        }

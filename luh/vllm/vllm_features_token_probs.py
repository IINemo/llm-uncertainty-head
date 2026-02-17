import numpy as np
import torch
from lm_polygraph.stat_calculators import StatCalculator
from lm_polygraph.model_adapters import WhiteboxModelvLLM


class VLLMTokenProbabilities(StatCalculator):
    def __init__(self, top_n: int):
        super().__init__()
        self.top_n = top_n

    @staticmethod
    def meta_info() -> tuple[list[str], list[str]]:
        # expects deps to contain logprobs / prompt_logprobs (or vllm_output from which they can be read)
        return (["vllm_token_probs", "full_attention_mask"], ["vllm_output"])

    def _topn_probs_from_lp_dict(self, lp_dict) -> np.ndarray:
        """
        lp_dict: token_id -> object with .logprob (log p)
        returns: (top_n,) probs (p), zero-padded
        """
        out = np.zeros((self.top_n,), dtype=np.float32)
        if not lp_dict:
            return out

        # Take highest logprob entries
        items = [float(info.logprob) for info in lp_dict.values()]
        items.sort(reverse=True)
        k = min(self.top_n, len(items))
        if k > 0:
            out[:k] = np.exp(np.asarray(items[:k], dtype=np.float32))
        return out

    def __call__(
        self,
        dependencies: dict[str, np.array],
        texts: list[str],
        model: WhiteboxModelvLLM,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> dict[str, np.ndarray]:

        # --- Support both patterns:
        # (A) deps provide logprobs/prompt_logprobs directly (what you're doing in VLLMWithUncertainty)
        # (B) fallback: read from vllm_output object
        logprobs = dependencies.get("logprobs", None)
        prompt_logprobs = dependencies.get("prompt_logprobs", None)

        if logprobs is None or prompt_logprobs is None:
            out_obj = dependencies.get("vllm_output", None)
            if out_obj is not None:
                if logprobs is None:
                    logprobs = getattr(out_obj, "logprobs", None)
                if prompt_logprobs is None:
                    prompt_logprobs = getattr(out_obj, "prompt_logprobs", None)

        # Normalize to lists (empty if missing)
        if logprobs is None:
            logprobs = []
        if prompt_logprobs is None:
            prompt_logprobs = []

        # Optional sanity: ensure prompt_logprobs length matches context_length if provided
        ctx_len = dependencies.get("context_length", None)
        if ctx_len is not None and len(prompt_logprobs) not in (0, int(ctx_len)):
            # don't hard fail: sometimes prompt_logprobs may include special tokens differently
            # but it's useful to know
            pass

        # Build sequence of per-position logprob dicts for full sequence:
        # prompt positions first, then generated positions
        per_pos = list(prompt_logprobs) + list(logprobs)
        T = len(per_pos)

        token_probs = torch.zeros((T, self.top_n), dtype=torch.float32)
        mask = torch.zeros((T,), dtype=torch.int64)

        for t, lp_dict in enumerate(per_pos):
            if not lp_dict:
                continue
            token_probs[t, :] = torch.FloatTensor(self._topn_probs_from_lp_dict(lp_dict))
            mask[t] = 1

        # This calculator is typically called per-sample in your loop,
        # so return with batch dimension = 1.
        token_probs = token_probs[None, :, :]          # (1, T, top_n)
        full_attention_mask = mask[None, :]            # (1, T)

        return {
            "vllm_token_probs": token_probs[:, :-1, :],
            "full_attention_mask": full_attention_mask,
        }

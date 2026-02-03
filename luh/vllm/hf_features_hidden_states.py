import logging
import torch
from lm_polygraph.stat_calculators import StatCalculator
from lm_polygraph.model_adapters import WhiteboxModel

from luh.feature_extractors import FeatureExtractorBasicHiddenStates

log = logging.getLogger(__name__)


class HFHiddenStatesFromVLLM(StatCalculator):
    """
    Recompute hidden states using HF by concatenating:
      prompt_ids + generation_ids (HF-tokenized).

    This guarantees exact token alignment.
    """

    def __init__(self, base_model: WhiteboxModel, layer_nums: list[int]):
        super().__init__()
        self.base_model = base_model
        self.extractor = FeatureExtractorBasicHiddenStates(base_model, layer_nums)

        log.info(
            "Initialized HFHiddenStatesFromVLLM | layers=%s | model=%s",
            layer_nums,
            type(base_model).__name__,
        )

    @staticmethod
    def meta_info():
        return (
            [
                "vllm_hidden_states",
                "full_attention_mask",
                "context_lengths",
                "greedy_texts",
                "greedy_tokens",
            ],
            ["vllm_output"],
        )

    def __call__(self, dependencies, texts, model, **kwargs):
        tokens = dependencies["token_ids"]
        if not isinstance(tokens[0], list):
            tokens = [tokens]

        tokenizer = self.base_model.tokenizer
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        sequences = []
        context_lengths = []
        greedy_texts = []
        greedy_tokens = []

        # ---- 1. Tokenize prompt + generation separately ----
        for i, gen_ids in enumerate(tokens):
            gen_text = tokenizer.decode(gen_ids)

            prompt_text = texts[i]

            # Prompt tokens (with specials)
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=True,
                return_attention_mask=False,
                return_tensors=None,
            )["input_ids"]

            sequences.append(prompt_ids + gen_ids)
            context_lengths.append(len(prompt_ids))
            greedy_texts.append(gen_text)
            greedy_tokens.append(gen_ids)

        log.info(
            "HFHiddenStatesFromVLLM | batch=%d | max_prompt=%d | max_gen=%d",
            len(sequences),
            max(context_lengths),
            max(len(t) for t in greedy_tokens),
        )

        # ---- 2. Pad batch ----
        max_len = max(len(s) for s in sequences)
        B = len(sequences)

        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)

        for i, s in enumerate(sequences):
            L = len(s)
            input_ids[i, :L] = torch.tensor(s)
            attention_mask[i, :L] = 1

        device = self.base_model.device()
        batch = {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
        }

        # ---- 3. HF forward ----
        with torch.no_grad():
            hf_out = self.base_model(
                **batch,
                output_hidden_states=True,
                use_cache=False,
            )

        # ---- 4. Extract hidden states ----
        hidden_states = self.extractor(batch, hf_out)

        return {
            # LM-style alignment
            "vllm_hidden_states": hidden_states,
            "full_attention_mask": attention_mask,
            "context_lenghts": torch.tensor(context_lengths, dtype=torch.long),
            "greedy_texts": greedy_texts,
            "greedy_tokens": greedy_tokens,
        }

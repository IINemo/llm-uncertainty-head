from lm_polygraph.stat_calculators.stat_calculator import StatCalculator
from lm_polygraph.utils.model import Model
from .disk_cache import DiskCache

from typing import Dict, List, Tuple
import torch
import time
import numpy as np
import os

import logging
log = logging.getLogger()


class CalculatorInferLuh(StatCalculator):
    def __init__(
        self,
        uncertainty_head,
        n_alternatives=10,
        tokenize=True,
        return_embeddings=True,
        generations_cache_dir=None,
        args_generate=dict(),
        predict_token_uncertainties=True,
        device="cuda"
    ):
        super().__init__()

        self.n_alternatives = n_alternatives
        self._tokenize = tokenize
        self._return_embeddings = return_embeddings
        self.args_generate = args_generate
        self.generations_cache_dir = generations_cache_dir if ((generations_cache_dir is not None) and generations_cache_dir.strip()) else None
        if self.generations_cache_dir is not None:
            os.makedirs(self.generations_cache_dir, exist_ok=True)
        else:
            log.info('Generations cache is None')

        self.uncertainty_head = uncertainty_head.to(device)
        self.uncertainty_head.eval()
        self.output_attentions = self.uncertainty_head.output_attentions
        # Request router logits only when the underlying feature extractor needs them.
        self.output_router_logits = getattr(
            self.uncertainty_head, "output_router_logits", lambda: False
        )()
        self.predict_token_uncertainties = predict_token_uncertainties

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        """
        Returns the statistics and dependencies for the calculator.
        """

        return [
            "hidden_states",
            "greedy_log_probs",
            "greedy_logits",
            "greedy_tokens",
            "greedy_tokens_alternatives",
            "greedy_texts",
            "greedy_log_likelihoods",
            "uncertainty_logits",
            "uhead_features",
            "input_texts",
            "input_tokens",
        ], []

    def infer_cached(
        self,
        cache: DiskCache,
        texts: List[str],
        model: Model,
        max_new_tokens: int,
    ) -> Dict:
        log.info("Infering from cache")
        gen_tokens: List[List[int]] = [cache.get(t) for t in texts]

        input_batch: Dict[str, torch.Tensor] = model.tokenize(texts)

        combined_tokens = [
            it.tolist() + ht for it, ht in zip(input_batch["input_ids"], gen_tokens)
        ]
        combined_batch = model.tokenizer.pad(
            {"input_ids": combined_tokens},
            padding=True,
            return_tensors="pt",
        )
        combined_batch = {k: v.to(model.device()) for k, v in combined_batch.items()}

        with torch.no_grad():
            out = model(
                **combined_batch,
                output_attentions=self.output_attentions,
                output_router_logits=self.output_router_logits,
                output_hidden_states=True,
            )
            logits = out.logits.log_softmax(-1) # Why log_softmax?
            
        cut_logits = []
        cut_sequences = []
        cut_texts = []
        cut_alternatives = []
        output_bounds = []
        for i in range(len(texts)):
            begin_pos = len(input_batch["input_ids"][i])
            end_pos = begin_pos + len(gen_tokens[i])
            cut_sequences.append(gen_tokens[i])
            cut_texts.append(model.tokenizer.decode(gen_tokens[i]))
            cut_logits.append(logits[i][begin_pos - 1 : end_pos - 1].cpu().numpy())
            output_bounds.append((begin_pos - 1, end_pos - 1))
            cut_alternatives.append([[] for _ in range(begin_pos, end_pos)])

            for j in range(begin_pos, end_pos):
                lt = logits[i, j - 1, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)[
                    -self.n_alternatives :
                ]
                best_tokens = best_tokens[np.argsort(-lt[best_tokens])].tolist()

                # as hyp_texts are not necessarily greedy, so
                # need to make sure that first token is from hyp_texts
                cur_token = gen_tokens[i][j - begin_pos]
                if cur_token not in best_tokens:
                    best_tokens = [cur_token] + best_tokens[:-1]
                else:
                    best_tokens = [cur_token] + [
                        t for t in best_tokens if t != cur_token
                    ]

                for t in best_tokens:
                    cut_alternatives[-1][j - begin_pos].append((t, lt[t].item()))

        ll = []
        for i in range(len(texts)):
            log_probs = cut_logits[i]
            tokens = cut_sequences[i]
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        result_dict = {
            "input_texts": texts,
            "input_tokens": input_batch["input_ids"].tolist(),
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_log_likelihoods": ll,
            "logits": logits,
        }


        out["context_lengths"] = torch.tensor([len(it) for it in input_batch["input_ids"]])
        combined_batch["context_lenghts"] = out["context_lengths"]

        if self.predict_token_uncertainties:
            uncertainty_logits = self.uncertainty_head(combined_batch, out)
            result_dict["uncertainty_logits"] = [
                ue[output_bounds[i][0]: output_bounds[i][1]]  # only output tokens
                for i, ue in enumerate(uncertainty_logits.cpu().detach().squeeze(-1))
            ]
            assert all(
                len(greedy_tokens) == len(ue_logits)
                for greedy_tokens, ue_logits in zip(
                    result_dict["greedy_tokens"],
                    result_dict["uncertainty_logits"]
                )
            )

        else:
            result_dict["uhead_features"] = self.uncertainty_head.feature_extractor(combined_batch, out)
            result_dict["llm_inputs"] = combined_batch
            result_dict["full_attention_mask"] = combined_batch['attention_mask']

        return result_dict

    def postprocess_predictions(self, batch, out, tokenizer):
        logits = torch.stack(out.scores, dim=1)
        sequences = out.sequences

        cut_logits = []
        cut_sequences = []
        cut_texts = []
        cut_alternatives = []
        ll = []
        batch_size = batch['input_ids'].shape[0]
        for i in range(batch_size):
            idx = batch["input_ids"].shape[1]
            seq = sequences[i, idx:].cpu()
            length, text_length = len(seq), len(seq)
            for j in range(len(seq)):
                if seq[j] == tokenizer.eos_token_id:
                    length = j + 1
                    text_length = j
                    break

            cut_sequences.append(seq[:length].tolist())
            cut_texts.append(tokenizer.decode(seq[:text_length]))
            cut_logits.append(logits[i, :length, :].cpu().numpy())
            cut_alternatives.append([[] for _ in range(length)])
            for j in range(length):
                lt = logits[i, j, :].cpu().numpy()
                best_tokens = np.argpartition(lt, -self.n_alternatives)
                ln = len(best_tokens)
                best_tokens = best_tokens[ln - self.n_alternatives : ln]
                for t in best_tokens:
                    cut_alternatives[-1][j].append((t.item(), lt[t].item()))

                cut_alternatives[-1][j].sort(
                    key=lambda x: x[0] == cut_sequences[-1][j],
                    reverse=True,
                )

            log_probs = cut_logits[-1]
            tokens = cut_sequences[-1]
            assert len(tokens) == len(log_probs)
            ll.append([log_probs[j, tokens[j]] for j in range(len(log_probs))])

        result_dict = {
            "input_tokens": batch["input_ids"].to("cpu").tolist(),
            "greedy_log_probs": cut_logits,
            "greedy_tokens": cut_sequences,
            "greedy_tokens_alternatives": cut_alternatives,
            "greedy_texts": cut_texts,
            "greedy_log_likelihoods": ll,
            "logits": logits[:, :-1, :] # Why?
        }

        return result_dict

    def __call__(
        self,
        dependencies: Dict[str, np.array],
        texts: List[str],
        model: Model,
        max_new_tokens: int = 100,  # TODO: move to args_generate
        **kwargs,
    ) -> Dict[str, np.ndarray]:
        if "num_return_sequences" in kwargs:
            num_return_sequences = kwargs["num_return_sequences"]
            kwargs.pop("num_return_sequences")
            texts = [t for t in texts for _ in range(num_return_sequences)]

        cache = None
        if self.generations_cache_dir is not None:
            cache = DiskCache(
                os.path.join(
                    self.generations_cache_dir,
                    "generations_" + model.model_path.replace("/", "__"),
                )
            )
        if cache is not None and all(cache.contains(t) for t in texts):
            return self.infer_cached(cache, texts, model, max_new_tokens)


        if self._tokenize:
            batch: Dict[str, torch.Tensor] = model.tokenize(texts)
        else:
            batch = texts

        device_batch = batch.to(model.device())
        start_time = time.time()

        gen_args = {
            "output_scores": True,
            "return_dict_in_generate": True,
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 2,
            "output_attentions": self.output_attentions,
            "output_router_logits": self.output_router_logits,
            "output_hidden_states": True,
            "num_return_sequences": 1,
            "do_sample": False,
        }
        gen_args.update(kwargs)

        log.info(f"Generating on device={model.device()} with args {gen_args}")

        with torch.no_grad():
            out = model.generate(
                **device_batch,
                **gen_args,
            )
        log.info(f"Done generating in {round(time.time() - start_time, 2)} seconds")

        result_dict = self.postprocess_predictions(batch, out, model.tokenizer)
        result_dict["input_texts"] = texts
        result_dict["out"] = out

        if cache is not None:
            for i in range(len(texts)):
                cache.get(texts[i], lambda: result_dict["greedy_tokens"][i])


        output_bounds = []
        full_attn_mask = torch.zeros_like(out.sequences).bool()
        batch_size = batch['input_ids'].shape[0]
        for i in range(batch_size):
            idx = batch["input_ids"].shape[1]
            full_attn_mask[i, :idx] = batch["attention_mask"][i] # TODO: take into account <eos>
            length = len(result_dict["greedy_tokens"][i])
            #length = len(out.sequences[i, idx:])
            full_attn_mask[i][idx : idx + length] = 1
            output_bounds.append((idx - 1, idx + length - 1))

        out["full_attention_mask"] = full_attn_mask
        out["context_lengths"] = torch.tensor([len(it) for it in batch["input_ids"]])
        batch["context_lenghts"] = out["context_lengths"]
        if self.predict_token_uncertainties:
            with torch.no_grad():
                uncertainty_logits = self.uncertainty_head(batch, out)
                result_dict["uncertainty_logits"] = [
                    ue[output_bounds[i][0]: output_bounds[i][1]]  # only output tokens
                    for i, ue in enumerate(uncertainty_logits.cpu().detach().squeeze(-1))
                ]

            assert all(
                len(greedy_tokens) == len(ue_logits)
                for greedy_tokens, ue_logits in zip(
                    result_dict["greedy_tokens"],
                    result_dict["uncertainty_logits"]
                )
            )

        else:
            result_dict["uhead_features"] = self.uncertainty_head.feature_extractor(batch, out)
            result_dict["llm_inputs"] = batch
            result_dict["full_attention_mask"] = full_attn_mask

        return result_dict

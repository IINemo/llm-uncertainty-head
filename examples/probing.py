#!/usr/bin/env python3
import hashlib
import io
import json, argparse
import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from datasets import load_from_disk, DatasetDict, concatenate_datasets


def _parse_int_list(spec: str) -> List[int]:
    if spec is None:
        return []
    parts = [p.strip() for p in str(spec).split(",")]
    out = []
    for p in parts:
        if not p:
            continue
        out.append(int(p))
    return out


def load_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


NEW_LABELS = [
    "label_unrec_unknown_entity",
    "label_unrec_low_quality",
    "label_rec_fact_unknown",
    "label_rec_fact_known",
    "label_unrec_but_guessed_fact",
]

def build_recognition(dataset: str, model_tag: str) -> List[dict]:
    tag = model_tag.replace("/", "_").replace("-", "_")
    jpath = Path(f"out_{dataset}/type2_bias/type2_{dataset}_{tag}.jsonl")
    recs = []
    for o in load_jsonl(jpath):
        if o.get("type") != "pos":
            continue
        recs.append({"task": "recognition", "text": o["question"], "label": 1 if o.get("is_correct") else 0})
    return recs

def build_factual(dataset: str, model_tag: str) -> List[dict]:
    tag = model_tag.replace("/", "_").replace("-", "_")
    jpath = Path(f"out_{dataset}/feature_eval/feature_{dataset}_{tag}.jsonl")
    recs = []
    for o in load_jsonl(jpath):
        recs.append({"task": "factual", "text": o["feature_question"], "label": 1 if o.get("judge_correct") else 0})
    return recs


def load_hf_rows(dataset_paths: List[Path], split: str) -> List[dict]:
    chunks = []
    split_names = [s.strip() for s in split.split(",")] if split != "all" else None
    for p in dataset_paths:
        ds_obj = load_from_disk(str(p))
        if isinstance(ds_obj, DatasetDict):
            if split_names is None:
                names = [n for n in ["train", "validation", "test"] if n in ds_obj]
            else:
                names = [n for n in split_names if n in ds_obj]
            if not names:
                raise ValueError(f"No requested splits found in {p}. Available: {list(ds_obj.keys())}")
            chunks.extend([ds_obj[n] for n in names])
        else:
            chunks.append(ds_obj)
    if not chunks:
        return []
    ds = chunks[0] if len(chunks) == 1 else concatenate_datasets(chunks)
    return list(ds)


def _hash_lines(lines):
    h = hashlib.sha1()
    for x in lines:
        h.update(str(x).encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _cache_path_hf(args, rows):
    ds_paths = [str(p.resolve()) for p in (args.hf_dataset or [])]
    row_sig = []
    for r in rows:
        row_sig.append((
            r.get("dataset", ""),
            r.get("image_path", ""),
            r.get(args.text_field, ""),
            r.get("subject", r.get("entity_name", "")),
        ))
    signature = _hash_lines([
        "mode=hf",
        f"vlm={args.vlm}",
        f"hf_split={args.hf_split}",
        f"multimodal={args.multimodal}",
        f"text_field={args.text_field}",
        f"image_field={args.image_field}",
        f"fp16={args.fp16}",
        f"bf16={args.bf16}",
        f"paths={ds_paths}",
        f"rows={len(rows)}",
        f"row_sig={_hash_lines(row_sig)}",
    ])
    return args.cache_dir / f"embeds_{signature[:16]}.npy"


def _rows_signature(rows, text_field):
    row_sig = []
    for r in rows:
        row_sig.append((
            r.get("dataset", ""),
            r.get("image_path", ""),
            r.get(text_field, ""),
            r.get("subject", r.get("entity_name", "")),
        ))
    return _hash_lines(row_sig)


def _cache_path_legacy(args, texts, tasks):
    signature = _hash_lines([
        "mode=legacy",
        f"vlm={args.vlm}",
        f"multimodal={args.multimodal}",
        f"fp16={args.fp16}",
        f"bf16={args.bf16}",
        f"n={len(texts)}",
        f"text_sig={_hash_lines(texts)}",
        f"task_sig={_hash_lines(tasks)}",
    ])
    return args.cache_dir / f"embeds_{signature[:16]}.npy"


def _token_cache_dir_hf(args, rows):
    ds_paths = [str(p.resolve()) for p in (args.hf_dataset or [])]
    signature = _hash_lines([
        "mode=hf-token-level",
        f"vlm={args.vlm}",
        f"hf_split={args.hf_split}",
        f"multimodal={args.multimodal}",
        f"text_field={args.text_field}",
        f"image_field={args.image_field}",
        f"hidden_dtype={args.token_cache_dtype}",
        f"cache_pregen={args.cache_pregen}",
        f"pregen_hidden_layers={args.pregen_hidden_layers}",
        f"cache_pregen_attn={args.cache_pregen_attn}",
        f"pregen_attn_layers={args.pregen_attn_layers}",
        f"paths={ds_paths}",
        f"rows={len(rows)}",
        f"row_sig={_rows_signature(rows, args.text_field)}",
    ])
    return args.token_cache_dir / f"tokens_{signature[:16]}"


def _token_cache_dir_legacy(args, texts, tasks):
    signature = _hash_lines([
        "mode=legacy-token-level",
        f"vlm={args.vlm}",
        f"multimodal={args.multimodal}",
        f"hidden_dtype={args.token_cache_dtype}",
        f"cache_pregen={args.cache_pregen}",
        f"pregen_hidden_layers={args.pregen_hidden_layers}",
        f"cache_pregen_attn={args.cache_pregen_attn}",
        f"pregen_attn_layers={args.pregen_attn_layers}",
        f"n={len(texts)}",
        f"text_sig={_hash_lines(texts)}",
        f"task_sig={_hash_lines(tasks)}",
    ])
    return args.token_cache_dir / f"tokens_{signature[:16]}"


class TokenStateCacheWriter:
    def __init__(
        self,
        out_dir: Path,
        hidden_dtype: str = "float16",
        overwrite: bool = False,
        store_attention_mask: bool = False,
    ):
        self.out_dir = out_dir
        self.hidden_dtype = hidden_dtype
        self.overwrite = overwrite
        self.store_attention_mask = store_attention_mask
        self.meta_path = self.out_dir / "meta.json"
        self.num_chunks = 0
        self.num_rows = 0

        if overwrite and self.out_dir.exists():
            for p in self.out_dir.glob("chunk_*.npz"):
                p.unlink()
            if self.meta_path.exists():
                self.meta_path.unlink()

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._dtype_np = np.float16 if hidden_dtype == "float16" else np.float32
        self._resume_existing_chunks()

    def _resume_existing_chunks(self):
        if self.overwrite:
            return
        chunk_paths = sorted(self.out_dir.glob("chunk_*.npz"))
        if not chunk_paths:
            return
        rows = 0
        for p in chunk_paths:
            try:
                with np.load(p) as z:
                    rows += int(z["hidden"].shape[0])
            except Exception:
                # Ignore unreadable chunks; user can overwrite to rebuild cleanly.
                continue
        self.num_chunks = len(chunk_paths)
        self.num_rows = rows

    @staticmethod
    def _sanitize_float_array(arr: np.ndarray, out_dtype) -> np.ndarray:
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if out_dtype == np.float16:
            finfo = np.finfo(np.float16)
            arr = np.clip(arr, finfo.min, finfo.max)
        return arr.astype(out_dtype, copy=False)

    def add_batch(self, hidden, input_ids, attention_mask=None, extras: Optional[Dict] = None):
        hidden_np = hidden.detach().float().cpu().numpy()
        hidden_np = self._sanitize_float_array(hidden_np, self._dtype_np)
        input_ids_np = input_ids.detach().cpu().numpy().astype(np.int32, copy=False)

        payload = {
            "hidden": hidden_np,
            "input_ids": input_ids_np,
        }
        if self.store_attention_mask:
            if attention_mask is None:
                attention_mask_np = np.ones_like(input_ids_np, dtype=np.uint8)
            else:
                attention_mask_np = attention_mask.detach().cpu().numpy().astype(np.uint8, copy=False)
            payload["attention_mask"] = attention_mask_np
        if extras:
            for k, v in extras.items():
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    arr = v.detach().cpu().numpy()
                else:
                    arr = np.asarray(v)
                if k in {"pregen_hidden", "pregen_attn"}:
                    arr = self._sanitize_float_array(arr.astype(np.float32, copy=False), self._dtype_np)
                elif np.issubdtype(arr.dtype, np.floating):
                    arr = self._sanitize_float_array(arr.astype(np.float32, copy=False), np.float32)
                payload[k] = arr

        out_path = self.out_dir / f"chunk_{self.num_chunks:06d}.npz"
        np.savez_compressed(out_path, **payload)
        self.num_chunks += 1
        self.num_rows += hidden_np.shape[0]

    def finalize(self, meta_extra: Optional[Dict] = None):
        meta = {
            "num_chunks": self.num_chunks,
            "num_rows": self.num_rows,
            "hidden_dtype": self.hidden_dtype,
            "store_attention_mask": self.store_attention_mask,
        }
        if meta_extra:
            meta.update(meta_extra)
        with self.meta_path.open("w") as f:
            json.dump(meta, f, indent=2)


def _resolve_hidden_layer_idx(hidden_states, layer_id: int) -> int:
    # hidden_states includes embedding output at index 0.
    if layer_id >= 0:
        idx = layer_id + 1
    else:
        idx = len(hidden_states) + layer_id
    if idx < 0 or idx >= len(hidden_states):
        raise IndexError(f"Hidden layer index {layer_id} out of range for {len(hidden_states)-1} transformer layers")
    return idx


def _resolve_attn_layer_idx(attentions, layer_id: int) -> int:
    # attentions only includes transformer layers.
    idx = layer_id if layer_id >= 0 else len(attentions) + layer_id
    if idx < 0 or idx >= len(attentions):
        raise IndexError(f"Attention layer index {layer_id} out of range for {len(attentions)} layers")
    return idx


def _extract_pregen_extras(outputs, input_ids, attention_mask, pregen_cfg):
    if pregen_cfg is None:
        return None
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    end_idx = attention_mask.long().sum(dim=1).clamp(min=1) - 1

    extras = {"pregen_end_idx": end_idx.detach().cpu().numpy().astype(np.int32)}

    hidden_layers = pregen_cfg.get("hidden_layers", [])
    if hidden_layers:
        hs = outputs.hidden_states
        hidden_vecs = []
        for b in range(input_ids.shape[0]):
            e = int(end_idx[b].item())
            parts = []
            for lid in hidden_layers:
                idx = _resolve_hidden_layer_idx(hs, lid)
                parts.append(hs[idx][b, e, :].detach().float().cpu())
            hidden_vecs.append(torch.cat(parts, dim=0))
        extras["pregen_hidden"] = torch.stack(hidden_vecs, dim=0).numpy()

    if pregen_cfg.get("include_attn", False):
        attn_layers = pregen_cfg.get("attn_layers", [])
        atts = outputs.attentions
        if atts is None or any(a is None for a in atts):
            raise RuntimeError(
                "Attention extraction requested, but attentions are unavailable. "
                "Use eager attention (model.set_attn_implementation('eager')) "
                "or disable --cache-pregen-attn."
            )
        attn_vecs = []
        for b in range(input_ids.shape[0]):
            valid_len = int(attention_mask[b].sum().item())
            valid_len = max(valid_len, 1)
            e = min(int(end_idx[b].item()), valid_len - 1)
            parts = []
            for lid in attn_layers:
                aidx = _resolve_attn_layer_idx(atts, lid)
                row = atts[aidx][b, :, e, :valid_len].detach().float()
                row = row / row.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                entropy = -(row * torch.log(row.clamp_min(1e-12))).sum(dim=-1)
                self_w = row[:, e]
                max_w = row.max(dim=-1).values
                parts.append(torch.cat([entropy, self_w, max_w], dim=0).cpu())
            attn_vecs.append(torch.cat(parts, dim=0))
        extras["pregen_attn"] = torch.stack(attn_vecs, dim=0).numpy()

    return extras


def load_or_compute_embeddings(cache_path, compute_fn, use_cache=False, overwrite=False):
    if use_cache and cache_path.exists() and not overwrite:
        print(f"Loading cached embeddings: {cache_path}")
        return np.load(cache_path)
    embeds = compute_fn()
    if use_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeds)
        print(f"Saved cached embeddings: {cache_path}")
    return embeds

def embed_text(
    texts,
    tokenizer,
    model,
    batch_size,
    device,
    token_writer: Optional[TokenStateCacheWriter] = None,
    pregen_cfg: Optional[Dict] = None,
    start_idx: int = 0,
    return_embeddings: bool = True,
):
    feats = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(start_idx, len(texts), batch_size), desc="Embedding"):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
            out = model(
                **enc,
                output_hidden_states=True,
                output_attentions=bool(pregen_cfg and pregen_cfg.get("include_attn", False)),
            )
            hidden = out.hidden_states[-1]  # [batch, seq, dim]
            if token_writer is not None:
                extras = _extract_pregen_extras(out, enc["input_ids"], enc.get("attention_mask"), pregen_cfg)
                token_writer.add_batch(hidden, enc["input_ids"], enc.get("attention_mask"), extras=extras)
            if return_embeddings:
                vec = hidden[:, 0, :].float().cpu()
                feats.append(vec)
    if not return_embeddings:
        return None
    if not feats:
        raise RuntimeError("No text features extracted.")
    return torch.cat(feats, dim=0).numpy()


def _to_pil(img):
    if img is None:
        return None
    if isinstance(img, PILImage.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        if img.get("bytes") is not None:
            return PILImage.open(io.BytesIO(img["bytes"])).convert("RGB")
        if img.get("path"):
            return PILImage.open(img["path"]).convert("RGB")
    if isinstance(img, str):
        return PILImage.open(img).convert("RGB")
    return None


def _move_to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if hasattr(v, "to") else v
    return out


def embed_multimodal(
    rows,
    text_field,
    image_field,
    processor,
    model,
    batch_size,
    device,
    model_id=None,
    token_writer: Optional[TokenStateCacheWriter] = None,
    pregen_cfg: Optional[Dict] = None,
    start_idx: int = 0,
    return_embeddings: bool = True,
):
    feats = []
    model.eval()
    model_id_l = (model_id or "").lower()
    is_gemma3 = "gemma-3" in model_id_l
    is_llama_vision = ("meta-llama" in model_id_l and "vision" in model_id_l) or ("llama-3.2-11b-vision" in model_id_l)
    is_qwen3 = "qwen3-vl" in model_id_l
    is_qwen2p5 = "qwen2.5-vl" in model_id_l
    with torch.no_grad():
        if is_gemma3:
            # Match eval.py formatting for Gemma-3 exactly (system + user(image,text)).
            for i in tqdm(range(start_idx, len(rows)), desc="Embedding-mm"):
                r = rows[i]
                text = str(r.get(text_field, ""))
                image = _to_pil(r.get(image_field))
                if image is None:
                    continue
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text},
                        ],
                    },
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)
                out = model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=bool(pregen_cfg and pregen_cfg.get("include_attn", False)),
                    use_cache=False,
                )
                hidden = out.hidden_states[-1]
                if token_writer is not None:
                    extras = _extract_pregen_extras(out, inputs["input_ids"], inputs.get("attention_mask"), pregen_cfg)
                    token_writer.add_batch(hidden, inputs["input_ids"], inputs.get("attention_mask"), extras=extras)
                if return_embeddings:
                    feats.append(hidden[:, 0, :].float().cpu())
            if return_embeddings:
                if not feats:
                    raise RuntimeError("No multimodal features extracted. Check image column and inputs.")
                return torch.cat(feats, dim=0).numpy()
            return None

        if is_llama_vision:
            # Match eval.py formatting for Llama-3.2-Vision exactly (system + user(image,text)).
            for i in tqdm(range(start_idx, len(rows)), desc="Embedding-mm"):
                r = rows[i]
                text = str(r.get(text_field, ""))
                image = _to_pil(r.get(image_field))
                if image is None:
                    continue
                messages = [
                    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": text},
                        ],
                    },
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)
                out = model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=bool(pregen_cfg and pregen_cfg.get("include_attn", False)),
                    use_cache=False,
                )
                hidden = out.hidden_states[-1]
                if token_writer is not None:
                    extras = _extract_pregen_extras(out, inputs["input_ids"], inputs.get("attention_mask"), pregen_cfg)
                    token_writer.add_batch(hidden, inputs["input_ids"], inputs.get("attention_mask"), extras=extras)
                if return_embeddings:
                    feats.append(hidden[:, 0, :].float().cpu())
            if return_embeddings:
                if not feats:
                    raise RuntimeError("No multimodal features extracted. Check image column and inputs.")
                return torch.cat(feats, dim=0).numpy()
            return None

        if is_qwen3 or is_qwen2p5:
            # Match eval.py formatting for qwen3/qwen2.5 exactly (user(image,text)).
            for i in tqdm(range(start_idx, len(rows)), desc="Embedding-mm"):
                r = rows[i]
                text = str(r.get(text_field, ""))
                image = _to_pil(r.get(image_field))
                if image is None:
                    continue
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text},
                    ],
                }]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(device)
                out = model(
                    **inputs,
                    output_hidden_states=True,
                    output_attentions=bool(pregen_cfg and pregen_cfg.get("include_attn", False)),
                    use_cache=False,
                )
                hidden = out.hidden_states[-1]
                if token_writer is not None:
                    extras = _extract_pregen_extras(out, inputs["input_ids"], inputs.get("attention_mask"), pregen_cfg)
                    token_writer.add_batch(hidden, inputs["input_ids"], inputs.get("attention_mask"), extras=extras)
                if return_embeddings:
                    feats.append(hidden[:, 0, :].float().cpu())
            if return_embeddings:
                if not feats:
                    raise RuntimeError("No multimodal features extracted. Check image column and inputs.")
                return torch.cat(feats, dim=0).numpy()
            return None

        for i in tqdm(range(start_idx, len(rows), batch_size), desc="Embedding-mm"):
            batch_rows = rows[i:i + batch_size]
            texts = [str(r.get(text_field, "")) for r in batch_rows]
            images = [_to_pil(r.get(image_field)) for r in batch_rows]
            valid = [(t, im) for t, im in zip(texts, images) if im is not None]
            if not valid:
                continue
            texts, images = zip(*valid)

            prompts = list(texts)
            if hasattr(processor, "apply_chat_template"):
                prompts = []
                for t in texts:
                    messages = [{
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": t},
                        ],
                    }]
                    prompts.append(
                        processor.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                    )

            enc = processor(
                text=prompts,
                images=list(images),
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            enc = _move_to_device(enc, device)
            out = model(
                **enc,
                output_hidden_states=True,
                output_attentions=bool(pregen_cfg and pregen_cfg.get("include_attn", False)),
            )
            hidden = out.hidden_states[-1]
            if token_writer is not None:
                extras = _extract_pregen_extras(out, enc["input_ids"], enc.get("attention_mask"), pregen_cfg)
                token_writer.add_batch(hidden, enc["input_ids"], enc.get("attention_mask"), extras=extras)
            if return_embeddings:
                vec = hidden[:, 0, :].float().cpu()
                feats.append(vec)

    if not return_embeddings:
        return None
    if not feats:
        raise RuntimeError("No multimodal features extracted. Check image column and inputs.")
    return torch.cat(feats, dim=0).numpy()

def train_eval_binary(task_name, X, y, balance_train=False, seed=42, max_iter=2000, use_scaler=True):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        print(f"{task_name}: skipped (single class, n={len(y)})")
        return
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    if balance_train:
        # undersample majority
        pos_idx = np.where(ytr==1)[0]
        neg_idx = np.where(ytr==0)[0]
        if len(pos_idx) < len(neg_idx):
            neg_idx = np.random.RandomState(seed).choice(neg_idx, size=len(pos_idx), replace=False)
        elif len(neg_idx) < len(pos_idx):
            pos_idx = np.random.RandomState(seed).choice(pos_idx, size=len(neg_idx), replace=False)
        keep = np.concatenate([pos_idx, neg_idx])
        Xtr, ytr = Xtr[keep], ytr[keep]
    clf = LogisticRegression(max_iter=max_iter, class_weight="balanced")
    if use_scaler:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
    else:
        model = clf

    model.fit(Xtr, ytr)
    pred = model.predict(Xte); prob = model.predict_proba(Xte)[:,1]
    acc = accuracy_score(yte, pred)
    roc_auc = roc_auc_score(yte, prob)
    precs, recs, _ = precision_recall_curve(yte, prob)
    pr_auc = auc(recs, precs)
    ap = average_precision_score(yte, prob)
    f1 = f1_score(yte, pred)
    print(f"{task_name}: acc={acc:.3f} roc_auc={roc_auc:.3f} pr_auc={pr_auc:.3f} ap={ap:.3f} f1={f1:.3f} (n={len(y)})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", required=True, help="HF id or local model path")
    ap.add_argument("--hf-dataset", nargs="+", type=Path, help="Local HF dataset path(s) from hf_data.py")
    ap.add_argument("--hf-split", default="all", help="train,validation,test,all or comma-separated list")
    ap.add_argument("--text-field", default="question", help="Text field for probing in HF dataset mode")
    ap.add_argument("--image-field", default="image", help="Image field for multimodal probing")
    ap.add_argument("--multimodal", action="store_true", help="Use image+text VLM features")
    ap.add_argument("--probe-labels", default="all", help="'all' or comma-separated label columns")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fp16", action="store_true", help="Use float16")
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--hf-token", default=None, help="HF token (or set HF_TOKEN/HUGGINGFACE_HUB_TOKEN env var)")
    ap.add_argument("--balance-train", action="store_true", help="Undersample majority class in training split")
    ap.add_argument("--max-iter", type=int, default=2000, help="Max iterations for logistic regression probe")
    ap.add_argument("--no-scale", action="store_true", help="Disable StandardScaler before logistic regression")
    ap.add_argument("--cache-features", action=argparse.BooleanOptionalAction, default=True, help="Cache extracted embeddings to speed repeated probe runs")
    ap.add_argument("--cache-dir", type=Path, default=Path("probe_cache"), help="Directory for embedding cache files")
    ap.add_argument("--overwrite-cache", action="store_true", help="Recompute features even if a cache file exists")
    ap.add_argument("--cache-token-level", action="store_true", help="Also cache full token-level hidden states (per batch) to disk")
    ap.add_argument("--token-cache-only", action="store_true", help="Only write/continue token-level cache; skip probe embedding/evaluation")
    ap.add_argument("--token-cache-dir", type=Path, default=Path("probe_cache_token_level"), help="Directory for token-level hidden-state cache")
    ap.add_argument("--token-cache-dtype", choices=["float16", "float32"], default="float16", help="Dtype for token-level hidden-state cache")
    ap.add_argument("--token-cache-overwrite", action="store_true", help="Overwrite token-level cache if it already exists")
    ap.add_argument("--cache-attention-mask", action="store_true", help="Store attention_mask in token cache (optional; disabled by default to save space)")
    ap.add_argument("--cache-pregen", action="store_true", help="When caching token-level states, also store prompt-end features from selected layers")
    ap.add_argument("--pregen-hidden-layers", default="-1,-4,-8,-12", help="Comma-separated transformer layer ids for prompt-end hidden features")
    ap.add_argument("--cache-pregen-attn", action="store_true", help="Also cache prompt-end attention summaries for selected layers")
    ap.add_argument("--pregen-attn-layers", default="-1,-4,-8,-12", help="Comma-separated transformer layer ids for prompt-end attention summaries")
    args = ap.parse_args()

    pregen_hidden_layers = _parse_int_list(args.pregen_hidden_layers) if args.cache_pregen else []
    pregen_attn_layers = _parse_int_list(args.pregen_attn_layers) if args.cache_pregen_attn else []
    if args.cache_pregen and not pregen_hidden_layers:
        raise ValueError("--cache-pregen is set but no valid --pregen-hidden-layers were provided.")
    if args.cache_pregen_attn and not pregen_attn_layers:
        raise ValueError("--cache-pregen-attn is set but no valid --pregen-attn-layers were provided.")
    pregen_cfg = None
    if args.cache_pregen or args.cache_pregen_attn:
        pregen_cfg = {
            "hidden_layers": pregen_hidden_layers,
            "include_attn": bool(args.cache_pregen_attn),
            "attn_layers": pregen_attn_layers,
        }

    dtype = None
    if args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16

    hf_token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    model_kwargs = {"trust_remote_code": True}
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if hf_token:
        model_kwargs["token"] = hf_token

    tok = None
    processor = None
    if args.multimodal:
        processor = AutoProcessor.from_pretrained(args.vlm, trust_remote_code=True, token=hf_token)
        try:
            model = AutoModelForImageTextToText.from_pretrained(args.vlm, **model_kwargs).to(args.device)
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(args.vlm, **model_kwargs).to(args.device)
    else:
        tok = AutoTokenizer.from_pretrained(args.vlm, trust_remote_code=True, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(args.vlm, **model_kwargs).to(args.device)

    if args.cache_pregen_attn:
        switched = False
        if hasattr(model, "set_attn_implementation"):
            try:
                model.set_attn_implementation("eager")
                switched = True
                print("Set attention implementation to eager for attention caching.")
            except Exception:
                switched = False
        if not switched and hasattr(model, "config"):
            try:
                if hasattr(model.config, "_attn_implementation"):
                    model.config._attn_implementation = "eager"
                    switched = True
                    print("Set model.config._attn_implementation='eager' for attention caching.")
            except Exception:
                switched = False
        if not switched:
            print(
                "Warning: could not switch model to eager attention. "
                "--cache-pregen-attn may fail on this model/runtime."
            )

    if args.hf_dataset:
        rows = load_hf_rows(args.hf_dataset, args.hf_split)
        if not rows:
            raise ValueError("No rows loaded from --hf-dataset.")
        if args.text_field not in rows[0]:
            raise ValueError(f"Text field '{args.text_field}' not found in dataset columns.")
        if args.multimodal and args.image_field not in rows[0]:
            raise ValueError(f"Image field '{args.image_field}' not found in dataset columns.")

        labels = NEW_LABELS if args.probe_labels == "all" else [s.strip() for s in args.probe_labels.split(",")]
        missing = [c for c in labels if c not in rows[0]]
        if missing:
            raise ValueError(f"Missing requested label columns: {missing}")

        token_cache_writer = None
        token_cache_force_compute = False
        token_cache_path = None

        if args.multimodal:
            # Keep only rows with readable images for aligned (X, y) pairs.
            rows = [r for r in rows if _to_pil(r.get(args.image_field)) is not None]
            if not rows:
                raise ValueError("No rows with valid images after filtering.")

        if args.cache_token_level:
            token_cache_path = _token_cache_dir_hf(args, rows)
            token_meta_path = token_cache_path / "meta.json"
            need_write = args.token_cache_overwrite or (not token_meta_path.exists())
            if need_write:
                token_cache_writer = TokenStateCacheWriter(
                    token_cache_path,
                    hidden_dtype=args.token_cache_dtype,
                    overwrite=args.token_cache_overwrite,
                    store_attention_mask=args.cache_attention_mask,
                )
                token_cache_force_compute = True
                print(f"Writing token-level cache to: {token_cache_path}")
                if token_cache_writer.num_rows > 0:
                    print(f"Resuming token cache from row {token_cache_writer.num_rows}/{len(rows)}")
            else:
                print(f"Token-level cache already exists: {token_cache_path}")

        compute_overwrite = args.overwrite_cache or token_cache_force_compute
        cache_path = _cache_path_hf(args, rows)

        if args.token_cache_only:
            if not args.cache_token_level:
                raise ValueError("--token-cache-only requires --cache-token-level.")
            if token_cache_writer is None:
                print("Token-level cache already complete; nothing to do.")
                return
            start_idx = token_cache_writer.num_rows
            if args.multimodal:
                embed_multimodal(
                    rows,
                    text_field=args.text_field,
                    image_field=args.image_field,
                    processor=processor,
                    model=model,
                    batch_size=args.batch_size,
                    device=args.device,
                    model_id=args.vlm,
                    token_writer=token_cache_writer,
                    pregen_cfg=pregen_cfg,
                    start_idx=start_idx,
                    return_embeddings=False,
                )
            else:
                texts = [str(r.get(args.text_field, "")) for r in rows]
                embed_text(
                    texts,
                    tok,
                    model,
                    args.batch_size,
                    args.device,
                    token_writer=token_cache_writer,
                    pregen_cfg=pregen_cfg,
                    start_idx=start_idx,
                    return_embeddings=False,
                )
            token_cache_writer.finalize(
                {
                    "mode": "hf",
                    "vlm": args.vlm,
                    "multimodal": args.multimodal,
                    "rows": len(rows),
                    "hf_split": args.hf_split,
                    "text_field": args.text_field,
                    "image_field": args.image_field,
                    "hf_dataset_paths": [str(p.resolve()) for p in (args.hf_dataset or [])],
                    "row_signature_sha1": _rows_signature(rows, args.text_field),
                    "cache_pregen": bool(args.cache_pregen),
                    "pregen_hidden_layers": pregen_hidden_layers,
                    "cache_pregen_attn": bool(args.cache_pregen_attn),
                    "pregen_attn_layers": pregen_attn_layers,
                }
            )
            print(f"Token-level cache ready at: {token_cache_path}")
            return

        needs_compute = not (args.cache_features and cache_path.exists() and not compute_overwrite)
        if token_cache_writer is not None and token_cache_writer.num_rows > 0 and needs_compute:
            raise ValueError(
                "Found partial token cache resume state while probe embeddings need recompute. "
                "Use --token-cache-only to resume cache writing, or use --token-cache-overwrite to restart."
            )

        if args.multimodal:
            embeds = load_or_compute_embeddings(
                cache_path,
                lambda: embed_multimodal(
                    rows,
                    text_field=args.text_field,
                    image_field=args.image_field,
                    processor=processor,
                    model=model,
                    batch_size=args.batch_size,
                    device=args.device,
                    model_id=args.vlm,
                    token_writer=token_cache_writer,
                    pregen_cfg=pregen_cfg,
                ),
                use_cache=args.cache_features,
                overwrite=compute_overwrite,
            )
        else:
            texts = [str(r.get(args.text_field, "")) for r in rows]
            embeds = load_or_compute_embeddings(
                cache_path,
                lambda: embed_text(
                    texts,
                    tok,
                    model,
                    args.batch_size,
                    args.device,
                    token_writer=token_cache_writer,
                    pregen_cfg=pregen_cfg,
                ),
                use_cache=args.cache_features,
                overwrite=compute_overwrite,
            )

        if token_cache_writer is not None:
            token_cache_writer.finalize(
                {
                    "mode": "hf",
                    "vlm": args.vlm,
                    "multimodal": args.multimodal,
                    "rows": len(rows),
                    "hf_split": args.hf_split,
                    "text_field": args.text_field,
                    "image_field": args.image_field,
                    "hf_dataset_paths": [str(p.resolve()) for p in (args.hf_dataset or [])],
                    "row_signature_sha1": _rows_signature(rows, args.text_field),
                    "cache_pregen": bool(args.cache_pregen),
                    "pregen_hidden_layers": pregen_hidden_layers,
                    "cache_pregen_attn": bool(args.cache_pregen_attn),
                    "pregen_attn_layers": pregen_attn_layers,
                }
            )

        embeds = np.nan_to_num(embeds, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"Loaded {len(rows)} rows from HF dataset. Running probes for: {labels}")
        for col in labels:
            y = np.array([int(r.get(col, 0)) for r in rows], dtype=int)
            train_eval_binary(
                col,
                embeds,
                y,
                balance_train=args.balance_train,
                max_iter=args.max_iter,
                use_scaler=not args.no_scale,
            )
    else:
        recs = []
        for ds in ["inat", "popvqa"]:
            recs.extend(build_recognition(ds, args.vlm))
            recs.extend(build_factual(ds, args.vlm))

        texts = [r["text"] for r in recs]
        labels = np.array([r["label"] for r in recs])
        tasks = [r["task"] for r in recs]

        token_cache_writer = None
        token_cache_force_compute = False
        if args.cache_token_level:
            token_cache_path = _token_cache_dir_legacy(args, texts, tasks)
            token_meta_path = token_cache_path / "meta.json"
            need_write = args.token_cache_overwrite or (not token_meta_path.exists())
            if need_write:
                token_cache_writer = TokenStateCacheWriter(
                    token_cache_path,
                    hidden_dtype=args.token_cache_dtype,
                    overwrite=args.token_cache_overwrite,
                    store_attention_mask=args.cache_attention_mask,
                )
                token_cache_force_compute = True
                print(f"Writing token-level cache to: {token_cache_path}")
                if token_cache_writer.num_rows > 0:
                    print(f"Resuming token cache from row {token_cache_writer.num_rows}/{len(texts)}")
            else:
                print(f"Token-level cache already exists: {token_cache_path}")

        compute_overwrite = args.overwrite_cache or token_cache_force_compute
        cache_path = _cache_path_legacy(args, texts, tasks)
        if args.token_cache_only:
            if not args.cache_token_level:
                raise ValueError("--token-cache-only requires --cache-token-level.")
            if token_cache_writer is None:
                print("Token-level cache already complete; nothing to do.")
                return
            embed_text(
                texts,
                tok,
                model,
                args.batch_size,
                args.device,
                token_writer=token_cache_writer,
                pregen_cfg=pregen_cfg,
                start_idx=token_cache_writer.num_rows,
                return_embeddings=False,
            )
            token_cache_writer.finalize(
                {
                    "mode": "legacy",
                    "vlm": args.vlm,
                    "multimodal": args.multimodal,
                    "rows": len(texts),
                    "text_signature_sha1": _hash_lines(texts),
                    "task_signature_sha1": _hash_lines(tasks),
                    "cache_pregen": bool(args.cache_pregen),
                    "pregen_hidden_layers": pregen_hidden_layers,
                    "cache_pregen_attn": bool(args.cache_pregen_attn),
                    "pregen_attn_layers": pregen_attn_layers,
                }
            )
            print(f"Token-level cache ready at: {token_cache_path}")
            return

        needs_compute = not (args.cache_features and cache_path.exists() and not compute_overwrite)
        if token_cache_writer is not None and token_cache_writer.num_rows > 0 and needs_compute:
            raise ValueError(
                "Found partial token cache resume state while probe embeddings need recompute. "
                "Use --token-cache-only to resume cache writing, or use --token-cache-overwrite to restart."
            )

        embeds = load_or_compute_embeddings(
            cache_path,
            lambda: embed_text(
                texts,
                tok,
                model,
                args.batch_size,
                args.device,
                token_writer=token_cache_writer,
                pregen_cfg=pregen_cfg,
            ),
            use_cache=args.cache_features,
            overwrite=compute_overwrite,
        )

        if token_cache_writer is not None:
            token_cache_writer.finalize(
                {
                    "mode": "legacy",
                    "vlm": args.vlm,
                    "multimodal": args.multimodal,
                    "rows": len(texts),
                    "text_signature_sha1": _hash_lines(texts),
                    "task_signature_sha1": _hash_lines(tasks),
                    "cache_pregen": bool(args.cache_pregen),
                    "pregen_hidden_layers": pregen_hidden_layers,
                    "cache_pregen_attn": bool(args.cache_pregen_attn),
                    "pregen_attn_layers": pregen_attn_layers,
                }
            )

        embeds = np.nan_to_num(embeds, nan=0.0, posinf=0.0, neginf=0.0)

        train_eval_binary(
            "recognition",
            embeds[[i for i, t in enumerate(tasks) if t == "recognition"]],
            labels[[i for i, t in enumerate(tasks) if t == "recognition"]],
            balance_train=args.balance_train,
            max_iter=args.max_iter,
            use_scaler=not args.no_scale,
        )
        train_eval_binary(
            "factual",
            embeds[[i for i, t in enumerate(tasks) if t == "factual"]],
            labels[[i for i, t in enumerate(tasks) if t == "factual"]],
            balance_train=args.balance_train,
            max_iter=args.max_iter,
            use_scaler=not args.no_scale,
        )

if __name__ == "__main__":
    main()

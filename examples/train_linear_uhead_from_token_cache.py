#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Allow importing local luh package from llm-uncertainty-head/.
REPO_ROOT = Path(__file__).resolve().parent
LUH_ROOT = REPO_ROOT / "llm-uncertainty-head"
if str(LUH_ROOT) not in sys.path:
    sys.path.insert(0, str(LUH_ROOT))

from omegaconf import OmegaConf
from luh.heads.linear_head_claim import LinearHeadClaim


def _json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _hash_lines(lines):
    import hashlib
    h = hashlib.sha1()
    for x in lines:
        h.update(str(x).encode("utf-8", errors="ignore"))
        h.update(b"\n")
    return h.hexdigest()


def _rows_signature(rows, text_field: str) -> str:
    row_sig = []
    for r in rows:
        row_sig.append((
            r.get("dataset", ""),
            r.get("image_path", ""),
            r.get(text_field, ""),
            r.get("subject", r.get("entity_name", "")),
        ))
    return _hash_lines(row_sig)


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


def extract_label(row: Dict, label_field: str) -> int:
    val = row.get(label_field, None)
    if isinstance(val, (list, tuple)):
        val = val[0] if len(val) > 0 else 0
    if val is None:
        return 0
    if isinstance(val, bool):
        return int(val)
    return int(val)


def pool_hidden(hidden: np.ndarray, attn_mask: np.ndarray, pooling: str, token_index: int) -> np.ndarray:
    bsz, seq_len, dim = hidden.shape
    if pooling == "first":
        return hidden[:, 0, :].astype(np.float32, copy=False)
    if pooling == "last_nonpad":
        last_idx = np.clip(attn_mask.sum(axis=1).astype(np.int64) - 1, 0, seq_len - 1)
        return hidden[np.arange(bsz), last_idx, :].astype(np.float32, copy=False)
    if pooling == "mean_nonpad":
        mask = attn_mask.astype(np.float32)
        denom = np.clip(mask.sum(axis=1, keepdims=True), 1.0, None)
        pooled = (hidden * mask[:, :, None]).sum(axis=1) / denom
        return pooled.astype(np.float32, copy=False)
    if pooling == "max_nonpad":
        mask = attn_mask.astype(bool)
        safe = hidden.copy()
        safe[~mask] = -np.inf
        pooled = np.max(safe, axis=1)
        pooled[~np.isfinite(pooled)] = 0.0
        return pooled.astype(np.float32, copy=False)
    if pooling == "token_index":
        idx = int(np.clip(token_index, 0, seq_len - 1))
        return hidden[:, idx, :].astype(np.float32, copy=False)
    raise ValueError(f"Unknown pooling: {pooling}")


def _mask_from_pregen_end_idx(end_idx: np.ndarray, seq_len: int) -> np.ndarray:
    end_idx = np.clip(end_idx.astype(np.int64, copy=False), 0, seq_len - 1)
    token_pos = np.arange(seq_len, dtype=np.int64)[None, :]
    return (token_pos <= end_idx[:, None]).astype(np.uint8, copy=False)


def load_features_from_cache(cache_dir: Path, feature_source: str, pooling: str, token_index: int) -> Tuple[np.ndarray, Dict]:
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {cache_dir}")
    with meta_path.open() as f:
        meta = json.load(f)

    chunk_paths = sorted(cache_dir.glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No chunk_*.npz files in {cache_dir}")

    feats = []
    for cp in tqdm(chunk_paths, desc="load-cache", unit="chunk"):
        with np.load(cp) as z:
            if feature_source == "pooled":
                hidden = z["hidden"]  # [B, T, H]
                if "attention_mask" in z:
                    attn_mask = z["attention_mask"]  # [B, T]
                else:
                    if "pregen_end_idx" in z:
                        attn_mask = _mask_from_pregen_end_idx(z["pregen_end_idx"], hidden.shape[1])
                    elif pooling in {"first", "token_index"}:
                        # These pooling modes do not depend on the mask; use a dummy one.
                        attn_mask = np.ones(hidden.shape[:2], dtype=np.uint8)
                    else:
                        raise KeyError(
                            f"Chunk {cp} has no attention_mask. Pooling '{pooling}' needs token masks. "
                            "Provide pregen_end_idx in cache (probing.py --cache-pregen) "
                            "or rebuild cache with probing.py --cache-attention-mask."
                        )
                feats.append(pool_hidden(hidden, attn_mask, pooling, token_index))
            elif feature_source == "pregen_hidden":
                if "pregen_hidden" not in z:
                    raise KeyError(
                        f"Chunk {cp} is missing 'pregen_hidden'. Rebuild cache with probing.py --cache-pregen."
                    )
                feats.append(z["pregen_hidden"].astype(np.float32, copy=False))
            elif feature_source == "pregen_hidden_attn":
                if "pregen_hidden" not in z or "pregen_attn" not in z:
                    raise KeyError(
                        f"Chunk {cp} is missing pregen arrays. Rebuild cache with probing.py "
                        "--cache-pregen --cache-pregen-attn."
                    )
                h = z["pregen_hidden"].astype(np.float32, copy=False)
                a = z["pregen_attn"].astype(np.float32, copy=False)
                feats.append(np.concatenate([h, a], axis=1))
            elif feature_source == "pregen_attn":
                if "pregen_attn" not in z:
                    raise KeyError(
                        f"Chunk {cp} is missing 'pregen_attn'. Rebuild cache with probing.py --cache-pregen-attn."
                    )
                feats.append(z["pregen_attn"].astype(np.float32, copy=False))
            else:
                raise ValueError(f"Unknown feature_source: {feature_source}")
    return np.concatenate(feats, axis=0), meta


def make_split_indices(rows: List[Dict], y: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_values = [str(r.get("split", "")).lower() for r in rows]
    has_named_splits = all(s in {"train", "validation", "test"} for s in split_values)
    if has_named_splits:
        tr = np.array([i for i, s in enumerate(split_values) if s == "train"], dtype=np.int64)
        va = np.array([i for i, s in enumerate(split_values) if s == "validation"], dtype=np.int64)
        te = np.array([i for i, s in enumerate(split_values) if s == "test"], dtype=np.int64)
        if len(tr) > 0 and len(va) > 0 and len(te) > 0:
            return tr, va, te

    idx = np.arange(len(rows))
    try:
        tr, temp = train_test_split(idx, test_size=0.2, random_state=seed, stratify=y)
        va, te = train_test_split(temp, test_size=0.5, random_state=seed, stratify=y[temp])
    except Exception:
        tr, temp = train_test_split(idx, test_size=0.2, random_state=seed, shuffle=True)
        va, te = train_test_split(temp, test_size=0.5, random_state=seed, shuffle=True)
    return np.asarray(tr), np.asarray(va), np.asarray(te)


class DummyFeatureExtractor:
    def __init__(self, in_dim: int):
        self._in_dim = int(in_dim)

    def feature_dim(self):
        return self._in_dim

    def output_attention(self):
        return False


def build_luh_linear_claim_head(in_dim: int):
    cfg = OmegaConf.create(
        {
            "head_type": "linear_claim",
            "feature_extractor": [
                {
                    "name": "luh.feature_extractors.basic_hidden_states",
                    "layer_nums": [-1],
                }
            ],
            "uncertainty_head": None,
            "offline_training": {
                "note": "trained from cached token-level hidden states",
            },
        }
    )
    fe = DummyFeatureExtractor(in_dim)
    head = LinearHeadClaim(fe, cfg=cfg)
    return head, cfg


def compute_metrics(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-logits))
    pred = (probs >= 0.5).astype(np.int64)
    out = {
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "ap": float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        "roc_auc": float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
    }
    return out


def batched_logits(model: nn.Module, x: torch.Tensor, batch_size: int, desc: str = "eval") -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in tqdm(range(0, x.shape[0], batch_size), desc=desc, leave=False):
            outs.append(model(x[i:i + batch_size]).squeeze(-1).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dataset", nargs="+", type=Path, required=True)
    ap.add_argument("--hf-split", default="all")
    ap.add_argument("--token-cache-dir", type=Path, required=True)
    ap.add_argument(
        "--uq-mode",
        choices=["custom", "fullseq", "pregen"],
        default="custom",
        help="Preset for feature extraction: fullseq=pooled mean over prompt tokens, pregen=prompt-end features.",
    )
    ap.add_argument("--label-field", default="verified", help="Binary label field (e.g., verified, label_rec_fact_known)")
    ap.add_argument("--feature-source", choices=["pooled", "pregen_hidden", "pregen_attn", "pregen_hidden_attn"], default="pooled", help="Which cached feature representation to train on")
    ap.add_argument("--pooling", choices=["first", "last_nonpad", "mean_nonpad", "max_nonpad", "token_index"], default="last_nonpad")
    ap.add_argument("--token-index", type=int, default=0, help="Used when --pooling token_index")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", type=Path, default=Path("workdir/linear_uhead_from_cache"))
    ap.add_argument("--save-luh-head", action=argparse.BooleanOptionalAction, default=True, help="Save luh-compatible head files (weights.pth + config.yaml)")
    ap.add_argument("--skip-cache-signature-check", action="store_true", help="Allow training even if cache row signature does not match loaded dataset rows")
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_hf_rows(args.hf_dataset, args.hf_split)
    if not rows:
        raise ValueError("No rows loaded from --hf-dataset.")

    meta_path = args.token_cache_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {args.token_cache_dir}")
    with meta_path.open() as f:
        cache_meta_hint = json.load(f)

    if args.uq_mode == "fullseq":
        args.feature_source = "pooled"
        args.pooling = "mean_nonpad"
        print("uq-mode=fullseq -> feature_source=pooled, pooling=mean_nonpad")
    elif args.uq_mode == "pregen":
        if cache_meta_hint.get("cache_pregen_attn", False):
            args.feature_source = "pregen_hidden_attn"
            print("uq-mode=pregen -> feature_source=pregen_hidden_attn")
        elif cache_meta_hint.get("cache_pregen", False):
            args.feature_source = "pregen_hidden"
            print("uq-mode=pregen -> feature_source=pregen_hidden (attention cache unavailable)")
        else:
            raise ValueError(
                "uq-mode=pregen requires pregen cache. Rebuild cache with probing.py --cache-pregen "
                "(optionally --cache-pregen-attn)."
            )

    X, cache_meta = load_features_from_cache(args.token_cache_dir, args.feature_source, args.pooling, args.token_index)
    if X.shape[0] != len(rows):
        raise ValueError(f"Cache rows ({X.shape[0]}) != dataset rows ({len(rows)}). Use matching cache and dataset.")

    if args.feature_source != "pooled":
        if not cache_meta.get("cache_pregen", False) and args.feature_source in {"pregen_hidden", "pregen_hidden_attn"}:
            raise ValueError("Cache meta indicates pregen hidden features are missing. Rebuild cache with --cache-pregen.")
        if not cache_meta.get("cache_pregen_attn", False) and args.feature_source in {"pregen_attn", "pregen_hidden_attn"}:
            raise ValueError("Cache meta indicates pregen attention features are missing. Rebuild cache with --cache-pregen-attn.")

    text_field = str(cache_meta.get("text_field", "question"))
    expected_sig = cache_meta.get("row_signature_sha1")
    if expected_sig:
        cur_sig = _rows_signature(rows, text_field)
        if cur_sig != expected_sig:
            msg = (
                "Cache row signature mismatch. The token cache likely comes from a different dataset/order. "
                f"cache_sig={expected_sig[:10]}..., current_sig={cur_sig[:10]}..."
            )
            if args.skip_cache_signature_check:
                print(f"WARNING: {msg}")
            else:
                raise ValueError(msg)

    y = np.array([extract_label(r, args.label_field) for r in rows], dtype=np.int64)
    uniq = np.unique(y)
    if len(uniq) < 2:
        raise ValueError(f"Label '{args.label_field}' has a single class: {uniq}.")

    tr_idx, va_idx, te_idx = make_split_indices(rows, y, args.seed)

    x_tr, y_tr = X[tr_idx], y[tr_idx]
    x_va, y_va = X[va_idx], y[va_idx]
    x_te, y_te = X[te_idx], y[te_idx]

    mean = x_tr.mean(axis=0, keepdims=True)
    std = x_tr.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_tr = (x_tr - mean) / std
    x_va = (x_va - mean) / std
    x_te = (x_te - mean) / std

    device = torch.device(args.device if (args.device.startswith("cpu") or torch.cuda.is_available()) else "cpu")
    ue_head, ue_cfg = build_luh_linear_claim_head(x_tr.shape[1])
    ue_head = ue_head.to(device)
    model = ue_head.classifier
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    pos = int((y_tr == 1).sum())
    neg = int((y_tr == 0).sum())
    pos_weight = torch.tensor([max(neg / max(pos, 1), 1.0)], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    xtr_t = torch.from_numpy(x_tr).float().to(device)
    ytr_t = torch.from_numpy(y_tr).float().to(device)
    xva_t = torch.from_numpy(x_va).float().to(device)
    xte_t = torch.from_numpy(x_te).float().to(device)

    best_val_ap = -1.0
    best_state = None
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(xtr_t.shape[0], device=device)
        total_loss = 0.0
        total_n = 0
        for i in tqdm(range(0, xtr_t.shape[0], args.batch_size), desc=f"train-ep{ep}", leave=False):
            idx = perm[i:i + args.batch_size]
            xb = xtr_t[idx]
            yb = ytr_t[idx]
            logits = model(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * xb.shape[0]
            total_n += xb.shape[0]

        tr_logits = batched_logits(model, xtr_t, args.eval_batch_size, desc=f"eval-tr-ep{ep}")
        va_logits = batched_logits(model, xva_t, args.eval_batch_size, desc=f"eval-va-ep{ep}")
        tr_m = compute_metrics(y_tr, tr_logits)
        va_m = compute_metrics(y_va, va_logits)
        ep_stats = {
            "epoch": ep,
            "train_loss": total_loss / max(total_n, 1),
            "train": tr_m,
            "validation": va_m,
        }
        history.append(ep_stats)
        print(
            f"epoch={ep} loss={ep_stats['train_loss']:.4f} "
            f"val_ap={va_m['ap']:.4f} val_auc={va_m['roc_auc']:.4f} val_f1={va_m['f1']:.4f}"
        )

        if va_m["ap"] > best_val_ap:
            best_val_ap = va_m["ap"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    te_logits = batched_logits(model, xte_t, args.eval_batch_size, desc="eval-test")
    te_m = compute_metrics(y_te, te_logits)
    print(
        f"test: ap={te_m['ap']:.4f} auc={te_m['roc_auc']:.4f} "
        f"acc={te_m['acc']:.4f} f1={te_m['f1']:.4f} n={len(y_te)}"
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "state_dict": model.state_dict(),
        "input_dim": x_tr.shape[1],
        "pooling": args.pooling,
        "token_index": args.token_index,
        "label_field": args.label_field,
        "mean": torch.from_numpy(mean.astype(np.float32)),
        "std": torch.from_numpy(std.astype(np.float32)),
    }
    torch.save(ckpt, args.out_dir / "linear_uhead.pt")

    if args.save_luh_head:
        luh_out = args.out_dir / "luh_linear_claim_head"
        luh_out.mkdir(parents=True, exist_ok=True)
        ue_head.cfg = ue_cfg
        ue_head.save(luh_out)

    report = {
        "args": _json_safe(vars(args)),
        "num_rows": len(rows),
        "split_sizes": {"train": int(len(tr_idx)), "validation": int(len(va_idx)), "test": int(len(te_idx))},
        "label_field": args.label_field,
        "feature_source": args.feature_source,
        "label_counts": {"0": int((y == 0).sum()), "1": int((y == 1).sum())},
        "best_validation_ap": best_val_ap,
        "test_metrics": te_m,
        "history": history,
    }
    with (args.out_dir / "metrics.json").open("w") as f:
        json.dump(report, f, indent=2)

    with (args.out_dir / "test_predictions.jsonl").open("w") as f:
        probs = 1.0 / (1.0 + np.exp(-te_logits))
        for ds_i, prob, logit in zip(te_idx.tolist(), probs.tolist(), te_logits.tolist()):
            row = rows[ds_i]
            rec = {
                "index": int(ds_i),
                "dataset": row.get("dataset", ""),
                "subject": row.get("subject", row.get("entity_name", "")),
                "image_path": row.get("image_path", ""),
                "question": row.get("question", ""),
                "label": int(extract_label(row, args.label_field)),
                "prob": float(prob),
                "logit": float(logit),
            }
            f.write(json.dumps(rec) + "\n")

    if args.save_luh_head:
        print(f"Saved checkpoint + metrics to: {args.out_dir} (luh head: {args.out_dir / 'luh_linear_claim_head'})")
    else:
        print(f"Saved checkpoint + metrics to: {args.out_dir}")


if __name__ == "__main__":
    main()

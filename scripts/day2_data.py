"""Verified train/validation input access; no test path is accepted."""
import day2_runtime
import json
import random
import numpy as np
from tokenizers import Tokenizer
from common import ROOT, file_digest, read_jsonl
from modernbert_mlx import model_folder


def balanced_rows(split, count, seed=42):
    if split not in ("train", "validation"):
        raise ValueError("Day2 only uses train/validation")
    manifest = json.loads((ROOT / "reports/split_manifest.json").read_text())
    path = ROOT / f"data/processed/{split}.jsonl"
    if file_digest(path) != manifest["jsonl_sha256"][split]:
        raise ValueError(f"Modified split: {split}")
    rows = list(read_jsonl(path))
    rng = random.Random(seed)
    rng.shuffle(rows)
    selected = []
    for label in (0, 1):
        group = [r for r in rows if r["label"] == label]
        if count % 2 or count // 2 > len(group):
            raise ValueError("Count must be even and within the split")
        selected.extend(group[:count // 2])
    rng.shuffle(selected)
    return selected


def encode(rows, length):
    tok = Tokenizer.from_file(str(model_folder() / "tokenizer.json"))
    tok.enable_truncation(max_length=length)
    tok.enable_padding(length=length, pad_id=50283, pad_token="[PAD]")
    items = tok.encode_batch([r["text"] for r in rows])
    return {"ids": np.array([e.ids for e in items], dtype=np.int32),
            "mask": np.array([e.attention_mask for e in items], dtype=np.int32),
            "labels": np.array([r["label"] for r in rows], dtype=np.int32)}

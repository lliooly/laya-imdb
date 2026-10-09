"""Verified Rust tokenizer caches for frozen train/validation data only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time
import uuid
import numpy as np
from tokenizers import Tokenizer
from common import ROOT, file_digest, read_jsonl, write_json

BINARY = ROOT / "rust-tokenizer/target/release/imdb-tokenizer"


def tokenizer_path():
    lock = json.loads((ROOT / "configs/model_lock.json").read_text())
    path = ROOT / ".cache/huggingface/hub" / ("models--" + lock["model"].replace("/", "--")) / "snapshots" / lock["revision"] / "tokenizer.json"
    if file_digest(path) != lock["files"]["tokenizer.json"]:
        raise ValueError("Tokenizer hash mismatch")
    return path


def run_rust(input_path, output_path, length):
    started = time.perf_counter()
    process = subprocess.run([str(BINARY), "--tokenizer", str(tokenizer_path()), "--input", str(input_path),
        "--output", str(output_path), "--length", str(length), "--batch-size", "256"],
        capture_output=True, text=True, check=True, env={**__import__("os").environ, "RAYON_NUM_THREADS": "2"})
    return {"wall_seconds": time.perf_counter()-started, "rust": json.loads(process.stderr)}


def reference_tokenizer(length):
    tokenizer = Tokenizer.from_file(str(tokenizer_path()))
    tokenizer.enable_truncation(max_length=length)
    tokenizer.enable_padding(length=length, pad_id=50283, pad_token="[PAD]")
    return tokenizer


def compare_chunk(rows, encoded_rows, tokenizer):
    reference = tokenizer.encode_batch([r["text"] for r in rows])
    for row, encoded, expected in zip(rows, encoded_rows, reference, strict=True):
        fields = {"id": row["id"], "label": row["label"], "ids": expected.ids,
                  "mask": expected.attention_mask, "offsets": [list(pair) for pair in expected.offsets]}
        if encoded != fields:
            differing = [key for key in fields if encoded.get(key) != fields[key]]
            raise ValueError(f"Rust/Python mismatch for {row['id']}: {differing}")


def verify_stream(input_path, output_path, length):
    tokenizer = reference_tokenizer(length)
    rows, encoded_rows, ids, masks, labels = [], [], [], [], []
    started = time.perf_counter()
    for row, encoded in zip(read_jsonl(input_path), read_jsonl(output_path), strict=True):
        rows.append(row)
        encoded_rows.append(encoded)
        ids.append(encoded["ids"])
        masks.append(encoded["mask"])
        labels.append(encoded["label"])
        if len(rows) == 256:
            compare_chunk(rows, encoded_rows, tokenizer)
            rows.clear()
            encoded_rows.clear()
    if rows:
        compare_chunk(rows, encoded_rows, tokenizer)
    arrays = {"ids": np.array(ids, dtype=np.int32), "mask": np.array(masks, dtype=np.int32),
              "labels": np.array(labels, dtype=np.int32)}
    return arrays, {"passed": True, "rows": len(labels), "fields": ["id", "label", "ids", "mask", "offsets"],
                    "verification_wall_seconds": time.perf_counter()-started}


def build_cache(split, length):
    if split not in ("train", "validation") or length not in (256, 512):
        raise ValueError("Only train/validation at length 256/512 is enabled")
    source = ROOT / f"data/processed/{split}.jsonl"
    manifest = json.loads((ROOT / "reports/split_manifest.json").read_text())
    if file_digest(source) != manifest["jsonl_sha256"][split]:
        raise ValueError("Frozen split changed")
    identity = {"schema": 1, "split": split, "length": length, "data_sha256": file_digest(source),
        "tokenizer_sha256": file_digest(tokenizer_path()), "binary_sha256": file_digest(BINARY),
        "rust_version": subprocess.check_output([str(BINARY), "--version"], text=True).strip(),
        "python_tokenizers": importlib.metadata.version("tokenizers"),
        "source": {name: file_digest(ROOT / name) for name in (
            "scripts/cache_tokens.py", "rust-tokenizer/Cargo.lock", "rust-tokenizer/src/main.rs")},
        "offset_unit": "unicode_codepoints", "add_special_tokens": True, "pad_id": 50283,
        "truncation": "right", "padding": "fixed_right", "rayon_threads": 2}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    parent = ROOT / ".cache/tokenized"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / f"{split}-l{length}-{key}"
    if target.exists():
        metadata = json.loads((target / "metadata.json").read_text())
        if metadata["identity"] != identity:
            raise ValueError("Cache identity mismatch")
        for name, sha in metadata["files"].items():
            if file_digest(target / name) != sha:
                raise ValueError(f"Corrupted cache {name}")
        return target, metadata
    staging = parent / f".building-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        timing = run_rust(source, staging / "tokens.jsonl", length)
        arrays, verification = verify_stream(source, staging / "tokens.jsonl", length)
        np.savez_compressed(staging / "arrays.npz", **arrays)
        metadata = {"identity": identity, "timing": timing, "verification": verification,
            "files": {name: file_digest(staging / name) for name in ("tokens.jsonl", "arrays.npz")}}
        write_json(staging / "metadata.json", metadata)
        staging.replace(target)
        return target, metadata
    except BaseException:
        shutil.rmtree(staging)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--length", type=int, choices=(256, 512), default=256)
    args = parser.parse_args()
    path, metadata = build_cache(args.split, args.length)
    report = {"cache": str(path.relative_to(ROOT)), **metadata}
    write_json(ROOT / f"reports/rust_cache_{args.split}_l{args.length}.json", report)
    print(json.dumps({"cache": str(path), "verification": metadata["verification"], "timing": metadata["timing"]}, indent=2))


if __name__ == "__main__":
    main()

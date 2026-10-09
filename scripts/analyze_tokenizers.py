"""Analyze original Rust-backed tokenizers on train/validation only."""
import json
import os
import time

from common import ROOT, distribution, file_digest, read_jsonl, write_json

os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from huggingface_hub import HfApi, hf_hub_download
from tokenizers import Tokenizer


def main():
    config = json.loads((ROOT / "configs/day1.json").read_text())
    manifest = json.loads((ROOT / "reports/split_manifest.json").read_text())
    rows = {}
    for split in ("train", "validation"):
        path = ROOT / f"data/processed/{split}.jsonl"
        if file_digest(path) != manifest["jsonl_sha256"][split]:
            raise ValueError(f"Split changed: {split}")
        rows[split] = list(read_jsonl(path))
    lock_path = ROOT / "configs/tokenizer_lock.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    results = {}
    for model in config["tokenizer_models"]:
        if model not in lock:
            lock[model] = {"revision": HfApi().model_info(model).sha}
        path = hf_hub_download(model, "tokenizer.json", revision=lock[model]["revision"])
        sha = file_digest(path)
        if "tokenizer_sha256" in lock[model] and lock[model]["tokenizer_sha256"] != sha:
            raise ValueError(f"Tokenizer changed: {model}")
        lock[model]["tokenizer_sha256"] = sha
        write_json(lock_path, lock)
        tok = Tokenizer.from_file(path)
        tok.no_truncation()
        tok.no_padding()
        texts = [r["text"] for r in rows["train"][:256]]
        tok.encode_batch(texts)  # Warm-up, excluded from timing.
        model_result = {"revision": lock[model]["revision"], "tokenizer_sha256": sha,
                        "vocab_size": tok.get_vocab_size(), "splits": {}}
        for split, reviews in rows.items():
            lengths = []
            elapsed = 0.0
            for start in range(0, len(reviews), config["tokenizer_batch_size"]):
                texts = [r["text"] for r in reviews[start:start + config["tokenizer_batch_size"]]]
                begin = time.perf_counter()
                encoded = tok.encode_batch(texts, add_special_tokens=True)
                elapsed += time.perf_counter() - begin
                lengths.extend(len(e.ids) for e in encoded)
            model_result["splits"][split] = {"tokens_including_specials": distribution(lengths),
                "truncated_fraction": {str(n): sum(x > n for x in lengths) / len(lengths)
                                       for n in config["length_candidates"]},
                "tokenize_seconds": elapsed, "reviews_per_second": len(lengths) / elapsed}
        model_result["examples"] = [{"text": s, "tokens": tok.encode(s).tokens, "ids": tok.encode(s).ids}
            for s in ("This movie is not good.", "The beginning was slow, but the ending made it worthwhile.")]
        results[model] = model_result
        write_json(ROOT / "reports/tokenizer_audit.json", {"data_hashes": manifest["jsonl_sha256"],
            "timing_policy": "warm-up excluded; encode_batch only; no disk/network; one pass, exploratory",
            "models": results})
        print(f"Finished {model}", flush=True)


if __name__ == "__main__":
    main()

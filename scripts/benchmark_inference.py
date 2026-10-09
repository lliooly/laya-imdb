"""Warm resident baseline latency on validation reviews; no official test access."""
import day2_runtime
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import mlx.core as mx
import numpy as np
from common import ROOT, file_digest, write_json
from modernbert_mlx import Classifier
from day2_data import balanced_rows
from cache_tokens import reference_tokenizer
from benchmark_day2 import require_parity


def distribution(times):
    return {"count": len(times), "mean_ms": statistics.mean(times)*1000,
            "median_ms": statistics.median(times)*1000, "p95_ms": float(np.quantile(times, .95))*1000}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/modernbert-l256-s42")
    args = parser.parse_args()
    require_parity()
    if os.environ["MLX_ENABLE_TF32"] != "0":
        raise ValueError("Benchmark requires TF32 disabled")
    folder = args.checkpoint
    run = json.loads((folder / "run.json").read_text())
    if run["status"] != "complete":
        raise ValueError("Run after training completes; do not contend for the GPU")
    best = json.loads((folder / "best.json").read_text())
    weights = folder / best["weights"]
    if file_digest(weights) != best["sha256"]:
        raise ValueError("Best checkpoint hash mismatch")
    model = Classifier(json.loads((folder / "config.json").read_text()))
    model.load_weights(str(weights), strict=True)
    model.eval()
    mx.eval(model.parameters())
    rows = balanced_rows("validation", 256, seed=42)
    length = run["inference_length"]
    tokenizer = reference_tokenizer(length)

    def encode(texts):
        encoded = tokenizer.encode_batch(texts)
        return mx.array([e.ids for e in encoded], dtype=mx.int32), mx.array([e.attention_mask for e in encoded], dtype=mx.int32)

    def forward(ids, mask):
        result = mx.softmax(model(ids, mask), axis=-1)
        mx.eval(result)
        return result

    probes = []
    for batch_size in (1, 8):
        token_times, model_times, total_times = [], [], []
        measured_ids = []
        warmup, measured = 5, 32
        mx.reset_peak_memory()
        for i in range(warmup+measured):
            offset = i*batch_size % len(rows)
            texts = [r["text"] for r in rows[offset:offset+batch_size]]
            # Component probes are separately synchronized; the end-to-end probe runs afresh.
            start = time.perf_counter()
            ids, mask = encode(texts)
            mx.eval(ids, mask)
            token_time = time.perf_counter()-start
            start = time.perf_counter()
            reference = forward(ids, mask)
            model_time = time.perf_counter()-start
            start = time.perf_counter()
            actual = forward(*encode(texts))
            total_time = time.perf_counter()-start
            if float(mx.max(mx.abs(reference-actual)).item()) != 0:
                raise ValueError("Inference nondeterminism")
            if i >= warmup:
                token_times.append(token_time)
                model_times.append(model_time)
                total_times.append(total_time)
                measured_ids.append([r["id"] for r in rows[offset:offset+batch_size]])
        probes.append({"batch_size": batch_size, "warmup": warmup, "tokenizer_plus_input_arrays": distribution(token_times),
            "model_plus_softmax": distribution(model_times), "end_to_end": distribution(total_times),
            "reviews_per_second": batch_size/statistics.mean(total_times), "mlx_peak_gib": mx.get_peak_memory()/2**30,
            "measured_batch_ids": measured_ids})
    report = {"scope": "warm resident tokenizer/model; validation reviews; fixed padding; no official test",
        "excludes": "process startup, model/tokenizer loading, preprocessing, CLI output, file/network I/O",
        "length": length, "dtype": "float32", "MLX_ENABLE_TF32": "0", "best_epoch": best["epoch"],
        "weights_sha256": best["sha256"], "validation_ids": [r["id"] for r in rows], "probes": probes,
        "caveat": "32 timed batches per size, one process; components and end-to-end measured separately and need not add exactly; not a Rust-vs-Python speed comparison"}
    write_json(ROOT / "reports" / run["config"]["run_id"] / "inference_benchmark.json", report)
    print(json.dumps([{key: value for key, value in probe.items() if key != "measured_batch_ids"} for probe in probes], indent=2))


if __name__ == "__main__":
    main()

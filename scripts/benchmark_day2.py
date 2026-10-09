"""Fresh-process full-finetuning probes on real train reviews, fixed padded shapes."""
import day2_runtime
import os
import argparse
import json
import math
import statistics
import subprocess
import sys
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from common import ROOT, file_digest, write_json
from day2_data import balanced_rows, encode
from modernbert_mlx import load_pretrained


def require_parity():
    report = json.loads((ROOT / "reports/day2_parity.json").read_text())
    if report["status"] != "passed":
        raise RuntimeError("Numerical parity gate has not passed")
    if not report.get("verified_file_hashes"):
        raise RuntimeError("Rerun parity to record verified source hashes")
    for path, expected in report["verified_file_hashes"].items():
        if file_digest(ROOT / path) != expected:
            raise RuntimeError(f"Changed after parity check: {path}; rerun validation")


def loss_fn(m, ids, mask, labels):
    return nn.losses.cross_entropy(m(ids, mask), labels, reduction="mean")


def worker(args, config):
    require_parity()
    mx.random.seed(config["seed"])
    dtype = getattr(mx, args.dtype)
    model, _ = load_pretrained(dtype)
    model.train()
    optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    grad_fn = nn.value_and_grad(model, loss_fn)
    total = config["benchmark_warmup_steps"] + config["benchmark_measured_steps"]
    rows = balanced_rows("train", total * args.batch)
    arrays = encode(rows, args.length)
    batches = [tuple(mx.array(arrays[k][s:s + args.batch]) for k in ("ids", "mask", "labels"))
               for s in range(0, len(rows), args.batch)]
    mx.eval(batches)
    mx.reset_peak_memory()
    times, losses = [], []
    for i, batch in enumerate(batches):
        start = time.perf_counter()
        loss, grads = grad_fn(model, *batch)
        grads, grad_norm = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss, grad_norm)
        elapsed = time.perf_counter() - start
        if not bool(mx.isfinite(loss).item()) or not bool(mx.isfinite(grad_norm).item()):
            raise RuntimeError("Non-finite training loss/gradient norm")
        if i >= config["benchmark_warmup_steps"]:
            times.append(elapsed)
            losses.append(float(loss.item()))
    finite = all(bool(mx.all(mx.isfinite(v)).item()) for _, v in tree_flatten(grads))
    mean = statistics.mean(times)
    result = {"length": args.length, "batch_size": args.batch, "dtype": args.dtype, "MLX_ENABLE_TF32": os.environ["MLX_ENABLE_TF32"],
        "full_encoder_finetuning": True, "gradient_checkpointing": False,
        "warmup_steps": config["benchmark_warmup_steps"], "measured_steps": len(times),
        "step_seconds": times, "mean_step_seconds": mean, "median_step_seconds": statistics.median(times),
        "reviews_per_second": args.batch / mean, "mlx_peak_gib": mx.get_peak_memory() / 2**30,
        "losses": losses, "gradients_finite": finite,
        "padded_token_fraction": 1 - float(arrays["mask"].mean()),
        "train_22500_epoch_minutes": math.ceil(22500 / args.batch) * mean / 60,
        "train_3_epochs_minutes": math.ceil(22500 / args.batch) * mean * 3 / 60,
        "estimate_excludes": "validation, checkpoint IO, startup, sustained thermal effects; fixed padding upper workload"}
    write_json(ROOT / f"reports/day2_probes/{args.dtype}_b{args.batch}_l{args.length}.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--length", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"))
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/day2.json").read_text())
    if args.worker:
        worker(args, config)
        return
    require_parity()
    results = []
    for dtype in config["dtypes"]:
        for length in config["lengths"]:
            for batch in config["batch_sizes"]:
                command = [sys.executable, __file__, "--worker", "--length", str(length),
                           "--batch", str(batch), "--dtype", dtype]
                print(f"Probe {dtype} length={length} batch={batch}", flush=True)
                completed = subprocess.run(command, check=True, capture_output=True, text=True)
                result = json.loads(completed.stdout.strip().splitlines()[-1])
                results.append(result)
                write_json(ROOT / "reports/day2_benchmark.json", {
                    "device": "Apple M5 Pro / MLX Metal GPU", "config": config,
                    "protocol": "fresh process per shape; real train reviews; pretokenized fixed padding; synchronized forward+backward+clipping+AdamW; 2 warmup +5 timed steps",
                    "precision_caveat": "bfloat16 uses low-precision parameters and optimizer states; finite steps do not prove convergence or accuracy",
                    "probes": results})
                print(f"{result['mean_step_seconds']:.3f}s/step, {result['mlx_peak_gib']:.2f}GiB peak, {result['train_22500_epoch_minutes']:.1f}min/epoch estimate", flush=True)


if __name__ == "__main__":
    main()

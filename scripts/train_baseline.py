"""Full train/validation baseline with warmup, early stopping and exact resume."""
import day2_runtime
import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from common import ROOT, file_digest, read_jsonl, write_json
from day2_data import balanced_rows, encode
from modernbert_mlx import load_pretrained
from benchmark_day2 import require_parity, loss_fn
from checkpointing import atomic_json, restore_latest, save_latest
from training_utils import classification_metrics, epoch_order, is_better, learning_rate


def load_split(split):
    if split not in ("train", "validation"):
        raise ValueError("Training cannot access official test")
    manifest = json.loads((ROOT / "reports/split_manifest.json").read_text())
    path = ROOT / f"data/processed/{split}.jsonl"
    if file_digest(path) != manifest["jsonl_sha256"][split]:
        raise ValueError("Modified frozen data")
    return list(read_jsonl(path))


def evaluate(model, arrays, batch_size, output=None, rows=None):
    model.eval()
    predictions, probabilities = [], []
    loss_sum = 0.0
    for offset in range(0, len(arrays["labels"]), batch_size):
        ids, mask, labels = [mx.array(arrays[k][offset:offset+batch_size]) for k in ("ids", "mask", "labels")]
        logits = model(ids, mask)
        loss = nn.losses.cross_entropy(logits, labels, reduction="sum")
        mx.eval(logits, loss)
        loss_sum += float(loss.item())
        predictions.extend(mx.argmax(logits, axis=-1).tolist())
        probabilities.extend(mx.softmax(logits, axis=-1)[:, 1].tolist())
    if not math.isfinite(loss_sum):
        raise RuntimeError("Non-finite evaluation loss")
    if output:
        with Path(output).open("w", encoding="utf-8") as f:
            for r, label, prediction, positive in zip(rows, arrays["labels"], predictions, probabilities):
                f.write(json.dumps({"id": r["id"], "label": int(label), "prediction": prediction,
                                    "positive_probability_uncalibrated": positive}) + "\n")
    return classification_metrics(arrays["labels"], predictions, loss_sum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/day3.json")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if os.environ["MLX_ENABLE_TF32"] != "0" or config["dtype"] != "float32":
        raise ValueError("Baseline requires precise float32")
    require_parity()
    folder = ROOT / "checkpoints" / config["run_id"]
    if folder.exists() and not args.resume:
        raise FileExistsError(f"Existing run: {folder}; use --resume or a new run_id")
    if args.resume and not folder.exists():
        raise FileNotFoundError(folder)
    folder.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / "reports" / config["run_id"]
    report_dir.mkdir(parents=True, exist_ok=True)
    identity = {"config": config, "split_manifest": file_digest(ROOT / "reports/split_manifest.json"),
        "model_lock": file_digest(ROOT / "configs/model_lock.json"),
        "source": {name: file_digest(ROOT / "scripts" / name) for name in (
            "train_baseline.py", "modernbert_mlx.py", "checkpointing.py", "training_utils.py", "day2_data.py", "day2_runtime.py")},
        "versions": {name: importlib.metadata.version(name) for name in ("mlx", "tokenizers", "numpy")}}
    train_rows, val_rows = load_split("train"), load_split("validation")
    if len(train_rows) != 22500 or len(val_rows) != 2500:
        raise ValueError("Unexpected full baseline sample counts")
    print("Encoding frozen train/validation splits...", flush=True)
    train, validation = encode(train_rows, config["length"]), encode(val_rows, config["length"])
    monitor_rows = balanced_rows("train", config["train_monitor_count"], config["seed"])
    monitor = encode(monitor_rows, config["length"])
    write_json(report_dir / "train_monitor_ids.json", [r["id"] for r in monitor_rows])
    mx.random.seed(config["seed"])
    model, model_config = load_pretrained(mx.float32)
    optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"], bias_correction=config["adam_bias_correction"])
    grad_fn = nn.value_and_grad(model, loss_fn)
    batch_size = config["batch_size"]
    batches_per_epoch = math.ceil(len(train_rows)/batch_size)
    total_steps = config["epochs"] * batches_per_epoch
    state = {"identity": identity, "global_step": 0, "epoch": 0, "next_batch": 0,
             "history": [], "best": None, "best_epoch": None, "bad_epochs": 0,
             "online_loss_sum": 0.0, "online_count": 0, "elapsed_training_seconds": 0.0}
    if args.resume:
        state = restore_latest(folder, model, optimizer, identity)
    write_json(folder / "config.json", model_config)
    write_json(folder / "run.json", {"config": config, "identity": identity, "inference_length": config["length"],
        "status": "running", "scope": "full train/validation baseline; no official test"})

    def status(phase):
        atomic_json(report_dir / "status.json", {"phase": phase, "global_step": state["global_step"],
            "total_steps": total_steps, "epoch": state["epoch"], "next_batch": state["next_batch"],
            "history": state["history"], "best_epoch": state["best_epoch"], "best_validation": state["best"],
            "elapsed_training_seconds": state["elapsed_training_seconds"], "mlx_peak_gib": mx.get_peak_memory()/2**30})

    try:
        for epoch in range(state["epoch"], config["epochs"]):
            if state["bad_epochs"] >= config["early_stopping_patience"]:
                break
            order = epoch_order(len(train_rows), config["seed"], epoch)
            model.train()
            start_batch = state["next_batch"] if state["epoch"] == epoch else 0
            for batch_index in range(start_batch, batches_per_epoch):
                chosen = order[batch_index*batch_size:(batch_index+1)*batch_size]
                inputs = [mx.array(train[k][chosen]) for k in ("ids", "mask", "labels")]
                lr = learning_rate(state["global_step"], total_steps, config["learning_rate"], config["warmup_fraction"])
                optimizer.learning_rate = lr
                began = time.perf_counter()
                loss, grads = grad_fn(model, *inputs)
                grads, norm = optim.clip_grad_norm(grads, config["gradient_clip_norm"])
                # Check before updating, so a failed step never poisons a checkpoint.
                mx.eval(loss, norm)
                value, norm_value = float(loss.item()), float(norm.item())
                if not math.isfinite(value) or not math.isfinite(norm_value):
                    raise RuntimeError("Non-finite train loss/gradient")
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)
                state["elapsed_training_seconds"] += time.perf_counter()-began
                state["global_step"] += 1
                state["next_batch"] = batch_index+1
                state["online_loss_sum"] += value * len(chosen)
                state["online_count"] += len(chosen)
                if state["global_step"] % config["log_every_steps"] == 0:
                    status("training")
                    print(f"epoch={epoch+1} step={state['global_step']}/{total_steps} loss={value:.4f} lr={lr:.3g} peak={mx.get_peak_memory()/2**30:.2f}GiB", flush=True)
                if state["global_step"] % config["checkpoint_every_steps"] == 0:
                    save_latest(folder, model, optimizer, state)
            status("validating")
            print(f"Evaluating epoch {epoch+1} on all 2500 validation reviews...", flush=True)
            val_metrics = evaluate(model, validation, batch_size, report_dir/f"validation-epoch{epoch+1}.jsonl", val_rows)
            monitor_metrics = evaluate(model, monitor, batch_size)
            record = {"epoch": epoch+1, "global_step": state["global_step"], "validation": val_metrics,
                "train_monitor_fixed_subset": monitor_metrics,
                "online_train_loss": state["online_loss_sum"]/state["online_count"],
                "elapsed_training_seconds": state["elapsed_training_seconds"]}
            state["history"].append(record)
            if is_better(val_metrics, state["best"]):
                state["best"], state["best_epoch"], state["bad_epochs"] = val_metrics, epoch+1, 0
                best = folder / f"best-epoch{epoch+1}.safetensors"
                model.save_weights(str(best))
                atomic_json(folder / "best.json", {"weights": best.name, "epoch": epoch+1,
                    "metrics": val_metrics, "sha256": file_digest(best)})
            else:
                state["bad_epochs"] += 1
            state.update(epoch=epoch+1, next_batch=0, online_loss_sum=0.0, online_count=0)
            save_latest(folder, model, optimizer, state)
            status("epoch_complete")
            print(json.dumps(record), flush=True)
        # Verify selected checkpoint reload and publish complete validation metrics.
        best = json.loads((folder / "best.json").read_text())
        model.load_weights(str(folder / best["weights"]), strict=True)
        selected_metrics = evaluate(model, validation, batch_size, report_dir/"validation-best.jsonl", val_rows)
        if abs(selected_metrics["macro_f1"]-best["metrics"]["macro_f1"]) > 1e-12:
            raise RuntimeError("Selected checkpoint reload metric mismatch")
        summary = {"status": "complete", "config": config, "identity": identity, "history": state["history"],
            "best_epoch": state["best_epoch"], "best_validation": selected_metrics,
            "epochs_completed": len(state["history"]), "early_stopped": state["bad_epochs"] >= config["early_stopping_patience"],
            "elapsed_training_seconds": state["elapsed_training_seconds"], "mlx_peak_gib": mx.get_peak_memory()/2**30,
            "scope": "validation metrics only; official test untouched", "checkpoint_reload_verified": True}
        write_json(report_dir/"result.json", summary)
        write_json(folder/"run.json", {**summary, "inference_length": config["length"]})
        status("complete")
        print("Baseline complete: " + json.dumps(selected_metrics), flush=True)
    except BaseException:
        status("interrupted_or_failed")
        # Last committed generation remains available; do not save a partial step.
        raise


if __name__ == "__main__":
    main()

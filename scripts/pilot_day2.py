"""Short real IMDB full-encoder finetuning; validation is diagnostic only."""
import day2_runtime
import os
import json
import time
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from common import ROOT, file_digest, write_json
from day2_data import balanced_rows, encode
from modernbert_mlx import Classifier, load_pretrained
from benchmark_day2 import loss_fn, require_parity


def evaluate(model, arrays, batch_size):
    model.eval()
    predictions, loss_sum = [], 0.0
    for start in range(0, len(arrays["labels"]), batch_size):
        ids, mask, labels = [mx.array(arrays[k][start:start + batch_size]) for k in ("ids", "mask", "labels")]
        logits = model(ids, mask)
        loss = nn.losses.cross_entropy(logits, labels, reduction="sum")
        mx.eval(logits, loss)
        predictions.extend(np.array(mx.argmax(logits, axis=-1)).tolist())
        loss_sum += float(loss.item())
    y, p = arrays["labels"], np.array(predictions)
    cm = [[int(((y == i) & (p == j)).sum()) for j in (0, 1)] for i in (0, 1)]
    f1 = [2 * cm[i][i] / max(1, 2 * cm[i][i] + cm[1-i][i] + cm[i][1-i]) for i in (0, 1)]
    return {"count": len(y), "loss": loss_sum / len(y), "accuracy": float((y == p).mean()),
            "macro_f1": sum(f1) / 2, "confusion_matrix_rows_true": cm}


def main():
    require_parity()
    config = json.loads((ROOT / "configs/day2.json").read_text())
    mx.random.seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    model, model_config = load_pretrained(mx.float32)
    batch_size, length = config["pilot_batch_size"], config["pilot_length"]
    train_rows = balanced_rows("train", config["pilot_train_count"])
    val_rows = balanced_rows("validation", config["pilot_validation_count"])
    train, val = encode(train_rows, length), encode(val_rows, length)
    before = {"train": evaluate(model, train, batch_size), "validation": evaluate(model, val, batch_size)}
    # A constant-LR short probe, not the final warmup/early-stopping recipe.
    optimizer = optim.AdamW(learning_rate=config["learning_rate"], weight_decay=config["weight_decay"])
    grad_fn = nn.value_and_grad(model, loss_fn)
    selected = rng.permutation(len(train_rows))
    losses, steps = [], []
    first_norm = None
    # Track pretrained encoder updates, not just the new classification head.
    norm_before = np.array(model.model.layers[-1].mlp.Wo.weight)
    model.train()
    mx.reset_peak_memory()
    for step in range(config["pilot_steps"]):
        offset = (step * batch_size) % len(selected)
        chosen = selected[offset:offset + batch_size]
        ids, mask, labels = [mx.array(train[k][chosen]) for k in ("ids", "mask", "labels")]
        started = time.perf_counter()
        loss, grads = grad_fn(model, ids, mask, labels)
        grads, grad_norm = optim.clip_grad_norm(grads, 1.0)
        if first_norm is None:
            first_norm = float(grad_norm.item())
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss, grad_norm)
        steps.append(time.perf_counter() - started)
        losses.append(float(loss.item()))
        if not np.isfinite(losses[-1]) or not bool(mx.isfinite(grad_norm).item()):
            raise RuntimeError("Non-finite pilot loss/gradient")
        if (step + 1) % 8 == 0:
            print(f"Pilot step {step + 1}/{config['pilot_steps']} loss={losses[-1]:.4f}", flush=True)
    finite = all(bool(mx.all(mx.isfinite(g)).item()) for _, g in tree_flatten(grads))
    updated = float(np.max(np.abs(np.array(model.model.layers[-1].mlp.Wo.weight) - norm_before)))
    peak_gib = mx.get_peak_memory() / 2**30
    after = {"train": evaluate(model, train, batch_size), "validation": evaluate(model, val, batch_size)}
    checkpoint = ROOT / "checkpoints/day2-pilot"
    checkpoint.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(checkpoint / "model.safetensors"))
    write_json(checkpoint / "config.json", model_config)
    check_arrays = {k: v[:batch_size] for k, v in val.items()}
    ids, mask = mx.array(check_arrays["ids"]), mx.array(check_arrays["mask"])
    expected = model(ids, mask)
    restored = Classifier(model_config)
    restored.load_weights(str(checkpoint / "model.safetensors"), strict=True)
    restored.eval()
    error = float(mx.max(mx.abs(restored(ids, mask) - expected)).item())
    if not finite or updated == 0 or error > 1e-6:
        raise RuntimeError("Pilot gradients, encoder updates or checkpoint reload failed")
    report = {"scope": "256 train /128 validation short diagnostic; no official test; not final accuracy",
        "config": config, "dtype": "float32", "MLX_ENABLE_TF32": os.environ["MLX_ENABLE_TF32"], "full_encoder_finetuning": True,
        "model_lock": json.loads((ROOT / "configs/model_lock.json").read_text()),
        "split_manifest_sha256": file_digest(ROOT / "reports/split_manifest.json"),
        "sample_ids": {"train": [r["id"] for r in train_rows], "validation": [r["id"] for r in val_rows]},
        "before": before, "after": after, "step_losses": losses, "step_seconds": steps,
        "first_gradient_norm": first_norm, "gradients_finite": finite,
        "encoder_last_mlp_max_update": updated, "mlx_training_peak_gib": peak_gib,
        "save_reload_max_abs_error": error, "checkpoint_sha256": file_digest(checkpoint / "model.safetensors"),
        "status": "passed"}
    write_json(checkpoint / "run.json", report)
    write_json(ROOT / "reports/day2_pilot.json", report)
    print(json.dumps({"before": before, "after": after, "reload_error": error, "status": "passed"}), flush=True)


if __name__ == "__main__":
    main()

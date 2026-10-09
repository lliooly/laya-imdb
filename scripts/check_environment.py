"""MLX GPU arithmetic, autodiff, optimizer and save/reload smoke test.

Synthetic linear classification only; this is NOT an IMDB model or timing estimate.
"""
import importlib.metadata
import json
import platform
import subprocess
import time

from common import ROOT, write_json


def main():
    report = {"python": platform.python_version(), "os": platform.platform(),
        "architecture": platform.machine(), "packages": {}, "scope": "synthetic MLX environment smoke only"}
    for package in ("mlx", "mlx-metal", "tokenizers", "huggingface-hub"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    # Select only non-identifying hardware fields.
    info = json.loads(subprocess.check_output(["system_profiler", "SPHardwareDataType", "-json"]))
    hardware = info["SPHardwareDataType"][0]
    report["hardware"] = {k: hardware.get(k) for k in ("chip_type", "physical_memory", "machine_model")}
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    if not mx.metal.is_available():
        raise RuntimeError("Metal unavailable; Day1 GPU smoke test did not pass")
    mx.set_default_device(mx.gpu)
    mx.random.seed(42)
    x = mx.random.normal((32, 8))
    y = (x[:, 0] + x[:, 1] > 0).astype(mx.int32)
    model = nn.Linear(8, 2)
    optimizer = optim.AdamW(learning_rate=0.03)

    def loss_fn(m, inputs, labels):
        return nn.losses.cross_entropy(m(inputs), labels, reduction="mean")

    initial = float(loss_fn(model, x, y).item())
    grad_fn = nn.value_and_grad(model, loss_fn)
    started = time.perf_counter()
    losses = []
    for _ in range(20):
        loss, grads = grad_fn(model, x, y)
        if not all(bool(mx.all(mx.isfinite(g)).item()) for _, g in tree_flatten(grads)):
            raise RuntimeError("Non-finite gradients")
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state, loss)
        losses.append(float(loss.item()))
    final = float(loss_fn(model, x, y).item())
    if not final < initial:
        raise RuntimeError("Synthetic loss did not decrease")
    target = ROOT / "artifacts/environment_smoke.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(target))
    restored = nn.Linear(8, 2)
    restored.load_weights(str(target))
    error = float(mx.max(mx.abs(model(x) - restored(x))).item())
    if error > 1e-6:
        raise RuntimeError("Save/reload output mismatch")
    report.update({"metal_available": True, "device": str(mx.default_device()),
        "initial_loss": initial, "final_loss": final, "losses": losses,
        "save_reload_max_abs_error": error, "synthetic_20_steps_seconds": time.perf_counter() - started,
        "status": "passed"})
    write_json(ROOT / "reports/environment.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

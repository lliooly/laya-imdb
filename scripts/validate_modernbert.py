"""Compare MLX with a pinned Transformers CPU float32 reference.

Valid tokens and masked pooled logits are compared; padded query outputs have no
semantic meaning. Tests cross the local attention window and vary right padding.
"""
import day2_runtime
import os
import importlib.metadata
import json
import numpy as np
import torch
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from transformers import AutoModel, ModernBertConfig
from common import ROOT, file_digest, write_json
from day2_data import balanced_rows, encode
from modernbert_mlx import Classifier, load_pretrained, model_folder


def compare(actual, expected, atol, rtol, relative_l2_limit=None):
    difference = np.abs(actual - expected)
    passed = bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    relative_l2 = float(np.linalg.norm(actual.astype(np.float64)-expected.astype(np.float64)) / max(1e-12, np.linalg.norm(expected.astype(np.float64))))
    return {"passed": passed if relative_l2_limit is None else relative_l2 <= relative_l2_limit,
            "elementwise_allclose": passed, "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()), "atol": atol, "rtol": rtol,
            "relative_l2": relative_l2, "relative_l2_limit": relative_l2_limit}


def tiny_architecture_check(config):
    """Strict structural/gradient check without pretrained activation outliers."""
    torch.manual_seed(42)
    c = dict(config)
    c.update(hidden_size=32, num_attention_heads=4, intermediate_size=48,
             vocab_size=64, num_hidden_layers=3, local_attention=8,
             global_attn_every_n_layers=2, pad_token_id=0, max_position_embeddings=64)
    tc = ModernBertConfig(**c, reference_compile=False)
    tc._attn_implementation = "eager"
    ref = AutoModel.from_config(tc).eval()
    m = Classifier(c)
    m.model.load_weights([(k, mx.array(v.detach().numpy())) for k, v in ref.state_dict().items()], strict=True)
    head = torch.nn.Linear(32, 2)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(np.array(m.classifier.weight)))
        head.bias.copy_(torch.from_numpy(np.array(m.classifier.bias)))
    ids_np = np.arange(32, dtype=np.int32).reshape(2, 16) + 1
    mask_np = np.ones((2, 16), dtype=np.int32)
    mask_np[1, 6:] = 0
    ids_np[1, 6:] = 0
    ids, mask, labels = mx.array(ids_np), mx.array(mask_np), mx.array([0, 1])
    ti, tm = torch.from_numpy(ids_np).long(), torch.from_numpy(mask_np).long()
    th = ref(ti, attention_mask=tm).last_hidden_state
    tl = head((th*tm[:, :, None]).sum(1)/tm.sum(1)[:, None])
    loss = torch.nn.functional.cross_entropy(tl, torch.tensor([0, 1]))
    loss.backward()
    def loss_fn(model, i, a, y):
        return nn.losses.cross_entropy(model(i, a), y, reduction="mean")
    mloss, grad = nn.value_and_grad(m, loss_fn)(m, ids, mask, labels)
    mh, logits = m.model(ids, mask), m(ids, mask)
    mx.eval(mloss, grad, mh, logits)
    valid = mask_np.astype(bool)
    checks = {"hidden": compare(np.array(mh)[valid], th.detach().numpy()[valid], 3e-5, 3e-5),
        "logits": compare(np.array(logits), tl.detach().numpy(), 3e-5, 3e-5),
        "loss": compare(np.array(mloss), loss.detach().numpy(), 3e-5, 3e-5)}
    params = dict(ref.named_parameters())
    for name, g in tree_flatten(grad):
        expected = (dict(head.named_parameters())[name.removeprefix("classifier.")].grad
                    if name.startswith("classifier.") else params[name.removeprefix("model.")].grad)
        checks["gradient:" + name] = compare(np.array(g), expected.numpy(), 3e-5, 3e-4)
    return {"passed": all(r["passed"] for r in checks.values()), "checks": checks,
            "scope": "3-layer random weights; both local/global attention; padding; every parameter gradient"}


def main():
    cfg = json.loads((ROOT / "configs/day2.json").read_text())
    if os.environ["MLX_ENABLE_TF32"] != "0":
        raise RuntimeError("Parity requires MLX_ENABLE_TF32=0")
    mx.random.seed(cfg["seed"])
    torch.set_num_threads(6)
    mlx_model, model_config = load_pretrained()
    mlx_model.eval()
    reference = AutoModel.from_pretrained(str(model_folder()), attn_implementation="eager",
        reference_compile=False, local_files_only=True).float().eval()
    head = torch.nn.Linear(reference.config.hidden_size, 2)
    with torch.no_grad():
        head.weight.copy_(torch.from_numpy(np.array(mlx_model.classifier.weight)))
        head.bias.copy_(torch.from_numpy(np.array(mlx_model.classifier.bias)))
    rows = balanced_rows("train", 4)
    results = {"reference": "Transformers CPU eager float32; reference_compile=False", "MLX_ENABLE_TF32": os.environ["MLX_ENABLE_TF32"],
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "transformers", "mlx", "numpy")},
        "model_lock": json.loads((ROOT / "configs/model_lock.json").read_text()),
        "cases": {}, "scope": "encoder + our masked-mean linear head; not HF sequence-classification head",
        "criteria_note": "Strict elementwise random-model check; pretrained hidden/gradients use relative L2 because large activation outliers amplify fp32 reduction-order rounding. Elementwise errors remain disclosed.",
        "tiny_architecture": tiny_architecture_check(model_config)}
    for length in dict.fromkeys((16, 160, *cfg["lengths"])):
        arrays = encode(rows[:2], length)
        ids, mask = mx.array(arrays["ids"]), mx.array(arrays["mask"])
        ti, tm = torch.from_numpy(arrays["ids"]).long(), torch.from_numpy(arrays["mask"]).long()
        with torch.no_grad():
            th = reference(input_ids=ti, attention_mask=tm).last_hidden_state
            pooled = (th * tm[:, :, None]).sum(1) / tm.sum(1)[:, None]
            tl = head(pooled)
        mh, ml = mlx_model.model(ids, mask), mlx_model(ids, mask)
        mx.eval(mh, ml)
        valid = arrays["mask"].astype(bool)
        results["cases"][str(length)] = {
            "valid_tokens": int(valid.sum()),
            "hidden": compare(np.array(mh)[valid], th.numpy()[valid], cfg["parity_atol"], cfg["parity_rtol"], cfg["hidden_relative_l2_limit"]),
            "logits": compare(np.array(ml), tl.numpy(), cfg["parity_atol"], cfg["parity_rtol"])}
        print(f"Forward length={length}: {results['cases'][str(length)]}", flush=True)
    # Padding invariance and mask isolation on the same real comments.
    short = encode(rows[:2], 32)
    padded_ids = np.pad(short["ids"], ((0, 0), (0, 128)), constant_values=50283)
    padded_mask = np.pad(short["mask"], ((0, 0), (0, 128)))
    original = mlx_model(mx.array(short["ids"]), mx.array(short["mask"]))
    extended = mlx_model(mx.array(padded_ids), mx.array(padded_mask))
    padded_ids[:, 32:] = 123  # masked IDs must not influence predictions.
    perturbed = mlx_model(mx.array(padded_ids), mx.array(padded_mask))
    results["padding_invariance"] = compare(np.array(extended), np.array(original), 3e-5, 3e-5)
    results["masked_id_invariance"] = compare(np.array(perturbed), np.array(extended), 3e-5, 3e-5)

    # Full pretrained backward, compare selected gradients rather than only loss.
    arrays = encode(rows[:2], cfg["gradient_check_length"])
    ids, mask, labels = [mx.array(arrays[k]) for k in ("ids", "mask", "labels")]
    ti, tm, ty = [torch.from_numpy(arrays[k]).long() for k in ("ids", "mask", "labels")]
    reference.zero_grad()
    head.zero_grad()
    th = reference(input_ids=ti, attention_mask=tm).last_hidden_state
    tl = head((th * tm[:, :, None]).sum(1) / tm.sum(1)[:, None])
    torch_loss = torch.nn.functional.cross_entropy(tl, ty)
    torch_loss.backward()

    def loss_fn(m, i, a, y):
        return nn.losses.cross_entropy(m(i, a), y, reduction="mean")

    mlx_loss, grads = nn.value_and_grad(mlx_model, loss_fn)(mlx_model, ids, mask, labels)
    mx.eval(mlx_loss, grads)
    flat = dict(tree_flatten(grads))
    results["loss"] = compare(np.array(mlx_loss), torch_loss.detach().numpy(), 3e-5, 3e-5)
    results["gradients"] = {}
    results["gradient_check_length"] = cfg["gradient_check_length"]
    reference_grads = dict(reference.named_parameters())
    for name in ("classifier.weight", "model.final_norm.weight", "model.layers.0.attn.Wqkv.weight",
                 "model.layers.21.mlp.Wo.weight", "model.embeddings.tok_embeddings.weight"):
        expected = head.weight.grad if name == "classifier.weight" else reference_grads[name.removeprefix("model.")].grad
        results["gradients"][name] = compare(np.array(flat[name]), expected.numpy(), cfg["gradient_atol"], cfg["gradient_rtol"], cfg["gradient_relative_l2_limit"])
    results["all_gradients_finite"] = all(bool(mx.all(mx.isfinite(v)).item()) for v in flat.values())
    results["status"] = "passed" if (all(v["hidden"]["passed"] and v["logits"]["passed"] for v in results["cases"].values())
        and all(v["passed"] for v in results["gradients"].values()) and results["loss"]["passed"]
        and results["padding_invariance"]["passed"] and results["masked_id_invariance"]["passed"]
        and results["all_gradients_finite"] and results["tiny_architecture"]["passed"]) else "failed"
    results["verified_file_hashes"] = {path: file_digest(ROOT / path) for path in (
        "scripts/modernbert_mlx.py", "scripts/day2_data.py", "scripts/day2_runtime.py",
        "scripts/validate_modernbert.py", "configs/model_lock.json", "configs/day2.json")}
    write_json(ROOT / "reports/day2_parity.json", results)
    print(f"Parity: {results['status']}", flush=True)
    if results["status"] != "passed":
        raise AssertionError("MLX/reference mismatch; do not train until resolved")


if __name__ == "__main__":
    main()

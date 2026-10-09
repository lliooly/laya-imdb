"""Export saved-checkpoint logits on frozen validation data; never loads the test split."""
import argparse
import gc
import json
import time
from pathlib import Path

import train_laya
train_laya.block_network()
import day2_runtime
import numpy as np
from common import ROOT, file_digest, read_jsonl, write_json

RUNS = {"laya_ce_full": "laya-l512-s42-ce", "laya_rlcd_pilot": "laya-l512-s42-pilot-rlcd",
        "laya_ce_pilot": "laya-l512-s42-pilot-ce"}


def export_baseline(rows, out):
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from day2_data import encode
    from modernbert_mlx import Classifier
    from benchmark_day2 import require_parity
    require_parity()
    folder = ROOT / "checkpoints/modernbert-l256-s42"
    best = json.loads((folder / "best.json").read_text())
    assert file_digest(folder / best["weights"]) == best["sha256"]
    target = out / "modernbert_base"
    target.mkdir(parents=True, exist_ok=True)
    if (target / "export.json").exists():
        meta = json.loads((target / "export.json").read_text())
        if meta["checkpoint_sha256"] == best["sha256"] and meta["predictions_sha256"] == file_digest(target / "raw.jsonl"):
            print("Reuse verified baseline export", flush=True)
            return
    model = Classifier(json.loads((folder / "config.json").read_text()))
    model.load_weights(str(folder / best["weights"]), strict=True)
    model.eval()
    mx.eval(model.parameters())
    named = tree_flatten(model.parameters())
    meta = {"checkpoint_sha256": best["sha256"], "total_parameters": sum(v.size for _, v in named),
            "trainable_parameters": sum(v.size for _, v in tree_flatten(model.trainable_parameters())),
            "encoder_parameters": sum(v.size for k, v in named if k.startswith("model.")),
            "head_parameters": sum(v.size for k, v in named if not k.startswith("model.")),
            "inference_dtype": "float32", "batch_size": 8}
    old = {r["id"]: r for r in read_jsonl(ROOT / "reports/modernbert-l256-s42/validation-best.jsonl")}
    data = encode(rows, 256)
    with (target / "raw.jsonl").open("w") as f:
        for start in range(0, len(rows), 8):
            z = model(mx.array(data["ids"][start:start+8]), mx.array(data["mask"][start:start+8]))
            mx.eval(z)
            for row, logits in zip(rows[start:start+8], np.asarray(z).tolist()):
                assert int(np.argmax(logits)) == old[row["id"]]["prediction"]
                f.write(json.dumps({"sample_id": row["id"], "true_label": row["label"], "logits": logits}) + "\n")
            if start % 400 == 0:
                print("ModernBERT export %d/2500" % min(start+8, len(rows)), flush=True)
    meta["predictions_sha256"] = file_digest(target / "raw.jsonl")
    write_json(target / "export.json", meta)
    del model, data, z
    gc.collect()
    mx.clear_cache()


def export_laya(key, run, rows, out, batch_size):
    import torch
    import laya.train as t
    folder = ROOT / "checkpoints" / run
    target = out / key
    target.mkdir(parents=True, exist_ok=True)
    sha = file_digest(folder / "model.safetensors")
    if (target / "export.json").exists():
        meta = json.loads((target / "export.json").read_text())
        if meta["checkpoint_sha256"] == sha and meta["predictions_sha256"] == file_digest(target / "raw.jsonl"):
            print("Reuse verified export: " + key, flush=True)
            return
    model, tok, cfg = t.load_checkpoint(str(folder))
    named = list(model.named_parameters())
    meta = {"run_id": run, "checkpoint_sha256": sha,
            "total_parameters": sum(p.numel() for _, p in named),
            "trainable_parameters": sum(p.numel() for _, p in named if p.requires_grad),
            "encoder_parameters": sum(p.numel() for n, p in named if n.startswith("encoder.")),
            "head_parameters": sum(p.numel() for n, p in named if not n.startswith("encoder.")),
            "action_head_parameters_not_used_by_classification_loss": sum(p.numel() for n, p in named if n.startswith("act_head.")),
            "saved_weights_dtype": "fp16; upstream loads as fp32", "inference_dtype": "float32",
            "batch_size": batch_size, "head_layers": cfg["head_layers"],
            "hidden_size": model.encoder.config.hidden_size,
            "head_attention_heads": max(1, model.encoder.config.hidden_size // 64),
            "native_temperature": cfg["temperature"][0]}
    typed = [{"state": r["text"], "questions": {"sentiment": train_laya.QUESTION},
              "expected": {"sentiment": ("negative", "positive")[r["label"]]}} for r in rows]
    items, skipped = t.items_from_rows(tok, typed, cfg["max_len"], cfg["head_max_len"])
    assert not skipped and len(items) == len(rows)
    head_ids, _, _ = t.build_head(tok, items[0]["q"], cfg["head_max_len"])
    meta.update(head_token_length=len(head_ids), review_token_budget=cfg["max_len"] - len(head_ids) - 1)
    model.to("mps")
    started = time.monotonic()
    correct = 0
    with (target / "raw.jsonl").open("w") as f:
        for start in range(0, len(items), 200):
            records = t.calibration_records(model, tok, items[start:start+200], torch.device("mps"),
                                           cfg["max_len"], cfg["head_max_len"], batch_size=batch_size)
            for row, (_, logits, _, _) in zip(rows[start:start+200], records):
                assert np.isfinite(logits).all()
                correct += int(np.argmax(logits) == row["label"])
                f.write(json.dumps({"sample_id": row["id"], "true_label": row["label"], "logits": logits.tolist()}) + "\n")
            print("%s export %d/2500, %.1fs" % (key, min(start+200, len(items)), time.monotonic()-started), flush=True)
    meta.update(validation_accuracy=correct / len(rows), elapsed_export_seconds=time.monotonic()-started)
    if key == "laya_ce_full" and correct != 2394:
        raise RuntimeError("Batch inference changed CE accuracy: rerun with batch_size=1 and remove this export metadata")
    meta["predictions_sha256"] = file_digest(target / "raw.jsonl")
    write_json(target / "export.json", meta)
    del model, records, items
    gc.collect()
    torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="reports/model-comparison")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "reports/split_manifest.json").read_text())
    path = ROOT / "data/processed/validation.jsonl"
    assert file_digest(path) == split["jsonl_sha256"]["validation"]
    rows = list(read_jsonl(path))
    assert len(rows) == 2500
    write_json(out / "evaluation_source.json", {"dataset": "IMDB", "split": "development validation",
               "sha256": file_digest(path), "ids": [r["id"] for r in rows],
               "test_accessed": False, "export_script_sha256": file_digest(Path(__file__))})
    export_baseline(rows, out)
    for key, run in RUNS.items():
        export_laya(key, run, rows, out, args.batch_size)


if __name__ == "__main__":
    main()

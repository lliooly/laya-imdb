"""Validation-only length strata and errors from the selected baseline."""
import json
import math
from tokenizers import Tokenizer
from common import ROOT, file_digest, read_jsonl, write_json
from cache_tokens import tokenizer_path
from training_utils import classification_metrics


def main():
    directory = ROOT / "reports/modernbert-l256-s42"
    result = json.loads((directory / "result.json").read_text())
    if result["status"] != "complete":
        raise ValueError("Baseline must be complete")
    path = ROOT / "data/processed/validation.jsonl"
    manifest = json.loads((ROOT / "reports/split_manifest.json").read_text())
    if file_digest(path) != manifest["jsonl_sha256"]["validation"]:
        raise ValueError("Validation split changed")
    rows = list(read_jsonl(path))
    predictions = list(read_jsonl(directory / "validation-best.jsonl"))
    tokenizer = Tokenizer.from_file(str(tokenizer_path()))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    items = []
    for offset in range(0, len(rows), 128):
        batch = rows[offset:offset+128]
        encodings = tokenizer.encode_batch([r["text"] for r in batch])
        for row, prediction, encoded in zip(batch, predictions[offset:offset+128], encodings, strict=True):
            if (row["id"], row["label"]) != (prediction["id"], prediction["label"]):
                raise ValueError("Prediction identity mismatch")
            positive = prediction["positive_probability_uncalibrated"]
            probability_true = positive if row["label"] else 1-positive
            items.append({**prediction, "original_token_count_including_special": len(encoded.ids),
                "truncated": len(encoded.ids) > result["config"]["length"],
                "loss": -math.log(max(1e-12, probability_true)), "excerpt": row["text"][:400]})
    if len(items) != 2500 or len(predictions) != len(items):
        raise ValueError("Expected full 2500 validation reviews")
    groups = {}
    for name, chosen in (("all", items), ("fits_256", [r for r in items if not r["truncated"]]),
                         ("truncated_256", [r for r in items if r["truncated"]])):
        groups[name] = classification_metrics([r["label"] for r in chosen], [r["prediction"] for r in chosen], sum(r["loss"] for r in chosen))
    errors = [r for r in items if r["prediction"] != r["label"]]
    errors.sort(key=lambda r: abs(r["positive_probability_uncalibrated"]-0.5), reverse=True)
    report = {"scope": "selected baseline on validation only; descriptive strata, not a causal length experiment",
        "loss_note": "stratum losses reconstructed from probabilities clipped at 1e-12; original logits-based full-validation loss is in result.json",
        "best_epoch": result["best_epoch"], "groups": groups,
        "truncated_fraction": groups["truncated_256"]["count"]/len(items),
        "highest_uncalibrated_confidence_errors": errors[:20]}
    write_json(directory / "error_analysis.json", report)
    print(json.dumps({"groups": groups, "truncated_fraction": report["truncated_fraction"]}, indent=2))


if __name__ == "__main__":
    main()

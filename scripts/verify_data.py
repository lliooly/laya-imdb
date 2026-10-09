"""Verify frozen artifacts before training without changing any files."""
import json
from collections import Counter

from common import ROOT, clean_text, digest, file_digest, read_jsonl
from prepare_data import partition


def main():
    manifest = json.loads((ROOT / "reports/split_manifest.json").read_text())
    config = json.loads((ROOT / "configs/day1.json").read_text())
    rows = {}
    for split, expected in (("train", 11250), ("validation", 1250), ("test", 12500)):
        path = ROOT / f"data/processed/{split}.jsonl"
        assert file_digest(path) == manifest["jsonl_sha256"][split], f"Modified file: {split}"
        rows[split] = list(read_jsonl(path))
        assert [r["id"] for r in rows[split]] == manifest["ids"][split]
        assert len({r["id"] for r in rows[split]}) == len(rows[split])
        assert Counter(r["label"] for r in rows[split]) == {0: expected, 1: expected}
        for r in rows[split]:
            assert r["text"] and clean_text(r["text"]) == r["text"]
            assert digest(r["text"]) == r["text_sha256"]
    train_hashes = {r["text_sha256"] for r in rows["train"]}
    assert not train_hashes & {r["text_sha256"] for r in rows["validation"]}
    official_train = sorted(rows["train"] + rows["validation"], key=lambda r: r["id"])
    reconstructed = partition(official_train, config["seed"], config["validation_per_class"])
    assert reconstructed == (rows["train"], rows["validation"])
    print("Verified: file hashes, source IDs, labels, text hashes, no train/validation exact overlap, deterministic split.")


if __name__ == "__main__":
    main()

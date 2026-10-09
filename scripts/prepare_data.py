"""Download the official archive, audit it and freeze duplicate-aware splits.

Only labeled review members are read. No tar extraction and no unsupervised data use.
Official test text is read for integrity/duplicate auditing, never used to choose splits.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import shutil
import tarfile
import urllib.request

from common import ROOT, clean_text, digest, distribution, file_digest, write_json

URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"


def download(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".part")
    with urllib.request.urlopen(URL, timeout=120) as src, partial.open("wb") as dst:
        shutil.copyfileobj(src, dst)
    # Validate before promoting a completed archive.
    with tarfile.open(partial, "r:gz") as archive:
        archive.getmembers()
    partial.replace(destination)


def load_archive(path):
    records = {"train": [], "test": []}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if (len(parts) != 4 or parts[0] != "aclImdb" or parts[1] not in records
                    or parts[2] not in ("pos", "neg") or not member.isfile()
                    or not parts[3].endswith(".txt")):
                continue
            raw = archive.extractfile(member).read().decode("utf-8", errors="strict")
            text = clean_text(raw)
            if not text:
                raise ValueError(f"Empty review: {member.name}")
            score = int(Path(parts[3]).stem.split("_")[-1])
            label = int(parts[2] == "pos")
            if not ((label == 0 and 1 <= score <= 4) or (label == 1 and 7 <= score <= 10)):
                raise ValueError(f"Invalid label/score: {member.name}")
            records[parts[1]].append({"id": "/".join(parts[1:]), "text": text,
                "label": label, "raw_sha256": digest(raw), "text_sha256": digest(text)})
    for split, rows in records.items():
        rows.sort(key=lambda r: r["id"])
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate source IDs in {split}")
        if Counter(r["label"] for r in rows) != {0: 12500, 1: 12500}:
            raise ValueError(f"Unexpected official {split} class counts")
    return records


def partition(rows, seed, per_class):
    """Keep identical cleaned texts together, including conflicting-label groups.

    Mixed-label duplicate groups stay in training. Single-label groups are shuffled
    deterministically and selected whole, up to the requested per-class target.
    """
    groups = defaultdict(list)
    for row in rows:
        groups[row["text_sha256"]].append(row)
    selected = set()
    rng = random.Random(seed)
    for label in (0, 1):
        candidates = sorted(h for h, group in groups.items() if {r["label"] for r in group} == {label})
        rng.shuffle(candidates)
        remaining = per_class
        for h in candidates:
            if len(groups[h]) <= remaining:
                selected.add(h)
                remaining -= len(groups[h])
            if remaining == 0:
                break
        if remaining:
            raise ValueError(f"Cannot meet validation target for label {label}")
    train = [r for r in rows if r["text_sha256"] not in selected]
    val = [r for r in rows if r["text_sha256"] in selected]
    assert not {r["text_sha256"] for r in train} & {r["text_sha256"] for r in val}
    return train, val


def audit(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["text_sha256"]].append(r)
    return {"count": len(rows), "labels": dict(Counter(r["label"] for r in rows)),
            "characters": distribution([len(r["text"]) for r in rows]),
            "whitespace_words": distribution([len(r["text"].split()) for r in rows]),
            "duplicate_extra_rows": len(rows) - len(groups),
            "conflicting_label_groups": sum(len({r["label"] for r in g}) > 1 for g in groups.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/raw/aclImdb_v1.tar.gz")
    args = parser.parse_args()
    config = json.loads((ROOT / "configs/day1.json").read_text())
    if not args.archive.exists():
        print("Downloading official IMDB archive...", flush=True)
        download(args.archive)
    records = load_archive(args.archive)
    train, val = partition(records["train"], config["seed"], config["validation_per_class"])
    splits = {"train": train, "validation": val, "test": records["test"]}
    output = ROOT / "data/processed"
    output.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name, rows in splits.items():
        path = output / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        hashes[name] = file_digest(path)
    train_hashes = {r["text_sha256"] for r in records["train"]}
    test_hashes = {r["text_sha256"] for r in records["test"]}
    report = {"source": URL, "archive_sha256": file_digest(args.archive), "seed": config["seed"],
        "preprocessing": config["preprocessing"], "split_policy": "stratified cleaned-text groups; mixed-label groups stay in train",
        "official_train_test_overlap_unique_texts": len(train_hashes & test_hashes),
        "train_validation_overlap_unique_texts": 0,
        "official_train": audit(records["train"]),
        "splits": {name: audit(rows) for name, rows in splits.items()}, "jsonl_sha256": hashes,
        "test_policy": "integrity audit only; no tokenizer fitting, selection or model evaluation"}
    write_json(ROOT / "reports/data_audit.json", report)
    # Lightweight immutable-by-hash manifest for later experiments.
    write_json(ROOT / "reports/split_manifest.json", {
        "seed": config["seed"], "archive_sha256": report["archive_sha256"],
        "jsonl_sha256": hashes, "ids": {name: [r["id"] for r in rows] for name, rows in splits.items()}})
    print(json.dumps({"counts": {n: len(r) for n, r in splits.items()},
                      "official_overlap": report["official_train_test_overlap_unique_texts"]}, indent=2))


if __name__ == "__main__":
    main()

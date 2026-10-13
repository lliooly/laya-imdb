"""Shared paths and deterministic preprocessing; no model dependencies."""
import hashlib
import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clean_text(text):
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    return " ".join(html.unescape(text).split())


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def distribution(values):
    values = sorted(values)
    if not values:
        return {}
    return {"count": len(values), "min": values[0], "mean": sum(values) / len(values),
            **{f"p{p}": values[round((len(values) - 1) * p / 100)] for p in (50, 90, 95, 99)},
            "max": values[-1]}

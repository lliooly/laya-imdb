"""Exercise character offsets, padding, truncation and batch remainders."""
import json
import subprocess
import tempfile
from pathlib import Path
from common import ROOT, file_digest, write_json
from cache_tokens import BINARY, run_rust, tokenizer_path, verify_stream


def main():
    examples = ["", "This isn't bad. I loved it!", "电影不错 🎬👍 café naïve résumé", "e\u0301 ≠ é; 👩‍💻", "[CLS] [SEP] [PAD] [MASK]", "bad good " * 900]
    examples += [f"Review {i}: outstanding acting; a boring ending." for i in range(257)]
    results = []
    rejected_cases = []
    with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as temporary:
        folder = Path(temporary)
        source = folder / "reviews.jsonl"
        source.write_text("".join(json.dumps({"id": str(i), "label": i % 2, "text": text}, ensure_ascii=False)+"\n" for i, text in enumerate(examples)), encoding="utf-8")
        for length in (256, 512):
            timing = run_rust(source, folder / f"tokens{length}.jsonl", length)
            arrays, verification = verify_stream(source, folder / f"tokens{length}.jsonl", length)
            assert arrays["ids"].shape == (len(examples), length)
            results.append({"length": length, **verification, "timing": timing})
        original_hash = file_digest(source)
        for name, extra in (
            ("overwrite_input", ["--output", str(source)]),
            ("overwrite_existing_output", ["--output", str(folder / "tokens256.jsonl")]),
            ("too_short", ["--output", str(folder / "short.jsonl"), "--length", "2"]),
            ("duplicate_option", ["--output", str(folder / "duplicate.jsonl"), "--length", "256", "--length", "512"]),
        ):
            args = [str(BINARY), "--tokenizer", str(tokenizer_path()), "--input", str(source)]
            if name not in ("too_short", "duplicate_option"):
                args += ["--length", "256"]
            process = subprocess.run(args+extra, capture_output=True, text=True)
            if process.returncode == 0 or file_digest(source) != original_hash:
                raise AssertionError(f"Safety rejection failed: {name}")
            rejected_cases.append(name)
    report = {"passed": True, "cases": ["empty", "negation", "Unicode accents", "combining characters", "emoji/ZWJ", "literal special tokens", "truncation", "batch remainder"], "results": results, "rejected_cases": rejected_cases,
        "scope": "Rust CLI vs Python bindings of the same pinned Hugging Face tokenizer; integration parity, not independent algorithm validation"}
    write_json(ROOT / "reports/day3_rust_tokenizer_check.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

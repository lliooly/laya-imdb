"""Download a pinned English Laya checkpoint and record local file hashes."""
import json
from pathlib import Path

from common import ROOT, file_digest, write_json

MODEL_ID = "convaiinnovations/laya"
REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
MODEL_DIR = ROOT / ".cache/laya/english"
FILES = ["model.safetensors", "rl_agent_config.json", "encoder/config.json",
         "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"]


def main():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_ID, revision=REVISION, local_dir=MODEL_DIR,
                      allow_patterns=FILES + ["README.md"])
    lock = {"model_id": MODEL_ID, "revision": REVISION, "laya_version": "0.4.1",
            "local_dir": str(MODEL_DIR.relative_to(ROOT)),
            "sha256": {name: file_digest(MODEL_DIR / name) for name in FILES}}
    write_json(ROOT / "configs/laya_model_lock.json", lock)
    print(json.dumps(lock, indent=2))


if __name__ == "__main__":
    main()

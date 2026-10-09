"""Download original pretrained ModernBERT at the Day1 tokenizer revision."""
import json
import os
from common import ROOT, file_digest, write_json

os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
from huggingface_hub import snapshot_download


def main():
    model = "answerdotai/ModernBERT-base"
    revision = json.loads((ROOT / "configs/tokenizer_lock.json").read_text())[model]["revision"]
    folder = snapshot_download(model, revision=revision,
        allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json",
                        "special_tokens_map.json", "model.safetensors", "README.md", "LICENSE*"])
    from pathlib import Path
    folder = Path(folder)
    required = ("config.json", "tokenizer.json", "model.safetensors")
    for name in required:
        if not (folder / name).exists():
            raise FileNotFoundError(name)
    write_json(ROOT / "configs/model_lock.json", {"model": model, "revision": revision,
        "files": {name: file_digest(folder / name) for name in required}})
    print(f"Downloaded and hashed {model}@{revision}")


if __name__ == "__main__":
    main()

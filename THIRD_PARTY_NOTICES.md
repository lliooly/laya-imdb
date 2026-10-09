# Third-party notices

The MIT license in `LICENSE` applies to this project's original contributions.
Third-party material retains its original license and attribution.

## ModernBERT reference implementation

`scripts/modernbert_mlx.py` identifies its architecture and weight-naming reference
as Hugging Face Transformers 4.57.6 ModernBERT, copyright 2024 Answer.AI, LightOn,
Hugging Face and contributors, under Apache-2.0:
https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/modernbert/modeling_modernbert.py

Preserve the source-file attribution and the accompanying
[Apache-2.0 text](docs/THIRD_PARTY_APACHE_LICENSE.txt). Any portions derived from
that implementation remain subject to those applicable terms.

## Models, libraries and dataset

- Laya: https://github.com/NandhaKishorM/laya and https://huggingface.co/convaiinnovations/laya
- ModernBERT weights: https://huggingface.co/answerdotai/ModernBERT-base
- MLX: https://github.com/ml-explore/mlx
- Laya MLX dependency: pinned as `laya-mlx==0.3.0` in `pyproject.toml` and `uv.lock`.
- Hugging Face Tokenizers: https://github.com/huggingface/tokenizers
- IMDB dataset: https://ai.stanford.edu/~amaas/data/sentiment/

Dependencies are installed separately. Model weights, tokenizer caches and raw
IMDB reviews are excluded from Git. Consult the corresponding upstream packages,
model cards and dataset terms when downloading or redistributing these assets.
The repository's MIT license does not relicense them.

Historical reports may contain predictions, identifiers and limited examples for
analysis; these outputs do not grant a license to the underlying review texts.

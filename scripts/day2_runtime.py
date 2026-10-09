"""Set precision policy before any MLX import (including transitive imports)."""
import os

# M5 reduced-precision float32 matmul is enabled by default in MLX.
# Exact reference checks and our initial baseline opt out explicitly.
os.environ.setdefault("MLX_ENABLE_TF32", "0")

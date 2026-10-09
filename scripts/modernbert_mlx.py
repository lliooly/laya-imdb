"""Padded ModernBERT encoder in MLX with an independent mean-pool classifier.

Architecture/weight naming reference: Transformers 4.57.6 ModernBERT, Apache-2.0,
Copyright 2024 Answer.AI, LightOn, Hugging Face and contributors.
https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/modernbert/modeling_modernbert.py
This implementation supports the pinned base config, not every ModernBERT variant.
"""
import day2_runtime
import json
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from common import ROOT, file_digest


def model_folder():
    lock = json.loads((ROOT / "configs/model_lock.json").read_text())
    folder = ROOT / ".cache/huggingface/hub" / ("models--" + lock["model"].replace("/", "--")) / "snapshots" / lock["revision"]
    return folder


def norm(c):
    return nn.LayerNorm(c["hidden_size"], eps=c["norm_eps"], bias=c["norm_bias"])


class Embeddings(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.tok_embeddings = nn.Embedding(c["vocab_size"], c["hidden_size"])
        self.norm = norm(c)

    def __call__(self, ids):
        return self.norm(self.tok_embeddings(ids))


def rotary(x, theta):
    dim, length = x.shape[-1], x.shape[-2]
    inv = theta ** (-mx.arange(0, dim, 2, dtype=mx.float32) / dim)
    freqs = mx.arange(length, dtype=mx.float32)[:, None] * inv[None, :]
    angle = mx.concatenate([freqs, freqs], axis=-1)
    cos, sin = mx.cos(angle).astype(x.dtype), mx.sin(angle).astype(x.dtype)
    rotated = mx.concatenate([-x[..., dim // 2:], x[..., :dim // 2]], axis=-1)
    return x * cos + rotated * sin


class Attention(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.heads = c["num_attention_heads"]
        self.dim = c["hidden_size"] // self.heads
        self.local = index % c["global_attn_every_n_layers"] != 0
        self.theta = c["local_rope_theta"] if self.local else c["global_rope_theta"]
        self.Wqkv = nn.Linear(c["hidden_size"], 3 * c["hidden_size"], bias=c["attention_bias"])
        self.Wo = nn.Linear(c["hidden_size"], c["hidden_size"], bias=c["attention_bias"])

    def __call__(self, x, global_mask, local_mask):
        b, length, hidden = x.shape
        qkv = self.Wqkv(x).reshape(b, length, 3, self.heads, self.dim)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        q, k = rotary(q, self.theta), rotary(k, self.theta)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.dim ** -0.5,
            mask=local_mask if self.local else global_mask)
        return self.Wo(out.transpose(0, 2, 1, 3).reshape(b, length, hidden))


class MLP(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.Wi = nn.Linear(c["hidden_size"], 2 * c["intermediate_size"], bias=c["mlp_bias"])
        self.Wo = nn.Linear(c["intermediate_size"], c["hidden_size"], bias=c["mlp_bias"])

    def __call__(self, x):
        value, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(nn.gelu(value) * gate)


class Layer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.attn_norm = nn.Identity() if index == 0 else norm(c)
        self.attn = Attention(c, index)
        self.mlp_norm = norm(c)
        self.mlp = MLP(c)

    def __call__(self, x, global_mask, local_mask):
        x = x + self.attn(self.attn_norm(x), global_mask, local_mask)
        return x + self.mlp(self.mlp_norm(x))


class Encoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        for key in ("embedding_dropout", "attention_dropout", "mlp_dropout"):
            if c[key] != 0:
                raise ValueError(f"Unsupported nonzero {key}")
        if c["hidden_activation"] != "gelu" or c.get("rope_scaling"):
            raise ValueError("Only default RoPE and exact GELU supported")
        self.embeddings = Embeddings(c)
        self.layers = [Layer(c, i) for i in range(c["num_hidden_layers"])]
        self.final_norm = norm(c)
        self.window = c["local_attention"] // 2
        self.max_length = c["max_position_embeddings"]
        self.checkpoint_layers = False

    def __call__(self, ids, mask):
        length = ids.shape[1]
        if length > self.max_length or ids.shape != mask.shape:
            raise ValueError("Invalid sequence length or attention mask shape")
        global_mask = mask[:, None, None, :].astype(mx.bool_)
        positions = mx.arange(length)
        local_mask = global_mask & (mx.abs(positions[:, None] - positions[None, :]) <= self.window)[None, None]
        x = self.embeddings(ids)
        for layer in self.layers:
            x = nn.checkpoint(layer)(x, global_mask, local_mask) if self.checkpoint_layers and self.training else layer(x, global_mask, local_mask)
        return self.final_norm(x)


class Classifier(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.model = Encoder(c)
        self.classifier = nn.Linear(c["hidden_size"], 2)
        self.classifier.weight = mx.random.normal((2, c["hidden_size"])) * 0.02
        self.classifier.bias = mx.zeros((2,))

    def __call__(self, ids, mask):
        hidden = self.model(ids, mask)
        weights = mask[:, :, None].astype(hidden.dtype)
        pooled = (hidden * weights).sum(axis=1) / weights.sum(axis=1)
        # Keep loss/logits in float32 for low-precision probes.
        return self.classifier(pooled).astype(mx.float32)


def load_pretrained(dtype=mx.float32):
    folder = model_folder()
    c = json.loads((folder / "config.json").read_text())
    lock = json.loads((ROOT / "configs/model_lock.json").read_text())
    for name, expected in lock["files"].items():
        if file_digest(folder / name) != expected:
            raise ValueError(f"Modified pretrained asset: {name}")
    model = Classifier(c)
    weights = mx.load(str(folder / "model.safetensors"))
    encoder_weights = {k.removeprefix("model."): v for k, v in weights.items() if k.startswith("model.")}
    # Strict check ensures no pretrained encoder layer silently remains random.
    model.model.load_weights(list(encoder_weights.items()), strict=True)
    model.set_dtype(dtype)
    mx.eval(model.parameters())
    return model, c

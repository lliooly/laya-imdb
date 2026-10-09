"""Layerwise diagnostic retained for investigating numerical drift."""
import day2_runtime
import numpy as np
import torch
import mlx.core as mx
from transformers import AutoModel
from day2_data import balanced_rows, encode
from modernbert_mlx import load_pretrained, model_folder
from common import ROOT, write_json


def main():
    torch.set_num_threads(6)
    m, _ = load_pretrained()
    m.eval()
    r = AutoModel.from_pretrained(str(model_folder()), attn_implementation="eager", reference_compile=False).eval()
    a = encode(balanced_rows("train", 4)[:2], 16)
    ids, mask = mx.array(a["ids"]), mx.array(a["mask"])
    with torch.no_grad():
        states = r(torch.from_numpy(a["ids"]).long(), attention_mask=torch.from_numpy(a["mask"]), output_hidden_states=True).hidden_states
        f32 = r(torch.from_numpy(a["ids"]).long(), attention_mask=torch.from_numpy(a["mask"])).last_hidden_state.numpy()
        r.double()
        f64 = r(torch.from_numpy(a["ids"]).long(), attention_mask=torch.from_numpy(a["mask"])).last_hidden_state.float().numpy()
        r.float()
    mfinal = np.array(m.model(ids, mask))
    arbitration = {name: {"max": float(np.abs(arr-f64).max()), "mean": float(np.abs(arr-f64).mean()),
        "relative_l2": float(np.linalg.norm(arr-f64)/np.linalg.norm(f64))} for name, arr in (("torch32",f32),("mlx32",mfinal))}
    write_json(ROOT / "reports/day2_float64_diagnostic.json", {"case": "same real train comments as parity, length16", "results": arbitration})
    print("float64 arbitration", arbitration, flush=True)
    x = m.model.embeddings(ids)
    def show(name, actual, expected):
        diff = np.abs(np.array(actual) - expected.detach().numpy())
        print(name, float(diff.max()), float(diff.mean()), flush=True)
    tx = torch.from_numpy(np.array(x))
    show("qkv", m.model.layers[0].attn.Wqkv(x), r.layers[0].attn.Wqkv(tx))
    from modernbert_mlx import rotary
    from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb
    ma = m.model.layers[0].attn
    qkv = ma.Wqkv(x).reshape(2, 16, 3, 12, 64)
    q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
    tq, tk, tv = [torch.from_numpy(np.array(t)) for t in (q, k, v)]
    cos, sin = r.layers[0].attn.rotary_emb(tq, torch.arange(16)[None])
    trq, trk = apply_rotary_pos_emb(tq, tk, cos, sin)
    show("rope", rotary(q, ma.theta), trq)
    rq, rk = rotary(q, ma.theta), rotary(k, ma.theta)
    tout = torch.nn.functional.scaled_dot_product_attention(torch.from_numpy(np.array(rq)), torch.from_numpy(np.array(rk)), tv)
    mout = mx.fast.scaled_dot_product_attention(rq, rk, v, scale=64**-0.5)
    show("attention", mout, tout)
    z = m.model.layers[0].mlp.Wi(x)
    val, gate = mx.split(z, 2, -1)
    import mlx.nn as nn
    show("gelu", nn.gelu(val), torch.nn.functional.gelu(torch.from_numpy(np.array(val))))
    g = mask[:, None, None, :].astype(mx.bool_)
    p = mx.arange(ids.shape[1])
    local = g & (mx.abs(p[:, None] - p[None, :]) <= m.model.window)[None, None]
    for i, expected in enumerate(states):
        if i:
            x = m.model.layers[i - 1](x, g, local)
        diff = np.abs(np.array(x) - expected.numpy())
        print(i, float(diff.max()), float(diff.mean()), float(np.abs(expected.numpy()).max()), flush=True)


if __name__ == "__main__":
    main()

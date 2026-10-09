"""Atomic resumable generations; pointer switches only after all files exist."""
import day2_runtime
import json
from pathlib import Path
import shutil
import uuid
import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten
from common import file_digest, write_json


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    write_json(temp, value)
    temp.replace(path)


def save_latest(folder, model, optimizer, state):
    folder = Path(folder)
    name = f"state-{state['global_step']:06d}-{uuid.uuid4().hex[:8]}"
    target = folder / name
    target.mkdir(parents=True)
    model.save_weights(str(target / "model.safetensors"))
    mx.save_safetensors(str(target / "optimizer.safetensors"), dict(tree_flatten(optimizer.state)))
    mx.save_safetensors(str(target / "rng.safetensors"), {str(i): v for i, v in enumerate(mx.random.state)})
    metadata = dict(state)
    metadata["files"] = {n: file_digest(target/n) for n in ("model.safetensors", "optimizer.safetensors", "rng.safetensors")}
    write_json(target / "state.json", metadata)
    pointer = folder / "latest.json"
    old = json.loads(pointer.read_text())["generation"] if pointer.exists() else None
    atomic_json(pointer, {"generation": name})
    if old and old != name:
        shutil.rmtree(folder / old)
    return target


def restore_latest(folder, model, optimizer, expected_identity):
    folder = Path(folder)
    target = folder / json.loads((folder / "latest.json").read_text())["generation"]
    state = json.loads((target / "state.json").read_text())
    if state["identity"] != expected_identity:
        raise ValueError("Resume config/source/data identity mismatch")
    for name, sha in state["files"].items():
        if file_digest(target / name) != sha:
            raise ValueError(f"Checkpoint hash mismatch: {name}")
    model.load_weights(str(target / "model.safetensors"), strict=True)
    optimizer.state = tree_unflatten(list(mx.load(str(target / "optimizer.safetensors")).items()))
    rng = mx.load(str(target / "rng.safetensors"))
    for i in range(len(rng)):
        mx.random.state[i][:] = rng[str(i)]
    mx.eval(model.parameters(), optimizer.state, mx.random.state)
    return state

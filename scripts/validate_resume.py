"""Verify restored optimizer + RNG produce the same next update as uninterrupted training."""
import day2_runtime
import tempfile
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from checkpointing import restore_latest, save_latest
from common import ROOT, write_json


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(8, 8), nn.Linear(8, 2)]
        self.drop = nn.Dropout(0.2)

    def __call__(self, x):
        return self.layers[1](self.drop(nn.gelu(self.layers[0](x))))


def main():
    mx.random.seed(42)
    model = Tiny()
    optimizer = optim.AdamW(learning_rate=0.001)
    x, y = mx.random.normal((4,8)), mx.array([0,1,0,1])
    mx.eval(x, y)
    def step(m, opt):
        def loss_fn(mm):
            return nn.losses.cross_entropy(mm(x), y, reduction="mean")
        loss, grads = nn.value_and_grad(m, loss_fn)(m)
        opt.update(m, grads)
        mx.eval(loss, m.parameters(), opt.state)
        return float(loss.item())
    step(model, optimizer)
    step(model, optimizer)
    with tempfile.TemporaryDirectory(dir=ROOT/"artifacts") as temporary:
        identity = {"test": "resume-with-dropout-and-list-layers"}
        save_latest(temporary, model, optimizer, {"identity": identity, "global_step": 2})
        expected_loss = step(model, optimizer)
        expected = dict(tree_flatten(model.parameters()))
        expected_state = dict(tree_flatten(optimizer.state))
        restored, opt = Tiny(), optim.AdamW(learning_rate=0.001)
        state = restore_latest(temporary, restored, opt, identity)
        actual_loss = step(restored, opt)
        parameter_error = max(float(mx.max(mx.abs(v-expected[k])).item()) for k,v in tree_flatten(restored.parameters()))
        optimizer_error = max(float(mx.max(mx.abs(v-expected_state[k])).item()) for k,v in tree_flatten(opt.state))
        try:
            restore_latest(temporary, restored, opt, {"test": "changed"})
            rejected = False
        except ValueError:
            rejected = True
    passed = abs(actual_loss-expected_loss)<1e-7 and parameter_error<1e-7 and optimizer_error<1e-7 and rejected
    report = {"passed": passed, "restored_global_step": state["global_step"],
        "loss_abs_error": abs(actual_loss-expected_loss), "parameter_max_abs_error": parameter_error,
        "optimizer_max_abs_error": optimizer_error, "identity_mismatch_rejected": rejected,
        "scope": "synthetic dropout model; saved optimizer state, list parameter trees and RNG continuation"}
    write_json(ROOT/"reports/day3_resume_check.json", report)
    print(report)
    if not passed:
        raise AssertionError("Resume diverged")


if __name__ == "__main__":
    main()

"""Offline IMDB adapter for the pinned upstream Laya trainer (PyTorch/MPS/CUDA)."""
import os

# Set before importing Hugging Face or Laya. Training only uses local files.
for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
             "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK"):
    os.environ[name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
from functools import partial
from importlib.metadata import version
import json
from pathlib import Path
import random
import socket
import time

from common import ROOT, file_digest, read_jsonl, write_json
from laya_optimization import optimize_training, validate_optimization
from laya_epoch_checkpoints import keep_epoch_checkpoints, validate_epoch_checkpoints
from laya_resampling import resample_training, validate_sampling
from laya_runtime import prepare_runtime, apply_runtime

QUESTION = {"type": "choice",
            "instructions": "What is the reviewer's overall opinion of this movie?",
            "criteria": {"negative": "The reviewer dislikes or criticizes the movie overall.",
                         "positive": "The reviewer likes or praises the movie overall."}}


def balanced_subset(rows, size, seed):
    if size is None:
        return rows
    if size < 2 or size % 2 or size > len(rows):
        raise ValueError("Subset size must be even, >=2, and within the source split")
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        group = [r for r in rows if r["label"] == label]
        rng.shuffle(group)
        selected.extend(group[:size // 2])
    rng.shuffle(selected)
    return selected


def prepare_rows(split, size, seed, directory, manifest):
    path = ROOT / "data/processed" / (split + ".jsonl")
    if file_digest(path) != manifest["jsonl_sha256"][split]:
        raise ValueError("Frozen dataset hash mismatch: " + split)
    rows = balanced_subset(list(read_jsonl(path)), size, seed)
    output = directory / (split + ".jsonl")
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            typed = {"state": row["text"], "questions": {"sentiment": QUESTION},
                     "expected": {"sentiment": ("negative", "positive")[row["label"]]}}
            f.write(json.dumps(typed, ensure_ascii=False) + "\n")
    return output, {"source_sha256": file_digest(path), "typed_sha256": file_digest(output),
                    "count": len(rows), "ids": [r["id"] for r in rows]}


def block_network():
    """Fail on Python TCP/UDP network attempts; also use library offline flags above."""
    original = socket.socket.connect

    def connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Offline training blocked a network connection: " + str(address))
        return original(sock, address)

    socket.socket.connect = connect
    original_sendto = socket.socket.sendto

    def sendto(sock, *args):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Offline training blocked a UDP network request")
        return original_sendto(sock, *args)

    socket.socket.sendto = sendto


def require_device(torch, device):
    """Check the requested accelerator before claiming preflight success."""
    resolved = torch.device(device)
    if resolved.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('Apple MPS unavailable. Run from macOS Terminal, not the restricted sandbox.')
    if resolved.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; check the NVIDIA driver and CUDA-enabled PyTorch installation.')
        index = resolved.index if resolved.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise RuntimeError(f'CUDA device index {index} is unavailable')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/laya-full-ce.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="Check hashes, config, dependencies and requested device without training")
    mode.add_argument("--smoke-test", action="store_true", help="8 train / 4 validation rows; real full-model updates")
    mode.add_argument("--validate-config", action="store_true", help="Validate config and installed trainer without data, model or accelerator access")
    args = parser.parse_args()
    spec = json.loads((ROOT / args.config).read_text())
    lock = json.loads((ROOT / "configs/laya_model_lock.json").read_text())
    if version("laya") != lock["laya_version"]:
        raise ValueError("Laya version differs from model lock")
    optimization = validate_optimization(spec.get('optimization'))
    epoch_checkpoints = validate_epoch_checkpoints(spec.get('epoch_checkpoints', False))
    sampling = validate_sampling(spec.get('sampling'))
    if (optimization and optimization.get('rdrop_alpha', 0) > 0
            and spec['training']['loss'] != 'soft-ce'):
        raise ValueError('R-Drop only supports soft-ce training')
    runtime_profile = prepare_runtime(spec)
    block_network()
    import laya.train as upstream
    upstream.TrainConfig(**spec['training']).validate()
    if args.validate_config:
        print(json.dumps({'state': 'config_valid', 'run_id': spec['run_id'],
                          'optimization': optimization, 'accelerator_checked': False,
                          'data_accessed': False, 'model_accessed': False}))
        return
    model_dir = ROOT / lock["local_dir"]
    for name, expected in lock["sha256"].items():
        if file_digest(model_dir / name) != expected:
            raise ValueError("Local Laya model hash mismatch: " + name)
    import torch
    runtime = apply_runtime(torch, runtime_profile)

    device = spec["device"]
    require_device(torch, device)
    manifest = json.loads((ROOT / "reports/split_manifest.json").read_text())
    for split in ("train", "validation"):
        if file_digest(ROOT / "data/processed" / (split + ".jsonl")) != manifest["jsonl_sha256"][split]:
            raise ValueError("Frozen dataset hash mismatch: " + split)
    print("Offline checks passed: local model, frozen data, pinned Laya, device=" + device, flush=True)
    if args.check_only:
        return
    if args.smoke_test:
        spec = dict(spec, run_id=spec['run_id'] + '-smoke', train_limit=8, validation_limit=4,
                    training=dict(spec["training"], epochs=1, calib_max=2, calib_frac=0.25,
                                  micro_batch=1, grad_accum=2, log_every=1))
    output = ROOT / "checkpoints" / spec["run_id"]
    report_dir = ROOT / "reports" / spec["run_id"]
    if output.exists() or report_dir.exists():
        raise FileExistsError("Run already exists; change run_id in config to start another run: " + spec["run_id"])
    output.mkdir(parents=True)
    report_dir.mkdir(parents=True)
    data_dir = report_dir / "data"
    data_dir.mkdir()
    seed = spec["training"]["seed"]
    train, train_info = prepare_rows("train", spec["train_limit"], seed, data_dir, manifest)
    validation, validation_info = prepare_rows("validation", spec["validation_limit"], seed, data_dir, manifest)
    config = upstream.TrainConfig(**dict(spec["training"], eval_data=str(validation)))
    config.validate()
    gradient_ids, calibration_ids = upstream.split_calibration(
        train_info["ids"], config.calib_max, config.calib_frac, config.calib_seed)
    write_json(report_dir / "manifest.json", {"spec": spec, "model_lock": lock,
               "train": train_info, "validation": validation_info, "question": QUESTION,
               "gradient_train_ids": gradient_ids, "calibration_ids": calibration_ids,
               "versions": {name: version(name) for name in ("laya", "torch", "transformers", "numpy")},
               "script_sha256": file_digest(Path(__file__)),
               "uv_lock_sha256": file_digest(ROOT / "uv.lock"),
               "upstream_train_sha256": file_digest(Path(upstream.__file__)),
               "cuda_fail_fast_guards": device.startswith('cuda'),
               "cuda_guard_sha256": (file_digest(ROOT / 'scripts/cuda_numerics.py') if device.startswith('cuda') else None),
               "runtime": runtime,
               "runtime_adapter_sha256": (file_digest(ROOT / "scripts/laya_runtime.py") if runtime else None),
               "optimization": optimization,
               "sampling": sampling,
               "sampling_adapter_sha256": (file_digest(ROOT / "scripts/laya_resampling.py") if sampling else None),
               "epoch_checkpoints": epoch_checkpoints,
               "epoch_checkpoint_adapter_sha256": (file_digest(ROOT / 'scripts/laya_epoch_checkpoints.py') if epoch_checkpoints else None),
               "optimization_adapter_sha256": (file_digest(ROOT / 'scripts/laya_optimization.py') if optimization is not None else None),
               "rdrop_adapter_sha256": (file_digest(ROOT / 'scripts/laya_rdrop.py')
                                        if optimization and optimization.get('rdrop_alpha', 0) > 0 else None),
               "label_smoothing_scope": ('gradient_training_only' if optimization is not None else 'upstream'),
               "network": "HF offline flags and Python IPv4/IPv6 socket guard enabled"})
    # Upstream's fixed eval batch=16 can consume too much MPS memory. Only its
    # evaluation batch size is adapted here. Opt-in training adaptation below
    # preserves the upstream loop/loss/checkpoint code.
    upstream.calibration_records = partial(upstream.calibration_records,
                                            batch_size=spec["eval_batch_size"])
    log_path = report_dir / "console.log"

    class Tee:
        def __init__(self, terminal, log):
            self.terminal, self.log = terminal, log

        def write(self, text):
            self.terminal.write(text)
            self.log.write(text)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()

    import sys
    original_out, original_err = sys.stdout, sys.stderr
    started = time.monotonic()
    status = {"state": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
    write_json(report_dir / "status.json", status)
    with log_path.open("w", encoding="utf-8") as log:
        sys.stdout, sys.stderr = Tee(original_out, log), Tee(original_err, log)
        try:
            print("Full Laya encoder + decision head fine-tuning; objective=" + config.loss, flush=True)
            print("Training rows %d; validation rows %d; calibration is held out FROM TRAINING."
                  % (train_info["count"], validation_info["count"]), flush=True)
            print("Loading local model; initial validation runs before the first training step.", flush=True)
            from cuda_numerics import guard_training
            with ExitStack() as stack:
                if device.startswith('cuda'):
                    stack.enter_context(guard_training(torch, upstream))
                if sampling:
                    gradient_id_set = set(gradient_ids)
                    label_by_id = {r['id']: r['label'] for r in read_jsonl(ROOT / 'data/processed/train.jsonl') if r['id'] in gradient_id_set}
                    stack.enter_context(resample_training(upstream, sampling, gradient_ids, label_by_id, report_dir))
                stack.enter_context(optimize_training(torch, upstream, optimization, report_dir))
                stack.enter_context(keep_epoch_checkpoints(upstream, epoch_checkpoints, output, report_dir))
                summary = upstream.finetune(str(train), str(model_dir), str(output), config, device)
            status.update(state="completed", elapsed_seconds=time.monotonic() - started,
                          mps_driver_allocated_bytes=(torch.mps.driver_allocated_memory() if device == "mps" else None))
            write_json(report_dir / "result.json", summary)
            print("Finished. Report: " + str(report_dir / "result.json"), flush=True)
        except BaseException as exc:
            status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error=repr(exc), elapsed_seconds=time.monotonic() - started)
            raise
        finally:
            write_json(report_dir / "status.json", status)
            sys.stdout, sys.stderr = original_out, original_err


if __name__ == "__main__":
    main()

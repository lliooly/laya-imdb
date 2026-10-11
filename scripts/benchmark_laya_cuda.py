"""Offline FP32 full-length training-step timing; no research checkpoint is saved."""
import argparse
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import train_laya
from common import ROOT, file_digest, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose an unused benchmark output')
    import torch
    import laya.train as t
    train_laya.block_network()
    train_laya.require_device(torch, 'cuda')
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lock = json.loads((ROOT / 'configs/laya_model_lock.json').read_text())
    model_dir = ROOT / lock['local_dir']
    for name, digest in lock['sha256'].items():
        if file_digest(model_dir / name) != digest:
            raise ValueError('Original model hash mismatch')
    source = ROOT / 'data/processed/train.jsonl'
    split = json.loads((ROOT / 'reports/split_manifest.json').read_text())
    if file_digest(source) != split['jsonl_sha256']['train']:
        raise ValueError('Frozen training hash mismatch')
    model, tok, _ = t.load_checkpoint(str(model_dir))
    def state_digest():
        digest = hashlib.sha256()
        for name, value in model.named_parameters():
            digest.update(name.encode())
            array = value.detach().cpu().contiguous().numpy().copy()
            # Adding an exact zero can change -0.0 to +0.0 without a numeric update.
            array[array == 0] = 0
            digest.update(array.tobytes())
        return digest.hexdigest()
    initial_digest = state_digest()
    # Only real training examples that actually fill the 512-token sequence.
    selected, ids = [], []
    for row in read_jsonl(source):
        typed = {'state': row['text'], 'questions': {'sentiment': train_laya.QUESTION},
                 'expected': {'sentiment': ('negative', 'positive')[row['label']]}}
        items, skipped = t.items_from_rows(tok, [typed], 512, 192)
        if not skipped:
            encoded = t.encode_item(tok, items[0], 512, 192)
            if len(encoded['ids']) == 512:
                selected.append(encoded)
                ids.append(row['id'])
        if len(selected) == 8:
            break
    if len(selected) != 8:
        raise ValueError('Need eight real full-length training examples')
    device = torch.device('cuda')
    model.to(device).train()
    model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.head_checkpointing = True
    # Zero LR measures optimizer work without altering the original model parameters.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.01)
    params = list(model.parameters())
    results = []
    started = time.monotonic()
    for size in (1, 4, 8):
        optimizer.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
        batch = t.collate_items([selected[:size]], tok.pad_token_id)
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        durations = []
        torch.cuda.reset_peak_memory_stats()
        for iteration in range(7):
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            begin = time.perf_counter()
            logits = t._forward(model, batch, device, amp=False, detach_encoder=False)
            loss = t.soft_ce_loss(logits, batch['target'], batch['marker_mask'])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite benchmark loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            duration = time.perf_counter() - begin
            if iteration >= 2:
                durations.append(duration)
        import statistics
        median = statistics.median(durations)
        entry = {'micro_batch': size, 'sequence_length': 512, 'sample_ids': ids[:size],
            'warmup_steps': 2, 'measured_steps': 5, 'seconds': durations,
            'median_seconds': median, 'reviews_per_second': size / median,
            'estimated_gradient_train_seconds_22100': 22100 / size * median,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
        results.append(entry)
        print(json.dumps(entry), flush=True)
        del batch, logits, loss
    final_digest = state_digest()
    if final_digest != initial_digest:
        raise ValueError('Zero LR benchmark unexpectedly modified model weights')
    report = {'test_accessed': False, 'checkpoint_saved': False,
        'dtype': 'float32', 'tf32': False, 'gradient_checkpointing': True,
        'optimizer_lr': 0.0, 'parameters_modified': False,
        'fp32_weights_digest_before': initial_digest, 'fp32_weights_digest_after': final_digest,
        'digest_scope': 'Named parameters, signed zeros normalized; excludes runtime buffers',
        'timing_scope': 'Synchronized forward/backward/clip/AdamW with preloaded GPU inputs; excludes tokenization, checkpoint IO and validation. Zero LR benchmark is not an accuracy experiment.',
        'gpu': torch.cuda.get_device_name(), 'source_sha256': file_digest(Path(__file__)),
        'train_sha256': split['jsonl_sha256']['train'], 'uv_lock_sha256': file_digest(ROOT / 'uv.lock'),
        'versions': {name: version(name) for name in ('torch', 'laya', 'transformers')},
        'elapsed_seconds': time.monotonic() - started, 'results': results}
    write_json(args.out, report)


if __name__ == '__main__':
    main()

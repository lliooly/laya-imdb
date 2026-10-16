"""Bounded training-only 512/2048 admission; never an accuracy experiment."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

STUDY = 'laya-long-context-pilot-20261016-v1'
BASE = 'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json'
ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def verify_contract(root=ROOT):
    from common import file_digest
    c = read(root/'configs'/f'{STUDY}.json')
    if c['study_id'] != STUDY or c['maximum_additional_compute_usd'] != 1:
        raise ValueError('Pilot scope changed')
    for rel, sha in c['source_sha256'].items():
        if file_digest(root/rel) != sha:
            raise ValueError('Bound source changed: '+rel)
    lock = read(root/'configs/laya_model_lock.json')
    for rel, sha in lock['sha256'].items():
        if file_digest(root/lock['local_dir']/rel) != sha:
            raise ValueError('Published asset changed: '+rel)
    return c, lock


def prepare(tok, rows, cap):
    """Keep full tokens until the final, length-aware state budget is applied."""
    import laya.train as upstream
    from laya.common import build_head
    from train_laya import QUESTION
    from laya_decision_views import full_tokens, window_tokens
    if cap not in (512, 2048):
        raise ValueError('Only the two approved lengths are allowed')
    typed = [{'state': r['text'], 'questions': {'sentiment': QUESTION},
              'expected': {'sentiment': ('negative', 'positive')[r['label']]}} for r in rows]
    items, skipped = upstream.items_from_rows(tok, typed, cap, 192, label_smoothing=0.)
    if skipped or len(items) != len(rows):
        raise ValueError('Every gradient row must produce exactly one item')
    lengths, prepared = [], []
    for item, full in zip(items, full_tokens(tok, [r['text'] for r in rows]), strict=True):
        head, _, _ = build_head(tok, item['q'], 192)
        if len(head) != 45:
            raise ValueError('Question template changed')
        chosen = window_tokens(full, 'head_tail', cap-46)
        value = upstream.encode_item(tok, dict(item, state_ids=chosen), cap, 192, None, False)
        if value['ids'][45:-1] != chosen or len(value['ids']) != len(chosen)+46:
            raise ValueError('Review was silently truncated or re-tokenized')
        lengths.append(len(full)); prepared.append(value)
    return prepared, lengths


def source_model(directory):
    """Reuse published CE weights, including the decision head, without resetting it."""
    from safetensors import safe_open
    from laya_head_control import from_source
    model = from_source(directory, 'original', 42)
    keys = set(model.state_dict())
    with safe_open(Path(directory)/'model.safetensors', framework='pt', device='cpu') as f:
        state = {key: f.get_tensor(key).float() for key in f.keys() if key in keys}
    model.load_state_dict(state, strict=True)
    return model


def strict_runtime():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    from laya_runtime import apply_runtime
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this pilot')
    apply_runtime(torch, 'strict')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch


def train():
    c, lock = verify_contract()
    output = ROOT/'reports'/STUDY
    if (output/'summary.json').exists() or (output/'started.json').exists():
        raise FileExistsError('Pilot already started; use a new version')
    from common import read_jsonl, write_json, file_digest
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'started.json', {'started_unix': time.time(), 'test_accessed': False})
    torch = strict_runtime()
    import laya.train as upstream
    from transformers import AutoTokenizer
    from laya_head_control import save_fp32, state_digest
    from laya_robust_training import capture_rng, finite_tree
    from train_laya import block_network
    block_network()
    manifest = read(ROOT/BASE)
    rows = list(read_jsonl(ROOT/'data/processed/train.jsonl'))
    if [r['id'] for r in rows] != manifest['train']['ids']:
        raise ValueError('Frozen training order changed')
    by_id = {r['id']: r for r in rows}
    gids = manifest['gradient_train_ids']
    if len(gids) != 22100 or len(set(gids)) != 22100 or set(gids)&set(manifest['calibration_ids']):
        raise ValueError('Invalid gradient/calibration partition')
    rows = [by_id[i] for i in gids]
    tok = AutoTokenizer.from_pretrained(str(ROOT/lock['local_dir']/'tokenizer'), local_files_only=True)
    from laya_decision_views import full_tokens
    lengths = [len(t) for t in full_tokens(tok, [r['text'] for r in rows])]
    longest = sorted(range(len(rows)), key=lambda i: (-lengths[i], i))[:8]
    selected = {'long': [rows[i] for i in longest], 'ordinary': rows[:8]}
    result = {'study_id': STUDY, 'state': 'running', 'test_accessed': False,
              'development_accessed': False, 'formal_training': False,
              'dtype': 'float32', 'amp': False, 'tf32': False,
              'gpu': torch.cuda.get_device_name(), 'cases': [],
              'coverage': [{'max_len': cap, 'complete': sum(n <= cap-46 for n in lengths),
                            'count': len(lengths)} for cap in (512, 1024, 1536, 2048, 4096)],
              'contract_sha256': file_digest(ROOT/'configs'/f'{STUDY}.json')}
    write_json(output/'summary.json', result)
    for cap, size, group in ((512,8,'long'), (2048,1,'long'), (2048,8,'long'), (2048,8,'ordinary')):
        chosen = selected[group][:size]
        inputs, review_lengths = prepare(tok, chosen, cap)
        case = {'max_len': cap, 'micro_batch': size, 'group': group,
                'sample_ids': [r['id'] for r in chosen], 'review_lengths': review_lengths,
                'input_lengths': [len(x['ids']) for x in inputs], 'input_retention_verified': True}
        model = source_model(ROOT/lock['local_dir'])
        case['initial_state_sha256'] = state_digest(model.state_dict())
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.head_checkpointing = True
        model.to('cuda').train()
        optimizer = torch.optim.AdamW([
            {'params': [p for n,p in model.named_parameters() if n.startswith('encoder.')], 'lr': 2.5e-5},
            {'params': [p for n,p in model.named_parameters() if not n.startswith('encoder.')], 'lr': 1e-4}],
            weight_decay=.01)
        batch = upstream.collate_items([inputs], tok.pad_token_id)
        durations, losses, norms = [], [], []
        torch.manual_seed(42); torch.cuda.reset_peak_memory_stats()
        try:
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(); begin = time.perf_counter()
                logits = upstream._forward(model, batch, torch.device('cuda'), False, False)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite logits')
                loss = upstream.soft_ce_loss(logits, batch['target'].to('cuda'), batch['marker_mask'].to('cuda'))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite updated parameter')
                torch.cuda.synchronize()
                if step:
                    durations.append(time.perf_counter()-begin)
                losses.append(float(loss.detach())); norms.append(float(norm))
            case.update(state='passed', steps=3, warmup_steps=1, measured_seconds=durations,
                        median_step_seconds=statistics.median(durations), loss=losses, grad_norm=norms,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if cap == 2048 and size == 8 and group == 'long':
                bundle = ROOT/'checkpoints'/STUDY/'pilot-2048'
                optimizer.zero_grad(set_to_none=True)
                model.eval()
                with torch.no_grad():
                    reference = upstream._forward(model, batch, torch.device('cuda'), False, False).cpu().tolist()
                metadata = save_fp32(model, bundle)
                # This is a three-step diagnostic snapshot, not a formal epoch resume package.
                from laya_robust_training import cpu_tree
                snapshot = {'format': 'laya-long-context-pilot-snapshot-v1',
                            'model_state_sha256': metadata['state_sha256'], 'updates': 3,
                            'optimizer': cpu_tree(optimizer.state_dict()), 'rng': capture_rng(torch.device('cuda'))}
                if not finite_tree(snapshot):
                    raise FloatingPointError('Nonfinite saved optimizer state')
                torch.save(snapshot, bundle/'optimizer_rng.pt')
                write_json(bundle/'reference.json', {'inputs': inputs, 'logits': reference,
                           'pad_token_id': tok.pad_token_id, 'max_len': cap})
                subprocess.run([sys.executable, str(Path(__file__)), '--verify', str(bundle), '--device', 'cpu'],
                               check=True, timeout=180)
                case['checkpoint'] = str(bundle.relative_to(ROOT))
        except torch.cuda.OutOfMemoryError as exc:
            case.update(state='out_of_memory', error=str(exc), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        result['cases'].append(case)
        write_json(output/'summary.json', result)
        print(json.dumps(case), flush=True)
        del model, optimizer, batch
        if 'logits' in locals(): del logits
        if 'loss' in locals(): del loss
        gc.collect(); torch.cuda.empty_cache()
    result['state'] = 'passed' if all(x['state']=='passed' for x in result['cases']) else 'failed_admission'
    result['same_initialization_verified'] = len({x['initial_state_sha256'] for x in result['cases']}) == 1
    if not result['same_initialization_verified']:
        raise ValueError('Pilot cases did not start from identical weights')
    write_json(output/'summary.json', result)


def verify(bundle, device):
    import torch
    import laya.train as upstream
    from laya_head_control import load_fp32
    from laya_robust_training import finite_tree
    torch.set_num_threads(4)
    bundle = Path(bundle)
    model, meta = load_fp32(bundle)
    snapshot = torch.load(bundle/'optimizer_rng.pt', map_location='cpu', weights_only=True)
    if snapshot['model_state_sha256'] != meta['state_sha256'] or snapshot['updates'] != 3 or not finite_tree(snapshot):
        raise ValueError('Pilot recovery snapshot differs')
    del snapshot
    ref = read(bundle/'reference.json')
    batch = upstream.collate_items([ref['inputs']], ref['pad_token_id'])
    model.to(device).eval()
    with torch.no_grad():
        actual = upstream._forward(model, batch, torch.device(device), False, False).cpu()
    # CPU and GPU math need not be bitwise identical. Report the measured tolerance.
    expected = torch.tensor(ref['logits'])
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    from common import write_json
    result = {'strict_load': True, 'optimizer_rng_finite': True, 'device': device,
              'state_sha256': meta['state_sha256'], 'max_abs_logit_difference': float((actual-expected).abs().max()),
              'rtol': 1e-4, 'atol': 1e-4}
    write_json(bundle/f'verification-{device}.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    if args.verify:
        verify(args.verify, args.device)
    else:
        train()

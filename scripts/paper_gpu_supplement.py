"""Seed-explicit, test-sealed supplementary training on one rented GPU.

Historical trainers and their source bindings are never modified. Laya delegates
to the original training loop and optimization adapter; DeBERTa delegates each
update to the original BF16 implementation, with explicit initialization/order RNG.
"""
import os

for _key in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_DATASETS_OFFLINE',
             'HF_HUB_DISABLE_TELEMETRY', 'DO_NOT_TRACK'):
    os.environ[_key] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import shutil
import time

from common import ROOT, file_digest, read_jsonl, write_json

STUDY = 'paper-gpu-supplement-20261019-v1'
TOKEN_ROOT = 'artifacts/deberta-heterogeneous-bf16-budget7-s42-20261017-v1'
BASE_MANIFEST = 'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json'
SEEDS = {'baseline': (43, 143, 243), 'combined': (42, 142, 242), 'deberta': (42, 142, 242)}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def paths(kind, seed, pilot=False):
    if kind not in SEEDS or seed not in SEEDS[kind]:
        raise ValueError('Run is outside the fixed three-repetition schedule')
    run = f'{STUDY}-{kind}-s{seed}' + ('-admission' if pilot else '')
    return run, ROOT/'reports'/run, ROOT/'checkpoints'/run


def runtime():
    import torch
    return dict(versions={n: version(n) for n in ('torch', 'transformers', 'numpy', 'safetensors')},
                gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda,
                bf16_native=torch.cuda.is_bf16_supported(including_emulation=False),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                tf32=torch.backends.cuda.matmul.allow_tf32,
                source_sha256=file_digest(Path(__file__)), test_accessed=False)


def setup(seed):
    import torch
    from deberta_training import seed_all
    seed_all(seed)
    torch.set_num_threads(8)
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Require one CUDA GPU')
    return torch.device('cuda')


def seed_audit():
    from laya_head_control import state_digest
    from deberta_training import groups_for, seed_all
    import torch
    out = ROOT/'reports'/STUDY/'seed-admission.json'
    if out.exists():
        raise FileExistsError(out)
    records = []
    for seed in (42, 142, 242, 42):
        model = initial_deberta(seed)
        state = model.state_dict()
        record = dict(seed=seed,
            encoder=state_digest({k: v for k, v in state.items() if k.startswith('deberta.')}),
            classifier=state_digest({k: v for k, v in state.items() if k.startswith('classifier.')}),
            pooler=state_digest({k: v for k, v in state.items() if k.startswith('pooler.')}),
            first_group=groups_for(22100, 0, seed)[0])
        seed_all(seed)
        record['dropout_rng_sha256'] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        records.append(record)
        del state, model
        gc.collect()
    if (records[0] != records[-1] or len({r['encoder'] for r in records}) != 1
            or len({r['classifier'] for r in records}) != 3
            or len({tuple(r['first_group']) for r in records}) != 3
            or len({r['dropout_rng_sha256'] for r in records}) != 3):
        raise ValueError('Initialization, dropout or order seed admission failed')
    write_json(out, dict(state='passed', records=records, test_accessed=False))


def verified_training_rows():
    from deberta_inputs import training_rows
    rows, split = training_rows(ROOT)
    for field in ('gradient_train_ids', 'calibration_ids'):
        if len(split[field]) != len(set(split[field])):
            raise ValueError('Repeated partition IDs')
    return rows, split


def initial_deberta(seed, loading_path=None):
    """Same loader/strict checks as the historical implementation, explicit seed."""
    import torch
    from transformers import AutoModelForSequenceClassification
    from deberta_training import seed_all
    seed_all(seed)
    model, info = AutoModelForSequenceClassification.from_pretrained(
        str(ROOT/'.cache/deberta-v3-large'), local_files_only=True, num_labels=2,
        id2label={0: 'neg', 1: 'pos'}, label2id={'neg': 0, 'pos': 1},
        dtype=torch.float32, output_loading_info=True, attn_implementation='eager')
    if (any(not k.startswith(('classifier.', 'pooler.')) for k in info['missing_keys'])
            or info['mismatched_keys'] or info['error_msgs']):
        raise ValueError('Unexpected pretrained encoder loading difference')
    for key in info['unexpected_keys']:
        unused = key == 'deberta.embeddings.position_embeddings.weight' and not model.config.position_biased_input
        if not unused and not key.startswith(('lm_predictions.', 'mask_predictions.')):
            raise ValueError('Unexpected pretrained tensor: '+key)
    if loading_path:
        write_json(loading_path, info)
    return model.float()


def seal(folder, **metadata):
    files = [dict(path=str(p.relative_to(folder)), bytes=p.stat().st_size, sha256=file_digest(p))
             for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'bundle.json']
    write_json(folder/'bundle.json', dict(format='paper-supplement-bundle-v1', files=files, **metadata))


def check_bundle(folder):
    from laya_checkpoint_delivery import within
    manifest = read(folder/'bundle.json')
    if manifest['format'] != 'paper-supplement-bundle-v1':
        raise ValueError('Unknown bundle format')
    for record in manifest['files']:
        path = within(folder, record['path'])
        if path.stat().st_size != record['bytes'] or file_digest(path) != record['sha256']:
            raise ValueError('Corrupt bundle: '+record['path'])
    return manifest


def train_deberta(seed, pilot=False):
    import torch
    import deberta_bf16_training as core
    from laya_head_control import state_digest
    device = setup(seed)
    run, report, checkpoint = paths('deberta', seed, pilot)
    report.mkdir(parents=True, exist_ok=False)
    rows, split = verified_training_rows()
    items = list(read_jsonl(ROOT/TOKEN_ROOT/'train-tokens.jsonl'))
    stats = read(ROOT/TOKEN_ROOT/'input-summary.json')
    if (file_digest(ROOT/TOKEN_ROOT/'train-tokens.jsonl') != stats['train_tokens_sha256']
            or [x['id'] for x in items] != split['gradient_train_ids']
            or any(x['label'] != rows[x['id']]['label'] for x in items)):
        raise ValueError('Frozen token cache changed')
    model = initial_deberta(seed, report/'loading.json')
    initial = {name: state_digest({k: v for k, v in model.state_dict().items() if k.startswith(prefix)})
               for name, prefix in (('encoder', 'deberta.'), ('new_head', 'classifier.'))}
    model = core.setup_model(model, device)
    optimizer, scheduler = core.optimizer_for(model)
    core.seed_all(seed)
    spec = read(ROOT/'configs/deberta-heterogeneous-bf16-budget7-s42-20261017-v1.json')
    write_json(report/'protocol.json', dict(run_id=run, kind='deberta', seed=seed,
        recipe=dict(spec['recipe'], seed=seed), initial_state_sha256=initial, runtime=runtime(),
        gradient_ids_sha256=hashlib.sha256(json.dumps(split['gradient_train_ids']).encode()).hexdigest(),
        official_test_accessed=False, development_used_for_selection=False,
        initialization_seed=seed, dropout_seed=seed, data_order_seed=seed))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    updates = exposures = 0
    update_seconds = 0.
    history = []
    stress = sorted(range(len(items)), key=lambda i: (-len(items[i]['input_ids']), i))[:32]
    with (report/'updates.jsonl').open('x', buffering=1) as log:
        for epoch in range(1 if pilot else 2):
            groups = core.groups_for(len(items), epoch, seed=seed)
            groups = groups[:3]+[stress] if pilot else groups
            loss_sum = 0.
            for step, indices in enumerate(groups):
                value = core.update(model, [items[i] for i in indices], stats['pad_token_id'],
                                    optimizer, scheduler, device, micro=8)
                updates += 1
                exposures += len(indices)
                update_seconds += value['seconds']
                loss_sum += value['loss']*len(indices)
                log.write(json.dumps(dict(epoch=epoch+1, update=updates,
                    ids=[items[i]['id'] for i in indices], **value))+'\n')
                if step < 3 or (step+1) % 16 == 0:
                    print(run, 'epoch', epoch+1, 'step', step+1, 'of', len(groups), value, flush=True)
            history.append(dict(epoch=epoch+1, mean_loss=loss_sum/sum(map(len, groups))))
            folder = checkpoint/f'epoch-{epoch+1:03d}'
            folder.mkdir(parents=True, exist_ok=False)
            model.save_pretrained(folder, safe_serialization=True)
            probe = [items[i] for i in groups[0][:8]]
            reference, _ = core.predict(model, probe, stats['pad_token_id'], device)
            write_json(folder/'reference.json', dict(items=probe, logits=reference, pad=stats['pad_token_id']))
            progress = dict(updates=updates, exposures=exposures, completed_epoch=epoch+1, seed=seed,
                            history=history, pilot=pilot, elapsed_seconds=time.monotonic()-started)
            write_json(folder/'training.json', progress)
            if epoch == 0 and not pilot:
                torch.save(core.snapshot(model, optimizer, scheduler, device, dict(run_id=run, seed=seed), progress),
                           folder/'optimizer_rng.pt')
            seal(folder, run_id=run, kind='deberta', seed=seed, epoch=epoch+1,
                 master_dtype='float32', compute_dtype='bfloat16', pilot=pilot)
    write_json(report/'training-complete.json', dict(state='trained', run_id=run, pilot=pilot,
        updates=updates, exposures=exposures, seconds=time.monotonic()-started,
        training_seconds=update_seconds,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), history=history, test_accessed=False))


def laya_items(tok, rows):
    import laya.train as upstream
    from train_laya import QUESTION
    typed = [dict(state=r['text'], questions={'sentiment': QUESTION},
                  expected={'sentiment': ('negative', 'positive')[r['label']]}) for r in rows]
    items, skipped = upstream.items_from_rows(tok, typed, 512, 192, label_smoothing=0.)
    if skipped or len(items) != len(rows):
        raise ValueError('Laya preprocessing skipped rows')
    return items


def train_laya_model(kind, seed, pilot=False):
    import torch
    import laya.train as upstream
    from cuda_numerics import guard_training
    from laya_optimization import optimize_training
    from laya_head_control import state_digest
    from laya_robust_training import capture_rng, restore_rng
    device = setup(seed)
    run, report, checkpoint = paths(kind, seed, pilot)
    report.mkdir(parents=True, exist_ok=False)
    rows, split = verified_training_rows()
    template = ('laya-rtx4090-ce-baseline-s43-epochs2-fp32.json' if kind == 'baseline'
                else 'laya-rtx4090-ce-epochs2-warmup005-ls005-fp32.json')
    spec = read(ROOT/'configs'/template)
    spec['training']['seed'] = seed
    model, tok, config = upstream.load_checkpoint(str(ROOT/'.cache/laya/english'))
    gradient = laya_items(tok, [rows[i] for i in split['gradient_train_ids']])
    calibration = laya_items(tok, [rows[i] for i in split['calibration_ids']])
    if pilot:
        order = list(range(len(gradient)))
        random.Random(seed).shuffle(order)
        gradient = [gradient[i] for i in order[:24]] + sorted(gradient, key=lambda x: -len(x['state_ids']))[:8]
        spec['training']['epochs'] = 1
        # Four original-size updates, including maximum-length inputs.
    cfg = upstream.TrainConfig(**spec['training'])
    initial_sha = state_digest(model.state_dict())
    write_json(report/'protocol.json', dict(run_id=run, kind=kind, seed=seed, spec=spec,
        runtime=runtime(), initial_state_sha256=initial_sha,
        pretrained_head_preserved=True, dropout_seed=seed, data_order_seed=seed,
        upstream_loop_sha256=hashlib.sha256(__import__('inspect').getsource(upstream.train_model).encode()).hexdigest(),
        gradient_ids=split['gradient_train_ids'], calibration_ids=split['calibration_ids'],
        development_used_for_selection=False, test_accessed=False))
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    checkpoint_seconds = 0.
    def epoch_end(epoch, loss):
        nonlocal checkpoint_seconds
        began = time.monotonic()
        # Preserve RNG across archival/probe forwards. Saved storage matches the
        # historical pipeline: FP32 training, FP16 Laya checkpoint, FP32 reload.
        state = capture_rng(device)
        training_mode = model.training
        folder = checkpoint/f'epoch-{epoch+1:03d}'
        if folder.exists():
            raise FileExistsError(folder)
        upstream.save_checkpoint(model, tok, dict(config, max_len=512, head_max_len=192), str(folder))
        write_json(folder/'training.json', dict(epoch=epoch+1, mean_loss=loss, seed=seed, pilot=pilot))
        restore_rng(state, device)
        model.train(training_mode)
        checkpoint_seconds += time.monotonic()-began
    options = spec.get('optimization')
    # A four-step pilot needs a valid warmup (ceil(.05*4)=1).
    train_started = time.monotonic()
    with guard_training(torch, upstream), optimize_training(torch, upstream, options, report):
        history = upstream.train_model(model, tok, gradient, cfg, device, 512, 192,
                                       on_epoch_end=epoch_end, parallel=False)
    training_seconds = time.monotonic()-train_started-checkpoint_seconds
    records = upstream.calibration_records(model, tok, calibration, device, 512, 192, batch_size=8)
    fitted = upstream.fit_temperature_map(records)
    native = float(fitted['temperature'][0])
    final = checkpoint/f'epoch-{1 if pilot else 2:03d}'
    saved_config = read(final/'rl_agent_config.json')
    saved_config.update(temperature=fitted['temperature'], temperature_by_options=fitted['temperature_by_options'])
    write_json(final/'rl_agent_config.json', saved_config)
    write_json(report/'native-calibration.json', dict(temperature=native, fitted=fitted,
        logits=[r[1].tolist() for r in records], sample_ids=split['calibration_ids'],
        fit_on='final_fp32_training_model', test_accessed=False))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded, _, _ = upstream.load_checkpoint(str(final))
    reloaded = reloaded.to(device).eval()
    probe = calibration[:8]
    reference = upstream.calibration_records(reloaded, tok, probe, device, 512, 192, batch_size=8)
    write_json(final/'reference.json', dict(items=probe, logits=[r[1].tolist() for r in reference]))
    seal(final, run_id=run, kind=kind, seed=seed, epoch=1 if pilot else 2,
         master_dtype='float32', storage_dtype='float16', inference_dtype='float32', pilot=pilot)
    write_json(report/'training-complete.json', dict(state='trained', run_id=run, pilot=pilot,
        updates=__import__('math').ceil(len(gradient)/8)*cfg.epochs, exposures=len(gradient)*cfg.epochs,
        seconds=time.monotonic()-started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        training_seconds=training_seconds, epoch_checkpoint_seconds=checkpoint_seconds,
        native_temperature=native, history=history, test_accessed=False))


def verify(kind, seed, pilot=False):
    import numpy as np
    import torch
    device = setup(seed)
    run, report, checkpoint = paths(kind, seed, pilot)
    folder = checkpoint/f'epoch-{1 if pilot else 2:03d}'
    manifest = check_bundle(folder)
    reference = read(folder/'reference.json')
    if kind == 'deberta':
        import deberta_bf16_training as core
        model = core.load_saved(folder).to(device).eval()
        z, _ = core.predict(model, reference['items'], reference['pad'], device)
    else:
        import laya.train as upstream
        model, tok, _ = upstream.load_checkpoint(str(folder))
        model = model.to(device).eval()
        records = upstream.calibration_records(model, tok, reference['items'], device, 512, 192, batch_size=8)
        z = [r[1].tolist() for r in records]
    if not np.array_equal(np.asarray(z), np.asarray(reference['logits'])):
        raise ValueError('Independent process checkpoint reload logits differ')
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise FloatingPointError('Nonfinite reloaded model')
    write_json(report/'reload-verification.json', dict(state='passed', strict_load=True,
        independent_process=True, logits_bitwise_equal=True, bundle_sha256=file_digest(folder/'bundle.json'),
        manifest=manifest, test_accessed=False, verified_utc=now()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train', 'verify', 'seed-audit'))
    p.add_argument('--kind', choices=tuple(SEEDS))
    p.add_argument('--seed', type=int)
    p.add_argument('--pilot', action='store_true')
    a = p.parse_args()
    if a.mode == 'seed-audit':
        seed_audit()
        return
    run, report, _ = paths(a.kind, a.seed, a.pilot)
    try:
        if a.mode == 'verify':
            verify(a.kind, a.seed, a.pilot)
        elif a.kind == 'deberta':
            train_deberta(a.seed, a.pilot)
        else:
            train_laya_model(a.kind, a.seed, a.pilot)
    except BaseException as exc:
        if report.exists():
            write_json(report/(a.mode+'-failure.json'), dict(error=repr(exc), utc=now(), test_accessed=False))
        raise


if __name__ == '__main__':
    main()

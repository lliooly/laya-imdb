"""Extract frozen Laya features, calibrate linear probes, then evaluate development."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

import numpy as np

import train_laya
from common import ROOT, file_digest, read_jsonl, write_json
from export_laya_posthoc import prepare_rules, input_digest
from laya_decision_views import stratified_metrics
from laya_frozen_probe import (LAMBDAS, LOCATIONS, MarkerCapture, standardize,
    fit_readout, logits, probability, score, select_candidate, paired_accuracy)
from run_laya_risk_coverage import wilson


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(value):
    p = Path(value)
    if p.is_absolute() or '..' in p.parts or not (ROOT/p).resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Require repository-relative paths')
    return ROOT/p


def validate(spec):
    expected = {'run_id', 'device', 'batch_size', 'checkpoint', 'manifest', 'source_sha256',
                'asset_sha256', 'cached_export', 'lambdas', 'input_rule', 'tolerance', 'bootstrap'}
    if (set(spec) != expected or spec['device'] != 'mps' or spec['batch_size'] != 8
            or spec['lambdas'] != list(LAMBDAS) or spec['input_rule'] != 'head_tail'
            or spec['tolerance'] != {'logits_absolute': .001, 'probabilities_absolute': .0001}
            or spec['bootstrap'] != {'repetitions': 10000, 'seed': 20261015}):
        raise ValueError('Unexpected fixed probe protocol')
    name = spec['run_id']
    if not isinstance(name, str) or not name or name in ('.', '..') or Path(name).name != name:
        raise ValueError('Require a new run_id')
    for key in ('checkpoint', 'manifest', 'cached_export'):
        relative(spec[key])
    required = {spec['manifest'], 'reports/split_manifest.json',
                'data/processed/train.jsonl', 'data/processed/validation.jsonl',
                f"{spec['cached_export']}/calibration-head_tail.json",
                f"{spec['cached_export']}/validation-head_tail.json"}
    if set(spec['source_sha256']) != required:
        raise ValueError('Bind exactly the required source files')
    for path, sha in spec['source_sha256'].items():
        relative(path)
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Invalid source SHA256')
    if 'model.safetensors' not in spec['asset_sha256']:
        raise ValueError('Bind checkpoint weight')
    for path in spec['asset_sha256']:
        relative(str(Path(spec['checkpoint'])/path))
    return spec


def verify_sources(spec):
    for path, sha in spec['source_sha256'].items():
        if file_digest(relative(path)) != sha:
            raise ValueError(f'Source hash mismatch: {path}')
    verify_assets(spec)


def verify_assets(spec):
    for path, sha in spec['asset_sha256'].items():
        if file_digest(relative(spec['checkpoint'])/path) != sha:
            raise ValueError(f'Checkpoint asset hash mismatch: {path}')


def admit_training(spec):
    manifest = json.loads(relative(spec['manifest']).read_text())
    split = json.loads((ROOT/'reports/split_manifest.json').read_text())
    if manifest['question'] != train_laya.QUESTION:
        raise ValueError('Question template differs')
    for name in ('train', 'validation'):
        if (manifest[name]['source_sha256'] != spec['source_sha256'][f'data/processed/{name}.jsonl']
                or manifest[name]['source_sha256'] != split['jsonl_sha256'][name]
                or manifest[name]['ids'] != split['ids'][name]):
            raise ValueError('Frozen split metadata differs')
    ids = {'gradient_train': manifest['gradient_train_ids'], 'calibration': manifest['calibration_ids'],
           'validation': manifest['validation']['ids']}
    if tuple(map(len, ids.values())) != (22100, 400, 2500):
        raise ValueError('Expected 22100/400/2500 partition')
    sets = [set(v) for v in ids.values()]
    if any(len(s) != len(v) for s, v in zip(sets, ids.values(), strict=True)):
        raise ValueError('Duplicate partition IDs')
    if any(sets[i] & sets[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Overlapping partitions')
    train = list(read_jsonl(ROOT/'data/processed/train.jsonl'))
    if [r['id'] for r in train] != split['ids']['train'] or sets[0] | sets[1] != set(split['ids']['train']):
        raise ValueError('Frozen training source IDs differ')
    by_id = {r['id']: r for r in train}
    for r in train:
        if not r['id'].startswith('train/') or type(r['label']) is not int or r['label'] not in (0, 1):
            raise ValueError('Expected original binary training data')
    original = ROOT/manifest['model_lock']['local_dir']/'tokenizer/tokenizer.json'
    if file_digest(original) != manifest['model_lock']['sha256']['tokenizer/tokenizer.json']:
        raise ValueError('Locked tokenizer differs')
    def semantics(path):
        d = json.loads(path.read_text()); d.pop('padding', None); d.pop('truncation', None)
        return d
    if semantics(original) != semantics(relative(spec['checkpoint'])/'tokenizer/tokenizer.json'):
        raise ValueError('Checkpoint tokenizer semantics differ')
    tc = json.loads((relative(spec['checkpoint'])/'tokenizer/tokenizer_config.json').read_text())
    if tc.get('tokenizer_class') in (None, 'TokenizersBackend') or isinstance(tc.get('extra_special_tokens'), list):
        raise ValueError('Refuse upstream automatic edits to historical tokenizer config')
    return manifest, ids, {key: [by_id[i] for i in ids[key]] for key in ('gradient_train', 'calibration')}


def extract(torch, upstream, model, tok, rows, split_name, spec, artifact, out):
    if (artifact/f'{split_name}.npz').exists():
        raise FileExistsError('Feature cache already exists')
    started = time.monotonic()
    blocks = {name: [] for name in LOCATIONS}
    logits_blocks, records = [], []
    device = torch.device(spec['device'])
    with torch.inference_mode(), MarkerCapture(model, torch) as capture:
        for start in range(0, len(rows), spec['batch_size']):
            chunk = rows[start:start+spec['batch_size']]
            # Ground-truth targets are intentionally replaced before input construction.
            placeholders = [dict(row, label=0) for row in chunk]
            _, tail, stats, _, _ = prepare_rules(tok, placeholders)
            prepared = [upstream.encode_item(tok, item, 512, 192, [0, 1], parallel=False) for item in tail]
            batch = upstream.collate_items([prepared], tok.pad_token_id)
            positions = batch['marker_pos'].to(device)
            tokens = batch['input_ids'].to(device)
            if not batch['marker_mask'].all().item() or not (torch.gather(tokens, 1, positions) == tok.mask_token_id).all().item():
                raise ValueError('Candidate markers must be exactly two actual MASK tokens')
            capture.begin(positions)
            z = upstream._forward(model, batch, device, False, False).cpu().numpy().copy()
            values = capture.consume()
            if z.shape != (len(chunk), 2) or not np.isfinite(z).all():
                raise FloatingPointError('Invalid full-model logits')
            for name in LOCATIONS:
                blocks[name].append(values[name])
            logits_blocks.append(z)
            for row, item, stat in zip(chunk, prepared, stats, strict=True):
                records.append({'sample_id': row['id'], 'true_label': row['label'],
                                'input_sha256': input_digest(item), **stat})
            if start % 400 == 0 or start+8 >= len(rows):
                print(f'{split_name}: {min(start+8,len(rows))}/{len(rows)}; {time.monotonic()-started:.1f}s', flush=True)
    arrays = {name: np.concatenate(parts) for name, parts in blocks.items()}
    arrays['full_logits'] = np.concatenate(logits_blocks)
    arrays['labels'] = np.array([r['label'] for r in rows])
    arrays['truncated'] = np.array([r['truncated'] for r in records], dtype=bool)
    np.savez(artifact/f'{split_name}.npz', **arrays)
    write_json(artifact/f'{split_name}-index.json', {'rows': records, 'split': split_name, 'test_accessed': False})
    write_json(out/f'{split_name}-extraction.json', {'count': len(rows), 'forward_rows': len(rows),
        'hidden_size': arrays['encoder'].shape[1], 'seconds': time.monotonic()-started,
        'truncated_count': int(arrays['truncated'].sum()),
        'cache': str((artifact/f'{split_name}.npz').relative_to(ROOT)),
        'cache_sha256': file_digest(artifact/f'{split_name}.npz'),
        'index_sha256': file_digest(artifact/f'{split_name}-index.json'), 'test_accessed': False})
    return arrays, records


def parity(spec, records, arrays, split_name, temperature):
    cached = json.loads((relative(spec['cached_export'])/f'{split_name}-head_tail.json').read_text())['rows']
    if len(cached) != len(records):
        raise ValueError('Cached prediction count differs')
    for a, b in zip(records, cached, strict=True):
        if any(a[k] != b[k] for k in ('sample_id', 'true_label', 'input_sha256', 'truncated', 'review_tokens')):
            raise ValueError('Inputs differ from historical cached inputs')
    reference = np.array([r['logits'] for r in cached])
    z = arrays['full_logits'].astype(float)/temperature
    p = np.exp(z-z.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
    dz = float(np.max(np.abs(arrays['full_logits']-reference)))
    dp = float(np.max(np.abs(p-np.array([r['probabilities'] for r in cached]))))
    mismatches = int(np.sum(arrays['full_logits'].argmax(1) != reference.argmax(1)))
    passed = (dz <= spec['tolerance']['logits_absolute'] and dp <= spec['tolerance']['probabilities_absolute'] and not mismatches)
    return {'passed': passed, 'max_abs_logits_delta': dz, 'max_abs_probability_delta': dp,
            'label_mismatches': mismatches, 'tolerance': spec['tolerance'], 'count': len(records)}


def fit(torch, training, calibration, out):
    frozen, grid = {}, {}
    began = time.monotonic()
    for name in LOCATIONS:
        mean, scale = standardize(training[name])
        candidates = []
        for penalty in LAMBDAS:
            started = time.monotonic()
            candidate = fit_readout(torch, training[name], training['labels'], penalty, mean, scale)
            candidate['training'] = score(logits(training[name], candidate, mean, scale), training['labels'])
            candidate['calibration'] = score(logits(calibration[name], candidate, mean, scale), calibration['labels'])
            candidate['seconds'] = time.monotonic()-started
            candidates.append(candidate)
            print(f"fit {name} lambda={penalty}: cal={candidate['calibration']['accuracy']:.4f}, grad={candidate['max_abs_gradient']:.3g}, {candidate['seconds']:.1f}s", flush=True)
        selected = select_candidate(candidates)
        frozen[name] = {**selected, 'train_mean': mean.tolist(), 'train_scale': scale.tolist()}
        grid[name] = [{k: v for k, v in candidate.items() if k != 'weight'} for candidate in candidates]
    write_json(out/'calibration-grid.json', grid)
    frozen_doc = {'frozen_utc': now(), 'validation_body_parsed': False,
                  'validation_predictions_used_for_selection': False, 'locations': frozen,
                  'calibration_count': 400, 'fitting_count': 22100,
                  'selection_rule': 'calibration Accuracy, then NLL, then stronger lambda',
                  'seconds': time.monotonic()-began}
    write_json(out/'frozen-fit.json', frozen_doc)
    return frozen_doc


def evaluate(spec, arrays, records, frozen, temperature, out):
    y, mask = arrays['labels'], arrays['truncated']
    z = arrays['full_logits'].astype(float)/temperature
    p = np.exp(z-z.max(axis=1, keepdims=True)); p /= p.sum(axis=1, keepdims=True)
    probabilities = {'full_model': p}
    for name in LOCATIONS:
        fit = frozen['locations'][name]
        values = logits(arrays[name], fit, np.asarray(fit['train_mean']), np.asarray(fit['train_scale']))
        probabilities[name] = probability(values)
    summary = {'run_id': spec['run_id'], 'state': 'completed', 'device': spec['device'],
               'dtype': 'float32 features; CPU float64 readouts', 'test_accessed': False,
               'encoder_training_performed': False, 'linear_readouts_fitted': True,
               'validation_used_for_fitting_or_selection': False, 'development_reused_exploratory': True,
               'frozen_fit_sha256': file_digest(out/'frozen-fit.json'),
               'selected_lambdas': {name: frozen['locations'][name]['lambda'] for name in LOCATIONS},
               'metrics': {}, 'versus_full_model': {}, 'encoder_versus_post_head': {}}
    groups = {'all': np.ones(len(y), dtype=bool), 'truncated': mask, 'untruncated': ~mask}
    for name, p in probabilities.items():
        metrics = stratified_metrics(y, p, mask)
        for group, indices in groups.items():
            correct = int(np.sum(p[indices].argmax(1) == y[indices]))
            metrics[group]['accuracy_wilson95'] = wilson(correct, int(indices.sum()))
        summary['metrics'][name] = metrics
    for name in LOCATIONS:
        summary['versus_full_model'][name] = {group: paired_accuracy(y[m], probabilities['full_model'][m].argmax(1),
            probabilities[name][m].argmax(1), **spec['bootstrap']) for group, m in groups.items()}
    summary['encoder_versus_post_head'] = {group: paired_accuracy(y[m], probabilities['post_head'][m].argmax(1),
        probabilities['encoder'][m].argmax(1), **spec['bootstrap']) for group, m in groups.items()}
    write_json(out/'validation-predictions.json', {'split': 'validation', 'rows': [dict(row,
        probabilities={name: p[i].tolist() for name, p in probabilities.items()}) for i, row in enumerate(records)], 'test_accessed': False})
    return summary


def run(spec):
    verify_sources(spec)
    manifest, ids, selected = admit_training(spec)
    out, artifact = ROOT/'reports'/spec['run_id'], ROOT/'artifacts'/spec['run_id']
    if out.exists() or artifact.exists():
        raise FileExistsError('Use a fresh run_id; no overwrite or implicit resume')
    out.mkdir(); artifact.mkdir()
    started = time.monotonic()
    status = {'state': 'admitted', 'started_utc': now(), 'test_accessed': False,
              'validation_body_parsed': False, 'encoder_updates': 0}
    write_json(out/'status.json', status)
    try:
        train_laya.block_network()
        import torch
        import laya.train as upstream
        train_laya.require_device(torch, spec['device'])
        torch.set_num_threads(4)
        torch.manual_seed(20261015)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        protocol = {'spec': spec, 'started_utc': now(), 'versions': {n: version(n) for n in ('torch', 'laya', 'transformers', 'numpy')},
                    'fitting_and_validation_sequence': ['calibration_extract_parity', 'gradient_train_extract', 'fit_and_freeze', 'validation_extract_evaluate'],
                    'source_code_sha256': {}, 'uv_lock_sha256': file_digest(ROOT/'uv.lock'),
                    'primary_location': 'raw encoder last_hidden_state before type_emb and Transformer head',
                    'diagnostic_location': 'scorer input after type_emb and Transformer head',
                    'feature': 'positive marker minus negative marker',
                    'source_bytes_hashed_before_run': True, 'validation_body_parsed_before_fit': False,
                    'excludes_uncertainty': ['training seeds', 'calibration selection', 'development historical reuse', 'distribution shift']}
        reproduction = out/'reproduction'; reproduction.mkdir()
        for name in ('run_laya_frozen_probe.py', 'laya_frozen_probe.py', 'export_laya_posthoc.py',
                     'train_laya.py', 'laya_decision_views.py', 'run_laya_risk_coverage.py'):
            source = ROOT/'scripts'/name
            protocol['source_code_sha256'][f'scripts/{name}'] = file_digest(source)
            shutil.copyfile(source, reproduction/name)
        for name in ('common.py', 'train.py'):
            source = Path(upstream.__file__).parent/name
            protocol['source_code_sha256'][f'laya/{name}'] = file_digest(source)
            shutil.copyfile(source, reproduction/f'upstream-{name}')
        write_json(out/'protocol.json', protocol)
        model, tok, cfg = upstream.load_checkpoint(str(relative(spec['checkpoint'])))
        verify_assets(spec)
        if (cfg['max_len'], cfg['head_max_len']) != (512, 192) or upstream.uses_parallel_layout(cfg) or cfg.get('temperature_by_options'):
            raise ValueError('Unexpected model layout or temperatures')
        model.float().eval().requires_grad_(False).to(spec['device'])
        if next(model.parameters()).device.type != spec['device'] or any(p.requires_grad or not torch.isfinite(p).all().item() for p in model.parameters()):
            raise FloatingPointError('Model must be finite, frozen and on the declared device')
        temperature = float(cfg['temperature'][0])
        cal, cal_records = extract(torch, upstream, model, tok, selected['calibration'], 'calibration', spec, artifact, out)
        check = parity(spec, cal_records, cal, 'calibration', temperature)
        write_json(out/'calibration-parity.json', check)
        if not check['passed']:
            raise RuntimeError('Calibration parity admission failed; development body remains sealed for this run')
        training, _ = extract(torch, upstream, model, tok, selected['gradient_train'], 'gradient_train', spec, artifact, out)
        frozen = fit(torch, training, cal, out)
        frozen_hash = file_digest(out/'frozen-fit.json')
        del training, cal
        status.update(state='fit_frozen', frozen_fit_sha256=frozen_hash, validation_body_parsed=False)
        write_json(out/'status.json', status)
        # First semantic read of development text happens AFTER frozen-fit persistence.
        validation = list(read_jsonl(ROOT/'data/processed/validation.jsonl'))
        if [r['id'] for r in validation] != ids['validation'] or any(type(r['label']) is not int or r['label'] not in (0, 1) for r in validation):
            raise ValueError('Development source IDs/labels differ')
        status.update(state='validation_extracting', validation_body_parsed=True)
        write_json(out/'status.json', status)
        dev, records = extract(torch, upstream, model, tok, validation, 'validation', spec, artifact, out)
        dev_check = parity(spec, records, dev, 'validation', temperature)
        write_json(out/'validation-parity.json', dev_check)
        summary = evaluate(spec, dev, records, frozen, temperature, out)
        summary.update(full_model_cached_parity=dev_check, calibration_parity=check,
                       selected_readouts_frozen_before_validation=True,
                       total_forward_rows=25000, elapsed_seconds=time.monotonic()-started,
                       artifact_bytes=sum(p.stat().st_size for p in artifact.iterdir()),
                       native_temperature=temperature, historical_reference_accuracy=.9684)
        verify_assets(spec)
        if frozen_hash != file_digest(out/'frozen-fit.json') or any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise RuntimeError('Frozen fit or original-model gradient boundary changed')
        for source, sha in protocol['source_code_sha256'].items():
            path = ROOT/source if source.startswith('scripts/') else Path(upstream.__file__).parent/source.split('/')[1]
            if file_digest(path) != sha:
                raise RuntimeError('Source code changed during extraction')
        summary['historical_parity_passed'] = dev_check['passed']
        summary['state'] = 'completed' if dev_check['passed'] else 'completed_with_development_parity_failure'
        write_json(out/'summary.json', summary)
        write_json(out/'verification.json', {'original_weights_unchanged': True,
            'original_model_gradients_absent': True, 'fit_unchanged_after_validation': True,
            'fit_frozen_before_validation_body_parsed': True, 'test_accessed': False,
            'protocol_sha256': file_digest(out/'protocol.json'), 'artifacts_sha256': {
                str(p.relative_to(ROOT)): file_digest(p) for p in artifact.iterdir()}})
        status.update(state=summary['state'], finished_utc=now(), elapsed_seconds=summary['elapsed_seconds'])
        write_json(out/'status.json', status)
        print(json.dumps({'state': summary['state'], 'selected_lambdas': summary['selected_lambdas'],
            'accuracies': {name: m['all']['accuracy'] for name, m in summary['metrics'].items()},
            'elapsed_seconds': summary['elapsed_seconds']}, indent=2), flush=True)
    except Exception as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}', finished_utc=now())
        write_json(out/'status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    spec = validate(json.loads(args.config.read_text()))
    if args.check_only:
        verify_sources(spec)
        admit_training(spec)
        print(json.dumps({'state': 'source_admission_passed', 'model_forward_rows': 0,
                          'validation_body_parsed': False, 'test_accessed': False}))
    else:
        run(spec)


if __name__ == '__main__':
    main()

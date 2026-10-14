"""N8: fixed epoch 1/2 average, calibration-only temperature, paired head-tail study."""
import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import time

import numpy as np

from average_laya_checkpoints import average_checkpoints
from common import ROOT, file_digest, read_jsonl, write_json
from comparison_metrics import fit_temperature, metrics
from export_laya_posthoc import input_digest, prepare_rules, select_splits
from laya_decision_views import compare_views, probabilities, stratified_metrics
from laya_posthoc import _binary_arrays
from laya_runtime import apply_runtime, prepare_runtime
from run_laya_decision_study import check_checkpoint, load_model
import train_laya

SOURCE_RUN = 'laya-decision-window-prefix-s42-strict-20261013-v1'
REFERENCE = 'reports/laya-decision-execution-20261014-v1-final-study'
REFERENCE_FILES = ('protocol.json', 'prefix-trained/calibration-head_tail.json',
                   'prefix-trained/validation-head_tail.json')


def validate_spec(spec):
    keys = {'run_id', 'source_run', 'device', 'runtime_profile', 'batch_size', 'weights',
            'inference_rule', 'temperature_bounds', 'manifest_sha256', 'epoch_report_sha256',
            'epoch_weight_sha256', 'final_weight_sha256', 'reference_sha256'}
    if (set(spec) != keys or spec['source_run'] != SOURCE_RUN or spec['device'] != 'cuda'
            or spec['runtime_profile'] != 'strict' or spec['batch_size'] != 8
            or spec['weights'] != [.5, .5] or spec['inference_rule'] != 'head_tail'
            or spec['temperature_bounds'] != [.25, 4.]):
        raise ValueError('Require approved fixed N8 control/weights/head-tail/temperature protocol')
    run = spec['run_id']
    if not isinstance(run, str) or not run or Path(run).name != run or run in ('.', '..'):
        raise ValueError('Require a new repository run ID')
    if len(spec['epoch_weight_sha256']) != 2 or set(spec['reference_sha256']) != set(REFERENCE_FILES):
        raise ValueError('Predeclare both epoch hashes and all frozen reference files')
    hashes = [spec[k] for k in ('manifest_sha256', 'epoch_report_sha256', 'final_weight_sha256')]
    hashes += spec['epoch_weight_sha256'] + list(spec['reference_sha256'].values())
    if any(not isinstance(v, str) or len(v) != 64 or any(c not in '0123456789abcdef' for c in v) for v in hashes):
        raise ValueError('Require predeclared SHA256 digests')
    return spec


def fit_temperature_only(margins, labels):
    margins, labels = _binary_arrays(margins, labels)
    temperature = fit_temperature(margins, labels, bounds=(.25, 4.))
    return {'temperature': temperature, 'bounds': [.25, 4.], 'bias': 0.,
            'calibration_count': len(labels), 'fit_split': 'original_calibration_only',
            'parameters_frozen_before_validation': True,
            'before': metrics(labels, margins), 'after': metrics(labels, margins, temperature)}


def checked_records(document, rows, split):
    if (document.get('split') != split or document.get('rule') != 'head_tail'
            or document.get('test_accessed') is not False):
        raise ValueError('Require sealed-test head-tail prediction block')
    records = document['rows']
    if [r['sample_id'] for r in records] != [r['id'] for r in rows]:
        raise ValueError('Prediction IDs/order differ from original partition')
    for record, row in zip(records, rows):
        digest = record.get('input_sha256')
        if (record['true_label'] != row['label'] or record.get('review_token_budget') != 466
                or type(record.get('truncated')) is not bool
                or record['truncated'] != (record['review_tokens'] > 466)
                or not isinstance(digest, str) or len(digest) != 64
                or any(c not in '0123456789abcdef' for c in digest)):
            raise ValueError('Prediction labels/input identity/truncation differ')
    logits = np.asarray([r['logits'] for r in records], dtype=float)
    probabilities(logits)
    return records, logits[:, 1]-logits[:, 0]


def export_head_tail(torch, upstream, model, tok, rows, device, batch_size, out, split, temperature=1.):
    _, items, stats, _, _ = prepare_rules(tok, rows)
    records = []
    for start in range(0, len(rows), batch_size):
        prepared = [upstream.encode_item(tok, item, 512, 192, parallel=False)
                    for item in items[start:start+batch_size]]
        batch = upstream.collate_items([prepared], tok.pad_token_id)
        logits = upstream._forward(model, batch, torch.device(device), False, False).cpu().numpy()
        prob = probabilities(logits, temperature)
        for j, value in enumerate(logits):
            i = start+j
            records.append({'sample_id': rows[i]['id'], 'true_label': rows[i]['label'],
                'input_sha256': input_digest(prepared[j]), 'logits': value.tolist(),
                'probabilities': prob[j].tolist(), 'pred_label': int(value.argmax()), **stats[i]})
        print(f'{split}/head_tail: {len(records)}/{len(rows)}', flush=True)
    doc = {'split': split, 'rule': 'head_tail', 'rows': records, 'test_accessed': False}
    write_json(out / f'{split}-head_tail.json', doc)
    return checked_records(doc, rows, split)


def save_calibrated_checkpoint(torch, raw, target, fit, fit_sha256):
    """Keep raw average immutable and save consistent temperature buffer/config separately."""
    from safetensors.torch import load_file, save_file
    if target.exists():
        raise FileExistsError('Use a new calibrated checkpoint directory')
    weights = load_file(raw / 'model.safetensors', device='cpu')
    weights['temperature'] = weights['temperature'].clone()
    weights['temperature'][0] = fit['temperature']
    target.mkdir(parents=True, exist_ok=False)
    for name in ('encoder', 'tokenizer'):
        shutil.copytree(raw / name, target / name)
    cfg = json.loads((raw / 'rl_agent_config.json').read_text())
    cfg['temperature'] = [fit['temperature'], 1., 1.]
    cfg['temperature_by_options'] = {}
    write_json(target / 'rl_agent_config.json', cfg)
    save_file(weights, target / 'model.safetensors')
    write_json(target / 'calibration_manifest.json', {'fit_sha256': fit_sha256,
        'raw_average_sha256': file_digest(raw / 'model.safetensors'),
        'weight_sha256': file_digest(target / 'model.safetensors'),
        'temperature': fit['temperature'], 'inference_rule': 'head_tail',
        'fit_split': 'original_calibration_only', 'training_performed': False, 'test_accessed': False})


def admit(spec, split):
    report = ROOT / 'reports' / SOURCE_RUN
    manifest_path = report / 'manifest.json'
    epoch_path = report / 'epoch_checkpoints.json'
    if file_digest(manifest_path) != spec['manifest_sha256'] or file_digest(epoch_path) != spec['epoch_report_sha256']:
        raise ValueError('Training/epoch manifest changed')
    for filename in ('status.json', 'window_audit.json', 'optimization_diagnostics.json'):
        doc = json.loads((report / filename).read_text())
        if doc['state'] != 'completed':
            raise ValueError('Source training must be completed')
        if filename == 'window_audit.json' and [e['consumed'] for e in doc['epochs']] != [22100, 22100]:
            raise ValueError('Incomplete source exposure')
        if filename == 'optimization_diagnostics.json' and doc['completed_optimizer_updates'] != 5526:
            raise ValueError('Incomplete optimizer budget')
    epochs = json.loads(epoch_path.read_text())['epochs']
    if ([e['epoch'] for e in epochs] != [1, 2]
            or [e['weights_sha256'] for e in epochs] != spec['epoch_weight_sha256']):
        raise ValueError('Epochs differ from predeclared 1/2 pair')
    for file, sha in spec['reference_sha256'].items():
        if file_digest(ROOT / REFERENCE / file) != sha:
            raise ValueError('Frozen control prediction/protocol changed')
    protocol = json.loads((ROOT / REFERENCE / 'protocol.json').read_text())
    reference = next(c for c in protocol['spec']['checkpoints'] if c['name'] == 'prefix-trained')
    if (reference['checkpoint'] != 'checkpoints/'+SOURCE_RUN
            or reference['weight_sha256'] != spec['final_weight_sha256']
            or reference['manifest_sha256'] != spec['manifest_sha256']
            or protocol['runtime']['profile'] != 'strict' or protocol['test_accessed'] is not False):
        raise ValueError('Cached control provenance differs')
    manifest = check_checkpoint(reference, split)
    if (manifest['spec']['window_training'] != {'mode': 'prefix'}
            or manifest['spec']['training']['seed'] != 42
            or manifest['spec']['training']['epochs'] != 2):
        raise ValueError('Require the approved seed42 two-epoch prefix control')
    for i, sha in enumerate(spec['epoch_weight_sha256'], 1):
        if file_digest(ROOT / 'checkpoints' / SOURCE_RUN / f'epoch-{i:03d}/model.safetensors') != sha:
            raise ValueError('Epoch weight changed')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--validate-config', action='store_true')
    args = parser.parse_args()
    spec = validate_spec(json.loads(args.config.read_text()))
    if args.validate_config:
        print(json.dumps({'state': 'config_valid', 'model_accessed': False, 'data_accessed': False}))
        return
    report = ROOT / 'reports' / spec['run_id']
    raw = ROOT / 'checkpoints' / (spec['run_id']+'-uncalibrated')
    calibrated = ROOT / 'checkpoints' / spec['run_id']
    if any(p.exists() for p in (report, raw, calibrated)):
        raise FileExistsError('Require fresh report and both checkpoint directories')
    train_laya.block_network()
    profile = prepare_runtime(dict(spec, training={'amp': False}))
    import torch
    import laya.train as upstream
    if version('laya') != '0.4.1':
        raise ValueError('Require pinned Laya 0.4.1')
    runtime = apply_runtime(torch, profile)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    train_laya.require_device(torch, 'cuda')
    split_path = ROOT / 'reports/split_manifest.json'
    split = json.loads(split_path.read_text())
    for name in ('train', 'validation'):
        if file_digest(ROOT / f'data/processed/{name}.jsonl') != split['jsonl_sha256'][name]:
            raise ValueError('Frozen data hash differs')
    manifest = admit(spec, split)
    train = list(read_jsonl(ROOT / 'data/processed/train.jsonl'))
    placeholders = [{'id': sid, 'label': 0} for sid in split['ids']['validation']]
    calibration = select_splits(manifest, split, train, placeholders)['calibration']
    report.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    status = {'state': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'training_performed': False, 'optimizer_updates': 0, 'test_accessed': False}
    def stage(name):
        status.update(stage=name, elapsed_seconds=time.monotonic()-started)
        write_json(report / 'status.json', status)
        print('STAGE', name, flush=True)
    try:
        stage('averaging')
        write_json(report / 'protocol.json', {'spec': spec, 'config_sha256': file_digest(args.config),
            'scripts_sha256': {name: file_digest(ROOT / 'scripts' / name) for name in (
                'run_laya_checkpoint_average.py', 'average_laya_checkpoints.py',
                'export_laya_posthoc.py', 'comparison_metrics.py')},
            'split_manifest_sha256': file_digest(split_path), 'uv_lock_sha256': file_digest(ROOT / 'uv.lock'),
            'runtime': runtime, 'dtype': 'float32', 'amp': False, 'tf32': False,
            'calibration_count': 400, 'development_count': 2500, 'training_performed': False,
            'optimizer_updates': 0, 'test_accessed': False,
            'versions': {name: version(name) for name in ('torch', 'laya', 'transformers', 'numpy')}})
        source = ROOT / 'checkpoints' / SOURCE_RUN
        average_checkpoints(source / 'epoch-001', source / 'epoch-002', raw, *spec['epoch_weight_sha256'])
        stage('calibration')
        with torch.inference_mode():
            cp = {'checkpoint': str(raw.relative_to(ROOT))}
            model, tok, _ = load_model(torch, upstream, cp, 'cuda')
            _, margins = export_head_tail(torch, upstream, model, tok, calibration, 'cuda', 8, report, 'calibration')
            labels = [r['label'] for r in calibration]
            control_doc = json.loads((ROOT / REFERENCE / REFERENCE_FILES[1]).read_text())
            _, control_margins = checked_records(control_doc, calibration, 'calibration')
            fits = {'average': fit_temperature_only(margins, labels),
                    'control': fit_temperature_only(control_margins, labels)}
            fit_path = report / 'frozen-fit.json'
            write_json(fit_path, fits)
            del model, tok
            gc.collect()
            save_calibrated_checkpoint(torch, raw, calibrated, fits['average'], file_digest(fit_path))
            # Reopen persisted calibrated weights before the first new development forward.
            model, tok, cfg = load_model(torch, upstream, {'checkpoint': str(calibrated.relative_to(ROOT))}, 'cuda')
            if cfg['temperature'][0] != fits['average']['temperature']:
                raise ValueError('Reloaded temperature differs from frozen fit')
            stage('development')
            validation = list(read_jsonl(ROOT / 'data/processed/validation.jsonl'))
            select_splits(manifest, split, train, validation)
            avg_records, _ = export_head_tail(torch, upstream, model, tok, validation, 'cuda', 8,
                report, 'validation', fits['average']['temperature'])
            control_doc = json.loads((ROOT / REFERENCE / REFERENCE_FILES[2]).read_text())
            control_records, _ = checked_records(control_doc, validation, 'validation')
            if [(r['input_sha256'], r['truncated']) for r in avg_records] != [
                    (r['input_sha256'], r['truncated']) for r in control_records]:
                raise ValueError('Average/control input identity or strata differ')
            labels = [r['label'] for r in validation]
            mask = [r['truncated'] for r in avg_records]
            probs = {}
            for name, records in (('average', avg_records), ('control', control_records)):
                probs[name] = probabilities([r['logits'] for r in records], fits[name]['temperature'])
            results = {name: stratified_metrics(labels, p, mask) for name, p in probs.items()}
            paired = compare_views([r['id'] for r in validation], labels, probs['control'], probs['average'], mask)
        for name, sha in spec['reference_sha256'].items():
            if file_digest(ROOT / REFERENCE / name) != sha:
                raise ValueError('Cached reference changed during study')
        for i, sha in enumerate(spec['epoch_weight_sha256'], 1):
            if file_digest(source / f'epoch-{i:03d}/model.safetensors') != sha:
                raise ValueError('Epoch input changed during study')
        summary = {'state': 'completed', 'training_performed': False, 'optimizer_updates': 0,
            'test_accessed': False, 'validation_used_for_fitting': False, 'primary_inference': 'head_tail',
            'averaging_weights': [.5, .5], 'models': results, 'versus_control': paired,
            'delta_accuracy_pp': 100*(results['average']['all']['accuracy']-results['control']['all']['accuracy']),
            'accuracy_improved': results['average']['all']['correct'] > results['control']['all']['correct'],
            'calibrated_checkpoint': str(calibrated.relative_to(ROOT)),
            'weight_sha256': file_digest(calibrated / 'model.safetensors'),
            'frozen_fit_sha256': file_digest(fit_path), 'reference_sha256': spec['reference_sha256'],
            'interpretation': 'One fixed N8 study; no validation selection or automatic deployment.'}
        write_json(report / 'summary.json', summary)
        status.update(state='completed', finished_utc=datetime.now(timezone.utc).isoformat())
        stage('completed')
    except BaseException as exc:
        status.update(state='failed', error=repr(exc))
        stage('failed')
        raise


if __name__ == '__main__':
    main()

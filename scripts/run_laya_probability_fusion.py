"""Replay fixed cached views: fit on calibration, freeze, then evaluate development."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from common import ROOT, digest, file_digest, read_jsonl, write_json
from laya_decision_views import (compare_views, gated_equal_fusion, probabilities,
                                 stratified_metrics)
from laya_probability_fusion import (MIXING_L2, apply_gated_mixture, apply_historical_ols,
                                    fit_historical_ols, fit_mixing_coefficient)

MODELS = ('baseline-s42', 'combined-s42', 'baseline-s43', 'combined-s43')
PRIMARY = ('baseline-s43', 'combined-s42')
RULES = ('head', 'head_tail')
SPLIT_PATH = 'reports/split_manifest.json'
TRAIN_PATH = 'data/processed/train.jsonl'
VALIDATION_PATH = 'data/processed/validation.jsonl'
HISTORY_PATH = 'reports/laya-posthoc-consolidated-v2/summary.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def repo_path(root, value):
    path = Path(value)
    if (path.is_absolute() or '..' in path.parts or not path.parts
            or path.parts[0] not in ('reports', 'data')
            or not (root/path).resolve().is_relative_to(root.resolve())):
        raise ValueError('Input must remain inside the approved repository paths')
    return root/path


def validate_spec(spec):
    if (set(spec) != {'run_id', 'models', 'source_sha256', 'primary_models', 'mixing_l2',
                      'primary_reference', 'minimum_delta_pp', 'maximum_mcnemar_p'}
            or spec['primary_models'] != list(PRIMARY) or spec['mixing_l2'] != MIXING_L2
            or spec['primary_reference'] != 'baseline-s43/head_tail'
            or spec['minimum_delta_pp'] != .10 or spec['maximum_mcnemar_p'] != .05):
        raise ValueError('Require the fixed two-model, one-coefficient protocol')
    run = spec['run_id']
    if not isinstance(run, str) or not run or Path(run).name != run or run in ('.', '..'):
        raise ValueError('Require a fresh run directory name')
    if not isinstance(spec['models'], list) or [m.get('name') for m in spec['models']] != list(MODELS):
        raise ValueError('Declare all four historical models in fixed order')
    required = {SPLIT_PATH, TRAIN_PATH, VALIDATION_PATH, HISTORY_PATH}
    for model in spec['models']:
        if set(model) != {'name', 'export', 'manifest', 'weight_sha256'}:
            raise ValueError('Invalid model identity fields')
        family, seed = model['name'].split('-s')
        if model['export'] != f'reports/laya-posthoc-{family}-s{seed}-fp32-v1':
            raise ValueError('Require the original posthoc exports')
        manifest = Path(model['manifest'])
        if (manifest.is_absolute() or '..' in manifest.parts or len(manifest.parts) != 3
                or manifest.parts[0] != 'reports' or manifest.name != 'manifest.json'):
            raise ValueError('Require a historical training manifest')
        required.update([model['manifest'], model['export']+'/metadata.json'])
        required.update(model['export']+f'/{split}-{rule}.json'
                        for split in ('calibration', 'validation') for rule in RULES)
        _sha(model['weight_sha256'])
    if not isinstance(spec['source_sha256'], dict) or set(spec['source_sha256']) != required:
        raise ValueError('Bind every input file; official test paths are not accepted')
    for digest in spec['source_sha256'].values():
        _sha(digest)
    return spec


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Require a valid SHA256')


def verify_inputs(root, spec):
    for path, digest in spec['source_sha256'].items():
        if file_digest(repo_path(root, path)) != digest:
            raise ValueError(f'Input hash mismatch: {path}')


def load_json(root, spec, path):
    target = repo_path(root, path)
    if file_digest(target) != spec['source_sha256'][path]:
        raise ValueError(f'Input changed before parsing: {path}')
    return json.loads(target.read_text())


def admit_sources(root, spec):
    split = load_json(root, spec, SPLIT_PATH)
    train_ids, dev_ids = split['ids']['train'], split['ids']['validation']
    if (len(train_ids) != 22500 or len(dev_ids) != 2500
            or len(set(train_ids+dev_ids)) != 25000
            or any(not isinstance(s, str) or not s.startswith('train/') for s in train_ids+dev_ids)):
        raise ValueError('Require frozen original training-archive partitions')
    for name, path in (('train', TRAIN_PATH), ('validation', VALIDATION_PATH)):
        if split['jsonl_sha256'][name] != spec['source_sha256'][path]:
            raise ValueError('Data source differs from original split')
    reference = None
    metadata = {}
    for model in spec['models']:
        manifest = load_json(root, spec, model['manifest'])
        meta = load_json(root, spec, model['export']+'/metadata.json')
        training = manifest['spec']['training']
        if ((training['max_len'], training['head_max_len']) != (512, 192)
                or training['amp'] is not False or manifest['spec']['device'] != 'cuda'
                or manifest['spec']['run_id'] != Path(model['manifest']).parent.name
                or training['seed'] != int(model['name'].rsplit('s', 1)[1])):
            raise ValueError('Historical training protocol mismatch')
        if (meta.get('state') != 'completed' or meta.get('test_accessed') is not False
                or meta['checkpoint_sha256'] != model['weight_sha256']
                or meta['manifest_sha256'] != spec['source_sha256'][model['manifest']]
                or meta['split_manifest_sha256'] != spec['source_sha256'][SPLIT_PATH]
                or meta['device'] != 'cuda' or meta['dtype'] != 'float32'
                or meta['amp'] is not False or meta['tf32'] is not False
                or meta['parallel_layout'] is not False or meta['review_token_budget'] != 466
                or meta['head_token_length'] != 45
                or meta['split_counts'] != {'calibration': 400, 'validation': 2500}
                or not np.isfinite(meta['native_temperature']) or meta['native_temperature'] <= 0):
            raise ValueError('Export provenance mismatch')
        for name, ids in (('train', train_ids), ('validation', dev_ids)):
            if (manifest[name]['ids'] != ids
                    or manifest[name]['source_sha256'] != split['jsonl_sha256'][name]
                    or meta['source_sha256'][name] != split['jsonl_sha256'][name]):
                raise ValueError('Manifest partitions differ')
        cal, gradient = manifest['calibration_ids'], manifest['gradient_train_ids']
        if (len(cal) != 400 or len(set(cal)) != 400 or len(gradient) != 22100
                or len(set(gradient)) != 22100 or set(cal) & set(gradient)
                or set(cal) | set(gradient) != set(train_ids)):
            raise ValueError('Calibration and gradient partitions must be fixed and disjoint')
        identity = (cal, gradient, manifest['question'], manifest['model_lock']['sha256']['tokenizer/tokenizer.json'])
        if reference is not None and identity != reference:
            raise ValueError('Models differ in calibration, question or tokenizer')
        reference = identity
        metadata[model['name']] = meta
    return split, reference[0], metadata


def read_source(root, spec, path, ids):
    if file_digest(root/path) != spec['source_sha256'][path]:
        raise ValueError('Source data changed')
    rows = list(read_jsonl(root/path))
    if [r['id'] for r in rows] != ids or any(r['label'] not in (0, 1) for r in rows):
        raise ValueError('Source labels or ordered IDs changed')
    return {r['id']: r for r in rows}


def read_predictions(root, spec, metadata, split_name, ids, source):
    result, reference_rows = {}, None
    for model in spec['models']:
        views = {}
        for rule in RULES:
            path = model['export']+f'/{split_name}-{rule}.json'
            doc = load_json(root, spec, path)
            if doc.get('test_accessed') is not False:
                raise ValueError('Require explicit test-sealed export')
            rows = doc.get('rows', [])
            if (doc.get('split') != split_name or doc.get('rule') != rule
                    or [r.get('sample_id') for r in rows] != ids):
                raise ValueError('Prediction split, rule or ordered IDs differ')
            for row in rows:
                original = source[row['sample_id']]
                z = np.asarray(row.get('logits'), dtype=float)
                if (row.get('true_label') != original['label'] or z.shape != (2,)
                        or not np.isfinite(z).all() or row.get('pred_label') != int(z.argmax())):
                    raise ValueError('Prediction label, logits or argmax differs')
                _sha(row.get('input_sha256'))
                if 'text_sha256' in row and row['text_sha256'] != digest(original['text']):
                    raise ValueError('Prediction text hash differs')
            logits = np.asarray([r['logits'] for r in rows])
            prob = np.asarray([r['probabilities'] for r in rows])
            expected = probabilities(logits, metadata[model['name']]['native_temperature'])
            if prob.shape != expected.shape or not np.allclose(prob, expected, atol=1e-14, rtol=1e-12):
                raise ValueError('Probabilities differ from saved native temperature/logits')
            for row in rows:
                tokens = row.get('review_tokens')
                if (type(tokens) is not int or tokens < 0 or type(row.get('truncated')) is not bool
                        or row.get('review_token_budget') != 466 or row['truncated'] != (tokens > 466)):
                    raise ValueError('Truncation metadata mismatch')
            views[rule] = {'rows': rows, 'prob': prob}
        head, tail = views['head']['rows'], views['head_tail']['rows']
        for a, b in zip(head, tail):
            if any(a[k] != b[k] for k in ('sample_id', 'true_label', 'truncated', 'review_tokens')):
                raise ValueError('Views must have identical sample and length metadata')
            if not a['truncated'] and any(a[k] != b[k] for k in ('input_sha256', 'logits', 'probabilities')):
                raise ValueError('Untruncated views must be exactly identical')
        signature = [[(r['input_sha256'], r['review_tokens'], r['truncated']) for r in views[v]['rows']]
                     for v in RULES]
        if reference_rows is not None and signature != reference_rows:
            raise ValueError('Model inputs differ for the same view')
        reference_rows = signature
        result[model['name']] = views
    return result


def evaluated_views(outputs, labels, mask, frozen):
    views, historical = {}, []
    for name in MODELS:
        for rule in RULES:
            p = outputs[name][rule]['prob']
            views[name+'/'+rule] = p
            historical.append(p)
        views[name+'/gated_equal'] = gated_equal_fusion(
            outputs[name]['head']['prob'], outputs[name]['head_tail']['prob'], mask)
    args = [outputs[name][rule]['prob'] for name in PRIMARY for rule in RULES]
    views['two_model_equal'] = apply_gated_mixture(*args, mask, .5)
    views['two_model_fitted'] = apply_gated_mixture(*args, mask, frozen['mixture']['coefficient'])
    views['historical_eight_equal'] = np.mean(historical, axis=0)
    views['historical_eight_ols'], clip_count = apply_historical_ols(historical, frozen['historical_ols'])
    return views, {name: stratified_metrics(labels, p, mask) for name, p in views.items()}, clip_count


def run_study(spec, root=ROOT):
    spec = validate_spec(spec)
    out = root/'reports'/spec['run_id']
    out.mkdir(exist_ok=False)
    began = time.monotonic()
    state = {'state': 'admitting', 'started_utc': now(), 'training_performed': False,
             'optimizer_updates': 0, 'test_accessed': False, 'runtime': 'cpu_numpy_cached_cuda_fp32'}
    write_json(out/'status.json', state)
    try:
        verify_inputs(root, spec)
        split, cal_ids, metadata = admit_sources(root, spec)
        protocol = {'spec': spec, 'source_sha256': spec['source_sha256'],
                    'source_code_sha256': {f'scripts/{name}': file_digest(ROOT/'scripts'/name)
                        for name in ('run_laya_probability_fusion.py', 'laya_probability_fusion.py',
                                     'laya_decision_views.py', 'comparison_metrics.py', 'common.py')},
                    'uv_lock_sha256': file_digest(ROOT/'uv.lock'), 'numpy_version': np.__version__,
                    'validation_reused_exploratory': True, 'official_test_accessed': False,
                    'source_export_metadata': metadata}
        write_json(out/'protocol.json', protocol)
        source = read_source(root, spec, TRAIN_PATH, split['ids']['train'])
        cal = read_predictions(root, spec, metadata, 'calibration', cal_ids, source)
        cal_rows = cal[PRIMARY[0]]['head']['rows']
        y = np.asarray([r['true_label'] for r in cal_rows])
        mask = np.asarray([r['truncated'] for r in cal_rows])
        gated = [gated_equal_fusion(cal[name]['head']['prob'], cal[name]['head_tail']['prob'], mask)
                 for name in PRIMARY]
        historical = [cal[name][rule]['prob'] for name in MODELS for rule in RULES]
        frozen = {'frozen_utc': now(), 'protocol_sha256': file_digest(out/'protocol.json'),
                  'mixture': fit_mixing_coefficient(*gated, y),
                  'historical_ols': fit_historical_ols(historical, y),
                  'historical_column_order': [n+'/'+r for n in MODELS for r in RULES],
                  'primary_models': list(PRIMARY), 'development_predictions_parsed': False,
                  'validation_used_for_fitting': False, 'parameters_frozen_before_validation': True}
        _, cal_metrics, clips = evaluated_views(cal, y, mask, frozen)
        frozen.update(calibration_metrics=cal_metrics, calibration_ols_clipped=clips)
        write_json(out/'frozen-fit.json', frozen)
        frozen_sha = file_digest(out/'frozen-fit.json')
        # Persisted parameters, rather than live optimizer objects, drive evaluation.
        frozen = json.loads((out/'frozen-fit.json').read_text())
        state.update(state='evaluating', frozen_fit_sha256=frozen_sha,
                     development_parse_started_utc=now())
        write_json(out/'status.json', state)
        dev_ids = split['ids']['validation']
        dev_source = read_source(root, spec, VALIDATION_PATH, dev_ids)
        dev = read_predictions(root, spec, metadata, 'validation', dev_ids, dev_source)
        rows = dev[PRIMARY[0]]['head']['rows']
        labels = np.asarray([r['true_label'] for r in rows])
        truncated = np.asarray([r['truncated'] for r in rows])
        views, scores, clip_count = evaluated_views(dev, labels, truncated, frozen)
        reference = views[spec['primary_reference']]
        comparisons = {name: compare_views(dev_ids, labels, reference, p, truncated)
                       for name, p in views.items() if name != spec['primary_reference']}
        primary_score = scores['two_model_fitted']
        base_score = scores[spec['primary_reference']]
        delta = 100*(primary_score['all']['accuracy']-base_score['all']['accuracy'])
        p_value = comparisons['two_model_fitted']['paired']['mcnemar_exact_two_sided_p']
        success = (delta >= spec['minimum_delta_pp']-1e-12
                   and primary_score['untruncated']['accuracy'] >= base_score['untruncated']['accuracy']
                   and p_value < spec['maximum_mcnemar_p'])
        summary = {'run_id': spec['run_id'], 'metrics': scores, 'versus_fixed_reference': comparisons,
                   'primary_reference': spec['primary_reference'], 'primary_method': 'two_model_fitted',
                   'primary_delta_pp': delta, 'primary_gate_passed': bool(success),
                   'mixture': frozen['mixture'], 'historical_ols_clipped_development': clip_count,
                   'historical_audit': {'claimed_equal_accuracy': .9692, 'claimed_ols_accuracy': .9696,
                       'equal_number_reproduced': scores['historical_eight_equal']['all']['correct'] == 2423,
                       'specified_ols_number_matches': scores['historical_eight_ols']['all']['correct'] == 2424,
                       'original_ols_method_available': False},
                   'frozen_fit_sha256': frozen_sha, 'test_accessed': False,
                   'validation_used_for_fitting': False, 'exploratory_development_only': True,
                   'inference_cost': {'new_model_forward_rows_this_run': 0,
                       'two_model_future_forward_rows': int(2*(len(rows)+truncated.sum())),
                       'eight_view_future_forward_rows': int(4*(len(rows)+truncated.sum())),
                       'latency_measured': False}, 'automatically_deployed': False,
                   'auto_search_or_training': False}
        # Isolate the change from within-model gating to cross-model mixing, too.
        summary['primary_versus_baseline_gated'] = compare_views(dev_ids, labels,
            views['baseline-s43/gated_equal'], views['two_model_fitted'], truncated)
        write_json(out/'validation-fused.json', {'split': 'validation', 'test_accessed': False,
            'rows': [{'sample_id': sid, 'true_label': int(label), 'truncated': bool(trunc),
                      'probabilities': views['two_model_fitted'][i].tolist()}
                     for i, (sid, label, trunc) in enumerate(zip(dev_ids, labels, truncated))]})
        verify_inputs(root, spec)
        if file_digest(out/'frozen-fit.json') != frozen_sha:
            raise ValueError('Frozen parameters changed during evaluation')
        write_json(out/'summary.json', summary)
        state.update(state='completed', completed_utc=now(), elapsed_seconds=time.monotonic()-began)
        write_json(out/'status.json', state)
        return summary
    except Exception as exc:
        state.update(state='failed', failed_utc=now(), error=f'{type(exc).__name__}: {exc}')
        write_json(out/'status.json', state)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--validate-config', action='store_true')
    args = parser.parse_args()
    spec = validate_spec(json.loads(args.config.read_text()))
    if args.validate_config:
        print('Fixed probability-fusion configuration is valid; no predictions parsed.')
    else:
        result = run_study(spec)
        print(json.dumps({k: result[k] for k in ('run_id', 'primary_delta_pp', 'primary_gate_passed', 'mixture')}, indent=2))


if __name__ == '__main__':
    main()

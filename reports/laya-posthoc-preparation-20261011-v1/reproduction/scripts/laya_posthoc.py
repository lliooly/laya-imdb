"""Fit fixed post-hoc calibration on original calibration IDs; evaluate frozen validation.

No model inference, training, network access, validation fitting or test-file reads.
The two input rules must already have been exported with the same saved checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from common import file_digest, read_jsonl, write_json
from comparison_metrics import metrics, paired, sigmoid

TEMPERATURE_BOUNDS = (.25, 4.)
BIAS_BOUNDS = (-3., 3.)
BIAS_L2 = .01
RULES = ('head', 'head_tail')


def _binary_arrays(margins, labels):
    margins = np.asarray(margins, dtype=np.float64)
    labels = np.asarray(labels)
    if margins.ndim != 1 or not len(margins) or labels.shape != margins.shape:
        raise ValueError('Expected nonempty paired binary margins and labels')
    if not np.isfinite(margins).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError('Margins must be finite and labels binary')
    if not np.isfinite(margins * 4).all():
        raise ValueError('Margins exceed safe calibration range')
    return margins, labels.astype(int)


def _bounded_root(derivative, lo, hi):
    if derivative(lo) >= 0:
        return lo
    if derivative(hi) <= 0:
        return hi
    for _ in range(80):
        mid = (lo + hi) / 2
        if derivative(mid) > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def fit_temperature_bias(margins, labels):
    """Global deterministic convex fit in inverse temperature and bias.

    Objective: mean calibration NLL + .01 * bias**2. Profile out bias using
    its monotone derivative, then bisect the convex profile's derivative.
    Validation data is deliberately absent from this API.
    """
    margins, labels = _binary_arrays(margins, labels)

    def best_bias(inverse_temperature):
        def derivative(bias):
            return float(np.mean(sigmoid(inverse_temperature * margins + bias) - labels)
                         + 2 * BIAS_L2 * bias)
        return _bounded_root(derivative, *BIAS_BOUNDS)

    def derivative(inverse_temperature):
        bias = best_bias(inverse_temperature)
        return float(np.mean((sigmoid(inverse_temperature * margins + bias) - labels) * margins))

    inverse = _bounded_root(derivative, 1 / TEMPERATURE_BOUNDS[1], 1 / TEMPERATURE_BOUNDS[0])
    bias = best_bias(inverse)
    temperature = 1 / inverse
    calibrated = margins / temperature + bias
    nll = float(np.mean(np.logaddexp(0., np.where(labels == 1, -calibrated, calibrated))))
    return {'temperature': float(temperature), 'bias': float(bias),
            'temperature_bounds': list(TEMPERATURE_BOUNDS), 'bias_bounds': list(BIAS_BOUNDS),
            'bias_l2': BIAS_L2, 'calibration_count': len(labels), 'calibration_nll': nll,
            'penalized_objective': nll + BIAS_L2 * bias * bias,
            'method': 'convex_profile_derivative_bisection_80_steps',
            'fit_split': 'original_calibration_only', 'parameters_frozen_before_validation': True}


def apply_calibration(margins, parameters):
    temperature, bias = parameters['temperature'], parameters.get('bias', 0.)
    if not np.isfinite([temperature, bias]).all() or temperature <= 0:
        raise ValueError('Calibration requires positive finite temperature and finite bias')
    margins = np.asarray(margins, dtype=np.float64)
    calibrated = margins / temperature + bias
    if margins.ndim != 1 or not np.isfinite(calibrated).all():
        raise ValueError('Invalid calibrated margins')
    return calibrated


def _unique_ids(ids, name):
    if not isinstance(ids, list) or not all(isinstance(v, str) and v.startswith('train/') for v in ids):
        raise ValueError(f'{name} must contain original training-archive IDs only')
    if len(ids) != len(set(ids)):
        raise ValueError(f'{name} contains duplicate IDs')
    return set(ids)


def _digest(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'Invalid SHA256: {name}')


def _prediction_block(path, split, rule, ids, source_rows):
    doc = json.loads(Path(path).read_text())
    if doc.get('split') != split or doc.get('rule') != rule:
        raise ValueError('Prediction block split/rule mismatch')
    rows = doc.get('rows', [])
    if [r.get('sample_id') for r in rows] != ids:
        raise ValueError(f'{split} predictions must match the original ordered IDs exactly')
    logits = []
    labels = []
    for row in rows:
        original = source_rows[row['sample_id']]
        if row.get('true_label') != original['label'] or row.get('true_label') not in (0, 1):
            raise ValueError('Prediction label differs from frozen source')
        _digest(row.get('input_sha256'), 'input_sha256')
        if 'text_sha256' in row:
            if row['text_sha256'] != hashlib.sha256(original['text'].encode()).hexdigest():
                raise ValueError('Prediction text hash differs from frozen source')
        z = np.asarray(row.get('logits'), dtype=float)
        if z.shape != (2,) or not np.isfinite(z).all():
            raise ValueError('Prediction logits must contain two finite values')
        if 'probabilities' in row:
            p = np.asarray(row['probabilities'], dtype=float)
            if p.shape != (2,) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1) or not np.isclose(p.sum(), 1):
                raise ValueError('Invalid saved probabilities')
        if 'pred_label' in row and row['pred_label'] != int(z.argmax()):
            raise ValueError('Saved prediction does not match logits')
        labels.append(original['label'])
        logits.append(z)
    logits = np.asarray(logits)
    margins = logits[:, 1] - logits[:, 0]
    margins, labels = _binary_arrays(margins, labels)
    return margins, labels


def _transitions(ids, labels, before, after):
    before_correct = (np.asarray(before) > 0) == labels
    after_correct = (np.asarray(after) > 0) == labels
    corrected = [sid for sid, keep in zip(ids, ~before_correct & after_correct) if keep]
    regressed = [sid for sid, keep in zip(ids, before_correct & ~after_correct) if keep]
    return {'corrected_ids': corrected, 'regressed_ids': regressed,
            'corrected': len(corrected), 'regressed': len(regressed),
            'paired': paired(labels, before, after)}


def analyze_export(export_dir, manifest_path, split_manifest_path, train_path, validation_path):
    """Validate provenance and frozen 400/2500 splits, then independently fit both rules."""
    export_dir = Path(export_dir)
    manifest = json.loads(Path(manifest_path).read_text())
    split = json.loads(Path(split_manifest_path).read_text())
    metadata_path = export_dir / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('schema_version') != 1 or metadata.get('state') != 'completed' or metadata.get('test_accessed') is not False:
        raise ValueError('Export must be completed schema 1 with sealed test')
    if metadata.get('manifest_sha256') != file_digest(manifest_path):
        raise ValueError('Training manifest hash mismatch')
    _digest(metadata.get('checkpoint_sha256'), 'checkpoint_sha256')
    native_temperature = metadata.get('native_temperature')
    if not isinstance(native_temperature, (int, float)) or not np.isfinite(native_temperature) or native_temperature <= 0:
        raise ValueError('Invalid original checkpoint temperature')
    source_hashes = {'train': file_digest(train_path), 'validation': file_digest(validation_path)}
    if metadata.get('source_sha256') != source_hashes:
        raise ValueError('Export source hash mismatch')
    for name, digest in source_hashes.items():
        if digest != split['jsonl_sha256'][name] or digest != manifest[name]['source_sha256']:
            raise ValueError(f'{name} frozen source hash mismatch')
    cal_ids = manifest['calibration_ids']
    gradient_ids = manifest['gradient_train_ids']
    val_ids = manifest['validation']['ids']
    train_ids = manifest['train']['ids']
    sets = {name: _unique_ids(ids, name) for name, ids in (
        ('calibration', cal_ids), ('gradient_train', gradient_ids), ('validation', val_ids), ('train', train_ids))}
    if len(cal_ids) != 400 or len(val_ids) != 2500:
        raise ValueError('Require original 400 calibration and full 2500 validation IDs')
    if (sets['calibration'] & sets['gradient_train'] or sets['calibration'] & sets['validation']
            or sets['gradient_train'] & sets['validation']):
        raise ValueError('Calibration, gradient training and validation must be disjoint')
    if sets['calibration'] | sets['gradient_train'] != sets['train']:
        raise ValueError('Original train partition differs from calibration plus gradient IDs')
    if train_ids != split['ids']['train'] or val_ids != split['ids']['validation']:
        raise ValueError('Manifest IDs differ from frozen split')
    sources = {}
    for path, expected_ids in ((train_path, train_ids), (validation_path, val_ids)):
        rows = list(read_jsonl(path))
        if [r['id'] for r in rows] != expected_ids:
            raise ValueError('Source ordered IDs differ from frozen manifest')
        sources.update({r['id']: r for r in rows})
    report = {'state': 'completed', 'test_accessed': False, 'training_performed': False,
              'inference_performed': False, 'validation_used_for_fitting': False,
              'calibration_ids': cal_ids, 'validation_ids': val_ids,
              'checkpoint_sha256': metadata['checkpoint_sha256'],
              'native_temperature': native_temperature, 'source_sha256': source_hashes,
              'manifest_sha256': file_digest(manifest_path),
              'split_manifest_sha256': file_digest(split_manifest_path),
              'export_sha256': {'metadata.json': file_digest(metadata_path)}, 'rules': {}}
    validation_margins = {}
    labels = None
    for rule in RULES:
        cal_path = export_dir / f'calibration-{rule}.json'
        val_path = export_dir / f'validation-{rule}.json'
        cal_margin, cal_labels = _prediction_block(cal_path, 'calibration', rule, cal_ids, sources)
        val_margin, labels = _prediction_block(val_path, 'validation', rule, val_ids, sources)
        parameters = fit_temperature_bias(cal_margin, cal_labels)
        fitted = apply_calibration(val_margin, parameters)
        native = apply_calibration(val_margin, {'temperature': native_temperature})
        validation_margins[rule] = {'raw': val_margin, 'native_temperature': native, 'temperature_bias': fitted}
        report['export_sha256'][cal_path.name] = file_digest(cal_path)
        report['export_sha256'][val_path.name] = file_digest(val_path)
        report['rules'][rule] = {'fit': parameters,
            'calibration_metrics': metrics(cal_labels, apply_calibration(cal_margin, parameters)),
            'validation_metrics': {name: metrics(labels, margin) for name, margin in validation_margins[rule].items()},
            'temperature_bias_vs_raw': _transitions(val_ids, labels, val_margin, fitted),
            'native_temperature_scope': 'original checkpoint temperature; not refit for changed input rule'}
    report['head_tail_vs_head'] = {
        name: _transitions(val_ids, labels, validation_margins['head'][name], validation_margins['head_tail'][name])
        for name in ('raw', 'native_temperature', 'temperature_bias')}
    report['interpretation'] = 'Development validation evaluation; fixed fit protocol, no validation-based parameter selection.'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ('export-dir', 'manifest', 'split-manifest', 'train-source', 'validation-source', 'out'):
        parser.add_argument('--' + option, type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Choose a new report path; historical reports are immutable')
    report = analyze_export(args.export_dir, args.manifest, args.split_manifest, args.train_source, args.validation_source)
    write_json(args.out, report)


if __name__ == '__main__':
    main()

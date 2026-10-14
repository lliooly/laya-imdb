"""One-shot official-test evaluation of the already selected, frozen Laya candidate."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
import train_laya
from common import ROOT, digest, file_digest, read_jsonl, write_json
from export_laya_posthoc import select_splits
from laya_fusion_inference import MODEL_NAMES, FrozenFusionPredictor
from laya_decision_views import probability_margin, stratified_metrics, compare_views
from validate_laya_fusion_inference import compare_trace, compare_fused, relative_path, verify_inputs


def now():
    return datetime.now(timezone.utc).isoformat()


def wilson(correct, count):
    if not 0 <= correct <= count or count <= 0:
        raise ValueError('Invalid binomial counts')
    z = 1.959963984540054
    p, d = correct/count, 1+z*z/count
    center = (p+z*z/(2*count))/d
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/d
    return {'lower': center-half, 'upper': center+half,
            'method': '95% Wilson binomial interval; does not quantify training/selection uncertainty'}


def validate_rows(rows, expected_ids, count):
    if (len(rows) != count or [r['id'] for r in rows] != expected_ids
            or len(set(expected_ids)) != count
            or Counter(r['label'] for r in rows) != {0: count//2, 1: count//2}
            or any(not isinstance(r['text'], str) or not r['text'].strip()
                   or digest(r['text']) != r['text_sha256'] for r in rows)):
        raise ValueError('Official test rows differ from the sealed dataset')


class ProgressInference:
    """Preserve each view's original global batch8 grouping while reporting progress."""
    def _infer(self, name, items, indices, stats, ids, reuse=None):
        output = [dict(r, reused_head_prediction=True) for r in reuse] if reuse is not None else [None]*len(items)
        for start in range(0, len(indices), 256):
            selected = indices[start:start+256]  # 256 is a multiple of the frozen batch size 8.
            block = super()._infer(name, items, selected, stats, ids)
            for i in selected:
                output[i] = block[i]
            self.forward_rows_completed += len(selected)
            self.progress(self.forward_rows_completed)
        return output


def run(freeze_path, backend):
    freeze = json.loads(freeze_path.read_text())
    if (freeze['schema_version'] != 1 or freeze['primary'] != 'two_model_fitted'
            or freeze['reference'] != 'baseline-s43/head_tail'
            or freeze['target_accuracy'] != .98 or freeze['test']['count'] != 25000
            or backend not in freeze['backends']):
        raise ValueError('Frozen evaluation protocol differs')
    for rel, sha in freeze['source_sha256'].items():
        if file_digest(relative_path(ROOT, rel)) != sha:
            raise ValueError('Frozen source changed: '+rel)
    config_path = relative_path(ROOT, freeze['backends'][backend]['config'])
    if file_digest(config_path) != freeze['backends'][backend]['sha256']:
        raise ValueError('Frozen inference config changed')
    spec = json.loads(config_path.read_text())
    out = ROOT/'reports'/freeze['run_id']
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    status = {'state': 'admitting', 'started_utc': now(), 'test_accessed': False,
              'training_performed': False, 'parameters_refitted': False, 'backend': backend,
              'freeze_sha256': file_digest(freeze_path)}
    write_json(out/'status.json', status)
    try:
        verify_inputs(ROOT, spec)
        if file_digest(ROOT/freeze['split_manifest']) != freeze['split_manifest_sha256']:
            raise ValueError('Split manifest changed')
        split = json.loads((ROOT/freeze['split_manifest']).read_text())
        if split['jsonl_sha256']['test'] != freeze['test']['sha256']:
            raise ValueError('Sealed test identity changed')
        frozen = json.loads((ROOT/spec['frozen_fit']).read_text())
        if frozen['mixture']['coefficient'] != freeze['coefficient']:
            raise ValueError('Frozen coefficient changed')
        train_laya.block_network()
        if backend == 'cuda':
            import torch
            train_laya.require_device(torch, 'cuda')
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            base = FrozenFusionPredictor
            hardware = {'platform': platform.platform(), 'gpu': torch.cuda.get_device_name(),
                        'dtype': 'float32', 'amp': False, 'tf32': False,
                        'versions': {n: version(n) for n in ('torch','laya','transformers','numpy')}}
        else:
            from laya_mlx_precise_inference import PreciseMLXFusionPredictor
            base = PreciseMLXFusionPredictor
            hardware = {'platform': platform.platform(), 'dtype': 'float32', 'mlx_enable_tf32': '0',
                        'versions': {n: version(n) for n in ('mlx','laya-mlx','laya','numpy')}}
        engine_class = type('FrozenTestPredictor', (ProgressInference, base), {})
        engine = engine_class(ROOT, spec['checkpoint_root'], spec['models'], frozen, spec['device'], 8)
        engine.forward_rows_completed = 0
        def progress(count):
            write_json(out/'progress.json', {'phase': status['state'], 'forward_rows_completed': count,
                'elapsed_seconds': time.monotonic()-started, 'updated_utc': now()})
        engine.progress = progress
        write_json(out/'protocol.json', {'freeze': freeze, 'freeze_sha256': file_digest(freeze_path),
            'backend': backend, 'hardware': hardware, 'source_config': spec,
            'primary_fixed_before_test': True, 'test_used_for_selection': False,
            'test_used_for_hyperparameters_or_fitting': False})
        write_json(out/'frozen-candidate.json', freeze)
        for name in MODEL_NAMES:
            engine.load(name)
        # Recheck all original 400 calibration inputs before opening the test body.
        from run_laya_probability_fusion import (validate_spec, admit_sources, load_json,
            read_predictions, evaluated_views)
        original = validate_spec(json.loads((ROOT/spec['fusion_spec']).read_text()))
        split, cal_ids, metadata = admit_sources(ROOT, original)
        manifest = load_json(ROOT, original, next(m['manifest'] for m in original['models'] if m['name']==MODEL_NAMES[0]))
        selected = select_splits(manifest, split, list(read_jsonl(ROOT/'data/processed/train.jsonl')),
                                 list(read_jsonl(ROOT/'data/processed/validation.jsonl')))
        rows = selected['calibration']
        status.update(state='preflight_calibration');write_json(out/'status.json', status)
        actual = engine.predict([r['text'] for r in rows], cal_ids)
        cached = read_predictions(ROOT, original, metadata, 'calibration', cal_ids, {r['id']:r for r in rows})
        checks = {n+'/'+v:compare_trace(actual['models'][n][v], cached[n][v]['rows'], freeze['preflight_tolerance'])
                  for n in MODEL_NAMES for v in ('head','head_tail')}
        expected, _, _ = evaluated_views(cached, np.array([r['label'] for r in rows]),
            np.array([r['truncated'] for r in actual['stats']]), frozen)
        checks['primary'] = compare_fused(cal_ids, actual['primary'], expected['two_model_fitted'], freeze['preflight_tolerance'])
        write_json(out/'preflight.json', checks)
        if not all(c['passed'] for c in checks.values()):
            raise RuntimeError('Calibration preflight failed; official test remains sealed')
        print('Calibration preflight passed; opening frozen official test', flush=True)
        # The primary, reference, backend, code and metrics are sealed before this read.
        status.update(state='official_test_inference', test_accessed=True, test_opened_utc=now())
        write_json(out/'status.json', status)
        test_path = ROOT/freeze['test']['path']
        if file_digest(test_path) != freeze['test']['sha256']:
            raise ValueError('Official test SHA256 mismatch')
        rows = list(read_jsonl(test_path))
        validate_rows(rows, split['ids']['test'], freeze['test']['count'])
        ids, labels = [r['id'] for r in rows], np.array([r['label'] for r in rows])
        # Only label-free texts/IDs enter prediction. Both scored models share these forwards.
        result = engine.predict([r['text'] for r in rows], ids)
        result.pop('equal')  # Historical auxiliary never becomes a test-selected candidate.
        write_json(out/'test-actual.json', result)
        mask = np.array([s['truncated'] for s in result['stats']])
        scores = {name:stratified_metrics(labels, result[key], mask)
                  for name,key in [('primary','primary'),('reference','baseline_head_tail')]}
        comparison = compare_views(ids, labels, result['baseline_head_tail'], result['primary'], mask)
        accuracy = scores['primary']['all']['accuracy']
        audit = json.loads((ROOT/'reports/data_audit.json').read_text())
        summary = {'run_id': freeze['run_id'], 'state':'completed', 'split':'official_test', 'count':len(rows),
            'freeze_sha256':file_digest(freeze_path), 'backend':backend, 'hardware':hardware,
            'metrics':scores, 'paired_comparison':comparison,
            'primary_accuracy_wilson_95':wilson(scores['primary']['all']['correct'],len(rows)),
            'target':{'accuracy':.98,'observed_primary_accuracy':accuracy,'reached':accuracy>=.98,
                      'correct_needed':24500,'primary_shortfall_count':max(0,24500-scores['primary']['all']['correct'])},
            'primary_fixed_before_test':True,'test_accessed':True,'test_used_for_selection':False,
            'training_performed':False,'parameters_refitted':False,'default_deployment_changed':False,
            'modernbert_training_started':False,
            'official_train_test_overlap_unique_texts_from_existing_audit':audit['official_train_test_overlap_unique_texts'],
            'forward_rows':result['forward_rows'],'elapsed_seconds':time.monotonic()-started}
        engine.verify_assets();verify_inputs(ROOT,spec)
        if file_digest(test_path)!=freeze['test']['sha256'] or file_digest(freeze_path)!=status['freeze_sha256']:
            raise ValueError('Frozen inputs changed during evaluation')
        for rel,sha in freeze['source_sha256'].items():
            if file_digest(ROOT/rel)!=sha:raise ValueError('Source changed during evaluation')
        write_json(out/'summary.json',summary)
        status.update(state='completed',completed_utc=now(),elapsed_seconds=time.monotonic()-started,
                      primary_accuracy=accuracy,target_reached=accuracy>=.98)
        write_json(out/'status.json',status)
        print(json.dumps({'primary_accuracy':accuracy,'reference_accuracy':scores['reference']['all']['accuracy'],
                          'target_reached':accuracy>=.98,'elapsed_seconds':status['elapsed_seconds']}),flush=True)
        return summary
    except Exception as exc:
        status.update(state='failed',error=f'{type(exc).__name__}: {exc}',failed_utc=now())
        write_json(out/'status.json',status)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',type=Path,required=True)
    parser.add_argument('--backend',choices=('cuda','mlx-fp32-precise'),required=True)
    args=parser.parse_args()
    run(args.freeze,args.backend)

if __name__=='__main__':main()

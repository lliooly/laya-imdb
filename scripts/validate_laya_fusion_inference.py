"""Verify frozen fusion against real FP32 forwards and measure inference cost."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import threading
import time

import numpy as np

import train_laya
from common import ROOT, file_digest, read_jsonl, write_json
from export_laya_posthoc import select_splits
from laya_decision_views import stratified_metrics
from laya_fusion_inference import (MODEL_NAMES, FrozenFusionPredictor, empty_cache, synchronize)


def now():
    return datetime.now(timezone.utc).isoformat()


def relative_path(root, value):
    path = Path(value)
    if path.is_absolute() or '..' in path.parts or not (root/path).resolve().is_relative_to(root.resolve()):
        raise ValueError('Require a repository-relative path')
    return root/path


def validate_config(spec):
    if (set(spec) != {'run_id', 'device', 'batch_size', 'checkpoint_root', 'models',
                     'fusion_spec', 'frozen_fit', 'cached_summary', 'cached_fused',
                     'input_sha256', 'tolerance', 'benchmark'}
            or spec['device'] not in ('cuda', 'mps') or spec['batch_size'] != 8
            or spec['tolerance'] != {'logits_absolute': .001, 'probabilities_absolute': .0001}
            or spec['benchmark'] != {'sample_count': 128, 'warmup': 2, 'repetitions': 5}
            or list(spec['models']) != list(MODEL_NAMES)):
        raise ValueError('Require fixed inference-only protocol, tolerances and benchmark')
    run = spec['run_id']
    if not isinstance(run, str) or not run or Path(run).name != run or run in ('.', '..'):
        raise ValueError('Require a new run ID')
    for field in ('checkpoint_root', 'fusion_spec', 'frozen_fit', 'cached_summary', 'cached_fused'):
        relative_path(ROOT, spec[field])
    for path, sha in spec['input_sha256'].items():
        relative_path(ROOT, path)
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Require input SHA256')
    for field in ('fusion_spec', 'frozen_fit', 'cached_summary', 'cached_fused'):
        if spec[field] not in spec['input_sha256']:
            raise ValueError('Bind frozen experiment files')
    return spec


def verify_inputs(root, spec):
    for rel, digest in spec['input_sha256'].items():
        if file_digest(relative_path(root, rel)) != digest:
            raise ValueError(f'Input SHA256 mismatch: {rel}')


class MemorySampler:
    """CUDA allocator peak counters; MPS sampled usage with explicit limitations."""
    def __init__(self, torch, device):
        self.torch, self.device = torch, device
        self.samples, self.stop_event, self.worker = [], threading.Event(), None

    def sample(self):
        if self.device.type == 'cuda':
            value = {'active_bytes': self.torch.cuda.memory_allocated(self.device),
                     'driver_or_reserved_bytes': self.torch.cuda.memory_reserved(self.device)}
        else:
            value = {'active_bytes': self.torch.mps.current_allocated_memory(),
                     'driver_or_reserved_bytes': self.torch.mps.driver_allocated_memory()}
        self.samples.append({'monotonic': time.monotonic(), **value})

    def start(self):
        if self.device.type == 'cuda':
            self.torch.cuda.reset_peak_memory_stats(self.device)
        self.sample()
        if self.device.type == 'mps':
            def poll():
                while not self.stop_event.wait(.02):
                    self.sample()
            self.worker = threading.Thread(target=poll, daemon=True)
            self.worker.start()

    def finish(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join()
        self.sample()
        result = {'samples': self.samples, 'sampled_peak_active_bytes': max(x['active_bytes'] for x in self.samples),
                  'sampled_peak_driver_or_reserved_bytes': max(x['driver_or_reserved_bytes'] for x in self.samples)}
        if self.device.type == 'cuda':
            result.update(peak_allocated_bytes=self.torch.cuda.max_memory_allocated(self.device),
                          peak_reserved_bytes=self.torch.cuda.max_memory_reserved(self.device),
                          method='CUDA allocator peak counters; includes resident model weights')
        else:
            result['method'] = 'MPS 20ms usage sampling; sampled maximum, not an exact hardware peak'
        return result


def benchmark(predictor, texts, mode, spec):
    torch, device = predictor.torch, predictor.device
    empty_cache(torch, device)
    for _ in range(spec['warmup']):
        predictor.predict(texts, mode=mode)
        synchronize(torch, device)
    # Keep warmed kernels, clear cached allocations before measuring memory.
    empty_cache(torch, device)
    durations, forward_rows = [], []
    sampler = MemorySampler(torch, device)
    sampler.start()
    try:
        for _ in range(spec['repetitions']):
            synchronize(torch, device)
            start = time.perf_counter()
            output = predictor.predict(texts, mode=mode)
            synchronize(torch, device)
            durations.append(time.perf_counter()-start)
            forward_rows.append(output['forward_rows'])
    finally:
        memory = sampler.finish()
    median = statistics.median(durations)
    return {'mode': mode, 'review_count': len(texts), 'seconds': durations,
            'median_seconds': median, 'reviews_per_second': len(texts)/median,
            'amortized_milliseconds_per_review': median/len(texts)*1000,
            'forward_rows_per_repetition': forward_rows, 'memory': memory,
            'truncated_count': sum(s['truncated'] for s in output['stats']),
            'review_token_counts': [s['review_tokens'] for s in output['stats']],
            'scope': 'tokenization, preparation, transfers, forwards and fusion; excludes model loading',
            'is_single_request_latency': False}


def compare_trace(actual, cached, tolerance):
    if len(actual) != len(cached):
        raise ValueError('Actual and cached prediction counts differ')
    logits_max = probability_max = 0.
    label_mismatches, out_of_tolerance = [], []
    for a, b in zip(actual, cached):
        if any(a.get(k) != b.get(k) for k in ('sample_id', 'input_sha256', 'review_tokens',
                                             'review_token_budget', 'truncated')):
            raise ValueError('Actual inputs differ from the frozen cached inputs')
        za, zb = np.asarray(a['logits']), np.asarray(b['logits'])
        pa, pb = np.asarray(a['probabilities']), np.asarray(b['probabilities'])
        if za.shape != (2,) or pa.shape != (2,) or not np.isfinite([za, zb, pa, pb]).all():
            raise FloatingPointError('Invalid comparison values')
        dz, dp = float(np.max(np.abs(za-zb))), float(np.max(np.abs(pa-pb)))
        logits_max, probability_max = max(logits_max, dz), max(probability_max, dp)
        if int(za.argmax()) != int(zb.argmax()):
            label_mismatches.append(a['sample_id'])
        if dz > tolerance['logits_absolute'] or dp > tolerance['probabilities_absolute']:
            out_of_tolerance.append(a['sample_id'])
    return {'max_abs_logits_delta': logits_max, 'max_abs_probability_delta': probability_max,
            'label_mismatches': label_mismatches, 'out_of_tolerance_ids': out_of_tolerance,
            'passed': not label_mismatches and not out_of_tolerance}


def compare_fused(ids, actual, expected, tolerance):
    a, b = np.asarray(actual), np.asarray(expected)
    if a.shape != b.shape or a.shape != (len(ids), 2) or not np.isfinite([a, b]).all():
        raise ValueError('Invalid fused comparison arrays')
    mismatch = a.argmax(1) != b.argmax(1)
    delta = np.max(np.abs(a-b), axis=1)
    return {'max_abs_probability_delta': float(delta.max()),
            'label_mismatches': [s for s, bad in zip(ids, mismatch) if bad],
            'out_of_tolerance_ids': [s for s, d in zip(ids, delta) if d > tolerance['probabilities_absolute']],
            'passed': bool(not mismatch.any() and delta.max() <= tolerance['probabilities_absolute'])}


def run(spec, root=ROOT):
    from run_laya_probability_fusion import (PRIMARY, admit_sources, evaluated_views,
        load_json, read_predictions, validate_spec, verify_inputs as verify_fusion_inputs)
    spec = validate_config(spec)
    out = root/'reports'/spec['run_id']
    out.mkdir(exist_ok=False)
    started = time.monotonic()
    status = {'state': 'admitting', 'started_utc': now(), 'test_accessed': False,
              'training_performed': False, 'parameters_refitted': False, 'device': spec['device']}
    write_json(out/'status.json', status)
    try:
        verify_inputs(root, spec)
        original = validate_spec(json.loads((root/spec['fusion_spec']).read_text()))
        if not set(original['source_sha256']) <= set(spec['input_sha256']):
            raise ValueError('Bind all original fusion sources')
        if any(spec['input_sha256'][p] != d for p, d in original['source_sha256'].items()):
            raise ValueError('Fusion input identities changed')
        verify_fusion_inputs(root, original)
        split, cal_ids, metadata = admit_sources(root, original)
        manifest = load_json(root, original, next(m['manifest'] for m in original['models'] if m['name'] == PRIMARY[0]))
        selected = select_splits(manifest, split, list(read_jsonl(root/'data/processed/train.jsonl')),
                                 list(read_jsonl(root/'data/processed/validation.jsonl')))
        frozen = json.loads((root/spec['frozen_fit']).read_text())
        cached_summary = json.loads((root/spec['cached_summary']).read_text())
        if file_digest(root/spec['frozen_fit']) != cached_summary['frozen_fit_sha256']:
            raise ValueError('Frozen mixture differs from the completed experiment')
        prior_protocol = str(Path(spec['frozen_fit']).parent/'protocol.json')
        if (prior_protocol not in spec['input_sha256']
                or spec['input_sha256'][prior_protocol] != frozen['protocol_sha256']):
            raise ValueError('Frozen original protocol differs')
        for name in MODEL_NAMES:
            if spec['models'][name]['sha256']['model.safetensors'] != metadata[name]['checkpoint_sha256']:
                raise ValueError('Actual weights differ from cached model identity')
        train_laya.block_network()
        import torch
        train_laya.require_device(torch, spec['device'])
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        predictor = FrozenFusionPredictor(root, spec['checkpoint_root'], spec['models'], frozen,
                                          spec['device'], spec['batch_size'])
        hardware = {'platform': platform.platform(), 'processor': platform.processor(),
                    'device': spec['device'], 'dtype': 'float32', 'amp': False, 'tf32': False,
                    'versions': {n: version(n) for n in ('torch', 'laya', 'transformers', 'numpy')}}
        if spec['device'] == 'cuda':
            hardware.update(gpu=torch.cuda.get_device_name(), total_gpu_bytes=torch.cuda.get_device_properties(0).total_memory)
        else:
            hardware['recommended_mps_bytes'] = torch.mps.recommended_max_memory()
        protocol = {'spec': spec, 'hardware': hardware, 'frozen_coefficient': predictor.coefficient,
                    'source_sha256': {p: file_digest(root/p) for p in
                        ('scripts/validate_laya_fusion_inference.py', 'scripts/laya_fusion_inference.py',
                         'scripts/laya_probability_fusion.py', 'scripts/export_laya_posthoc.py',
                         'scripts/train_laya.py', 'scripts/run_laya_probability_fusion.py',
                         'scripts/laya_decision_views.py', 'scripts/common.py', 'scripts/comparison_metrics.py', 'uv.lock')},
                    'official_test_accessed': False, 'parameters_refitted': False}
        write_json(out/'protocol.json', protocol)
        predictor.load(MODEL_NAMES[0])
        texts = [r['text'] for r in selected['calibration'][:spec['benchmark']['sample_count']]]
        status.update(state='benchmarking_single');write_json(out/'status.json', status)
        single = benchmark(predictor, texts, 'baseline_head_tail', spec['benchmark'])
        print('Single-model benchmark completed', flush=True)
        predictor.load(MODEL_NAMES[1])
        if any(predictor.configs[n]['temperature'][0] != metadata[n]['native_temperature'] for n in MODEL_NAMES):
            raise ValueError('Saved native temperature differs from cached export')
        status.update(state='benchmarking_fusion');write_json(out/'status.json', status)
        fused_benchmark = benchmark(predictor, texts, 'fusion', spec['benchmark'])
        benchmark_report = {'single': single, 'fusion': fused_benchmark,
                            'fusion_to_single_median_time_ratio': fused_benchmark['median_seconds']/single['median_seconds'],
                            'load_seconds': predictor.load_seconds,
                            'sample_ids': cal_ids[:len(texts)], 'warmup': spec['benchmark']['warmup'],
                            'repetitions': spec['benchmark']['repetitions'], 'hardware': hardware}
        write_json(out/'benchmark.json', benchmark_report)
        print('Fusion benchmark completed', flush=True)
        comparisons, all_scores = {}, {}
        all_passed = True
        for split_name in ('calibration', 'validation'):
            status.update(state='inferring_'+split_name);write_json(out/'status.json', status)
            rows = selected[split_name]
            ids = [r['id'] for r in rows]
            result = predictor.predict([r['text'] for r in rows], ids)
            write_json(out/f'{split_name}-actual.json', result)
            cache = read_predictions(root, original, metadata, split_name, ids, {r['id']: r for r in rows})
            labels = np.asarray([r['label'] for r in rows])
            mask = np.asarray([s['truncated'] for s in result['stats']])
            expected, _, _ = evaluated_views(cache, labels, mask, frozen)
            checks = {n+'/'+v: compare_trace(result['models'][n][v], cache[n][v]['rows'], spec['tolerance'])
                      for n in MODEL_NAMES for v in ('head', 'head_tail')}
            checks['primary_fused'] = compare_fused(ids, result['primary'], expected['two_model_fitted'], spec['tolerance'])
            checks['equal_fused'] = compare_fused(ids, result['equal'], expected['two_model_equal'], spec['tolerance'])
            if split_name == 'validation':
                saved = json.loads((root/spec['cached_fused']).read_text())['rows']
                if [r['sample_id'] for r in saved] != ids:
                    raise ValueError('Saved fused IDs differ')
                checks['persisted_primary_fused'] = compare_fused(ids, result['primary'],
                    [r['probabilities'] for r in saved], spec['tolerance'])
            comparisons[split_name] = checks
            all_passed = all_passed and all(v['passed'] for v in checks.values())
            all_scores[split_name] = {k: stratified_metrics(labels, result[k], mask)
                                     for k in ('primary', 'equal', 'baseline_head_tail')}
            print(f'{split_name}: {len(rows)} actual reviews, parity={all(v["passed"] for v in checks.values())}', flush=True)
        predictor.verify_assets()
        verify_inputs(root, spec)
        summary = {'run_id': spec['run_id'], 'parity_passed': all_passed, 'comparisons': comparisons,
                   'metrics': all_scores, 'benchmark': benchmark_report,
                   'test_accessed': False, 'training_performed': False, 'parameters_refitted': False,
                   'default_deployment_changed': False, 'backend_scope': spec['device']+' FP32',
                   'is_mlx_fp16_verification': False}
        write_json(out/'summary.json', summary)
        status.update(state='completed' if all_passed else 'verification_failed', completed_utc=now(),
                      parity_passed=all_passed, elapsed_seconds=time.monotonic()-started)
        write_json(out/'status.json', status)
        return summary
    except Exception as exc:
        status.update(state='failed', error=f'{type(exc).__name__}: {exc}', failed_utc=now())
        write_json(out/'status.json', status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--validate-config', action='store_true')
    args = parser.parse_args()
    spec = validate_config(json.loads(args.config.read_text()))
    if args.validate_config:
        print('Frozen inference validation configuration is valid; no model loaded.')
    else:
        result = run(spec)
        print(json.dumps({'parity_passed': result['parity_passed'],
                          'validation_accuracy': result['metrics']['validation']['primary']['all']['accuracy']}))
        if not result['parity_passed']:
            raise SystemExit(2)


if __name__ == '__main__':
    main()

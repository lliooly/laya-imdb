"""Frozen Mac precise MLX FP32 fusion validation; no fitting, training or official test access."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import numpy as np
import train_laya
from common import ROOT, file_digest, read_jsonl, write_json
from export_laya_posthoc import select_splits
from laya_decision_views import stratified_metrics
from laya_fusion_inference import MODEL_NAMES
from laya_mlx_fusion_inference import benchmark
from laya_mlx_precise_inference import PreciseMLXFusionPredictor as MLXFusionPredictor
from validate_laya_fusion_inference import (now, relative_path, verify_inputs, compare_trace, compare_fused,
                                           validate_config as validate_fp32_config)


def validate_config(spec):
    if spec.get('device') != 'mlx-fp32-precise' or spec.get('tolerance') != {
            'logits_absolute': .1, 'probabilities_absolute': .01}:
        raise ValueError('Require predeclared precise MLX FP32 protocol')
    # The other admission rules are identical to the verified CUDA experiment.
    validate_fp32_config(dict(spec, device='mps', tolerance={
        'logits_absolute': .001, 'probabilities_absolute': .0001}))
    return spec


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
        predictor = MLXFusionPredictor(root, spec['checkpoint_root'], spec['models'], frozen,
                                      spec['device'], spec['batch_size'])
        import mlx.core as mx
        hardware = {'platform': platform.platform(), 'device': 'MLX Metal GPU', 'dtype': 'float32', 'mlx_enable_tf32': '0',
                    'compile': False, 'cache_limit_bytes': 2*1024**3, 'metal': mx.metal.device_info(),
                    'versions': {n: version(n) for n in ('mlx', 'laya-mlx', 'laya', 'transformers', 'numpy')}}
        protocol = {'spec': spec, 'hardware': hardware, 'frozen_coefficient': predictor.coefficient,
                    'source_sha256': {p: file_digest(root/p) for p in
                        ('scripts/validate_laya_mlx_precise.py', 'scripts/laya_mlx_precise_inference.py',
                         'scripts/validate_laya_mlx_fusion.py', 'scripts/laya_mlx_fusion_inference.py',
                         'scripts/validate_laya_fusion_inference.py', 'scripts/laya_fusion_inference.py',
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
                   'default_deployment_changed': False, 'backend_scope': 'MLX Metal FP32, MLX_ENABLE_TF32=0',
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

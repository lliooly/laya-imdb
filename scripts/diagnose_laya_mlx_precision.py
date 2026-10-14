"""Posthoc numerical diagnosis, not a model-selection or deployment experiment."""
import gc
import json
from pathlib import Path
import time
import numpy as np
import train_laya
from common import ROOT, file_digest, read_jsonl, write_json
from export_laya_posthoc import prepare_rules, input_digest
from laya_fusion_inference import MODEL_NAMES
from validate_laya_fusion_inference import now, verify_inputs
from validate_laya_mlx_fusion import validate_config


def probability(z, temperature):
    z = np.asarray(z, dtype=np.float64)
    p = np.exp((z-z.max())/temperature)
    return p/p.sum()


def run():
    spec = validate_config(json.loads((ROOT/'configs/laya-fusion-inference-mlx-fp16-20261014-v1.json').read_text()))
    prior = ROOT/'reports'/spec['run_id']
    cuda = ROOT/'reports/laya-fusion-inference-cuda-fp32-20261014-v1'
    out = ROOT/'reports/laya-mlx-precision-diagnosis-20261014-v1'
    out.mkdir(exist_ok=False)
    status = {'state': 'running', 'started_utc': now(), 'test_accessed': False,
              'parameters_refitted': False, 'posthoc_diagnostic_only': True}
    write_json(out/'status.json', status)
    started = time.monotonic()
    try:
        verify_inputs(ROOT, spec)
        summary = json.loads((prior/'summary.json').read_text())
        old = json.loads((prior/'validation-actual.json').read_text())
        reference = json.loads((cuda/'validation-actual.json').read_text())
        ids = reference['sample_ids']
        if old['sample_ids'] != ids:
            raise ValueError('Reference order differs')
        selected = sorted({sid for v in summary['comparisons']['validation'].values()
                           for sid in v['out_of_tolerance_ids']})
        if len(selected) != 4:
            raise ValueError('Expected the four previously reported numerical outliers')
        rows = list(read_jsonl(ROOT/'data/processed/validation.jsonl'))
        if [r['id'] for r in rows] != ids:
            raise ValueError('Development order differs')
        sources = [prior/'summary.json', prior/'validation-actual.json',
                   cuda/'validation-actual.json', Path(__file__), ROOT/'uv.lock',
                   ROOT/'configs/laya-fusion-inference-mlx-fp16-20261014-v1.json']
        write_json(out/'protocol.json', {'selected_ids': selected,
            'source_sha256': {str(p.relative_to(ROOT)): file_digest(p) for p in sources},
            'variants': ['fp16_original_batch8', 'fp16_batch1', 'fp32_original_batch8', 'fp32_batch1'],
            'selection': 'Union of all prior validation numerical outliers; deliberately posthoc',
            'native_temperatures_and_coefficient_unchanged': True,
            'deployment_changed': False, 'official_test_accessed': False})
        train_laya.block_network()
        import mlx.core as mx
        from laya_mlx import Agent
        from laya_mlx.agent import collate_items
        from transformers import AutoTokenizer
        import laya.train as upstream
        from laya_mlx_fusion_inference import MLXFusionPredictor
        frozen = json.loads((ROOT/spec['frozen_fit']).read_text())
        checker = MLXFusionPredictor(ROOT, spec['checkpoint_root'], spec['models'], frozen)
        records = []
        for name in MODEL_NAMES:
            folder = checker.folder(name)
            tok = AutoTokenizer.from_pretrained(str(folder/'tokenizer'), local_files_only=True)
            head, tail, stats, _, _ = prepare_rules(tok, rows)
            for dtype in ('float16', 'float32'):
                agent = Agent(str(folder), device='gpu', dtype=dtype, compile=False, cache_prompts=False)
                temperature = agent.cfg['temperature'][0]
                for view, items in [('head', head), ('head_tail', tail)]:
                    # Nontruncated tail reuses head; reproduce the head batch for these samples.
                    for sid in selected:
                        index = ids.index(sid)
                        effective_view = 'head' if view == 'head_tail' and not stats[index]['truncated'] else view
                        effective_items = head if effective_view == 'head' else items
                        order = list(range(len(rows))) if effective_view == 'head' else [i for i,s in enumerate(stats) if s['truncated']]
                        pos = order.index(index)
                        original_batch = order[(pos//8)*8:(pos//8+1)*8]
                        for batch_name, batch_ids in [('original_batch8', original_batch), ('batch1', [index])]:
                            prepared = [upstream.encode_item(tok, effective_items[i], 512, 192, [0,1], parallel=False) for i in batch_ids]
                            slot = batch_ids.index(index)
                            expected = reference['models'][name][view][index]
                            if input_digest(prepared[slot]) != expected['input_sha256']:
                                raise ValueError('Prepared input changed')
                            batch = collate_items(prepared, tok.pad_token_id)
                            z, act = agent.forward(batch)
                            logits = np.asarray(z, dtype=np.float64)[slot]
                            if not np.isfinite(logits).all() or not np.isfinite(np.asarray(act)).all():
                                raise FloatingPointError('Nonfinite inference')
                            p = probability(logits, temperature)
                            previous = old['models'][name][view][index]
                            records.append({'model': name, 'view': view, 'effective_view': effective_view,
                                'sample_id': sid, 'dtype': dtype, 'batch': batch_name,
                                'batch_sample_ids': [ids[i] for i in batch_ids],
                                'padded_length': int(batch['input_ids'].shape[1]),
                                'input_sha256': input_digest(prepared[slot]), 'logits': logits.tolist(),
                                'probabilities': p.tolist(),
                                'max_probability_delta_cuda': float(np.max(abs(p-expected['probabilities']))),
                                'max_logit_delta_cuda': float(np.max(abs(logits-expected['logits']))),
                                'max_logit_delta_prior_fp16': float(np.max(abs(logits-previous['logits']))),
                                'class_changed_cuda': int(logits.argmax()) != expected['pred_label']})
                del agent
                gc.collect();mx.clear_cache()
                print(name, dtype, 'completed', flush=True)
        checker.verify_assets()
        aggregates = {}
        for dtype in ('float16', 'float32'):
            for batch in ('original_batch8', 'batch1'):
                subset = [r for r in records if r['dtype']==dtype and r['batch']==batch]
                aggregates[dtype+'/'+batch] = {
                    'view_sample_count': len(subset),
                    'max_probability_delta_cuda': max(r['max_probability_delta_cuda'] for r in subset),
                    'max_logit_delta_cuda': max(r['max_logit_delta_cuda'] for r in subset),
                    'class_changes': sum(r['class_changed_cuda'] for r in subset),
                    'probability_over_001': sum(r['max_probability_delta_cuda']>.01 for r in subset),
                    'max_logit_delta_prior_fp16': max(r['max_logit_delta_prior_fp16'] for r in subset)}
        result = {'posthoc_diagnostic_only': True, 'selected_ids': selected, 'aggregates': aggregates,
                  'records': records, 'test_accessed': False, 'deployment_changed': False,
                  'full_fp32_validation_performed': False}
        write_json(out/'summary.json', result)
        status.update(state='completed', completed_utc=now(), elapsed_seconds=time.monotonic()-started)
        write_json(out/'status.json', status)
        print(json.dumps(aggregates, indent=2))
    except Exception as exc:
        status.update(state='failed', error=str(exc));write_json(out/'status.json', status)
        raise

if __name__ == '__main__':
    run()

"""Predeclared EXP-1/2 and N1 zero-training study, calibration first, test sealed."""
import argparse
import gc
import json
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np

import train_laya
from common import ROOT, file_digest, read_jsonl, write_json
from export_laya_posthoc import prepare_rules, input_digest, select_splits
from laya_decision_views import (WINDOWS, candidate_features, canonical_logits, compare_views,
    fit_candidate_coefficient, gated_equal_fusion, probabilities, stratified_metrics)
from laya_runtime import prepare_runtime, apply_runtime


def validate_study(spec):
    expected = {'run_id', 'device', 'runtime_profile', 'batch_size', 'checkpoints', 'windows'}
    if (not isinstance(spec, dict) or set(spec) != expected or spec['device'] not in ('cuda', 'mps')
            or spec['runtime_profile'] != ('strict' if spec['device'] == 'cuda' else None)
            or type(spec['batch_size']) is not int or spec['batch_size'] < 1
            or spec['windows'] != list(WINDOWS)):
        raise ValueError('Study requires fixed windows, positive batch size and strict CUDA/FP32')
    run = spec['run_id']
    if not isinstance(run, str) or not run or Path(run).name != run or run in ('.', '..'):
        raise ValueError('run_id must be a fresh directory name')
    if not isinstance(spec['checkpoints'], list) or not spec['checkpoints']:
        raise ValueError('Predeclare checkpoint identities')
    names = set()
    for cp in spec['checkpoints']:
        if set(cp) != {'name', 'checkpoint', 'manifest', 'weight_sha256', 'manifest_sha256'}:
            raise ValueError('Unexpected checkpoint fields')
        if cp['name'] in names or not cp['name'] or Path(cp['name']).name != cp['name'] or cp['name'] in ('.', '..'):
            raise ValueError('Checkpoint names must be distinct directory names')
        names.add(cp['name'])
        for key, folder in (('checkpoint', 'checkpoints'), ('manifest', 'reports')):
            path = Path(cp[key])
            if path.is_absolute() or '..' in path.parts or not path.parts or path.parts[0] != folder:
                raise ValueError('Checkpoint/manifest must be repository-relative approved paths')
        for key in ('weight_sha256', 'manifest_sha256'):
            value = cp[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Predeclare valid checkpoint and manifest SHA256')
    return spec


def check_checkpoint(cp, split):
    folder, manifest_path = ROOT / cp['checkpoint'], ROOT / cp['manifest']
    if file_digest(folder / 'model.safetensors') != cp['weight_sha256'] or file_digest(manifest_path) != cp['manifest_sha256']:
        raise ValueError('Declared checkpoint/manifest hash mismatch')
    manifest = json.loads(manifest_path.read_text())
    if manifest['question'] != train_laya.QUESTION:
        raise ValueError('Study requires the frozen sentiment question')
    for name in ('train', 'validation'):
        if manifest[name]['source_sha256'] != split['jsonl_sha256'][name] or manifest[name]['ids'] != split['ids'][name]:
            raise ValueError('Checkpoint manifest differs from frozen split')
    if (len(manifest['gradient_train_ids']) != 22100 or len(manifest['calibration_ids']) != 400
            or len(manifest['validation']['ids']) != 2500):
        raise ValueError('Require full original 22100/400/2500 partition')
    original = ROOT / manifest['model_lock']['local_dir'] / 'tokenizer/tokenizer.json'
    if file_digest(original) != manifest['model_lock']['sha256']['tokenizer/tokenizer.json']:
        raise ValueError('Locked tokenizer hash mismatch')
    def semantics(path):
        value = json.loads(path.read_text())
        value.pop('padding', None)
        value.pop('truncation', None)
        return value
    if semantics(folder / 'tokenizer/tokenizer.json') != semantics(original):
        raise ValueError('Checkpoint tokenizer semantics changed')
    cfg = json.loads((folder / 'rl_agent_config.json').read_text())
    if (cfg['max_len'], cfg['head_max_len']) != (512, 192) or cfg.get('parallel_layout', False):
        raise ValueError('Require sequential 512/192 checkpoint')
    return manifest


def export_views(torch, upstream, model, tok, cfg, rows, device, batch_size, directory, split_name):
    head, tail, stats, _, _ = prepare_rules(tok, rows)
    temperature = float(cfg['temperature'][0])
    if cfg.get('temperature_by_options'):
        raise ValueError('Study uses saved scalar native temperature, not option bucket overrides')
    outputs, timings = {}, {}
    captured = []
    def capture(module, inputs):
        values = inputs[0].detach().float().cpu().numpy()
        if values.shape[1] != 2 or not np.isfinite(values).all():
            raise FloatingPointError('Invalid scorer representations')
        captured.append(candidate_features(values))
    indices_all = list(range(len(rows)))
    changed = [i for i, stat in enumerate(stats) if stat['truncated']]
    for rule, items, indices, order in (
            ('head', head, indices_all, [0, 1]), ('reverse', head, indices_all, [1, 0]),
            ('head_tail', tail, changed, [0, 1])):
        # Preserve exact input/probability identity on short samples.
        result = [dict(row, reused_head_prediction=True) for row in outputs['head']] if rule == 'head_tail' else [None]*len(rows)
        began = time.monotonic()
        hook = model.scorer.register_forward_pre_hook(capture) if rule == 'head' else None
        try:
            for start in range(0, len(indices), batch_size):
                ii = indices[start:start+batch_size]
                prepared = [upstream.encode_item(tok, items[i], 512, 192, order, parallel=False) for i in ii]
                batch = upstream.collate_items([prepared], tok.pad_token_id)
                logits = upstream._forward(model, batch, torch.device(device), False, False).cpu().numpy()
                features = captured.pop() if rule == 'head' else None
                if logits.shape != (len(ii), 2):
                    raise ValueError('Forward result shape mismatch')
                for j, i in enumerate(ii):
                    z = canonical_logits(logits[j], order)
                    result[i] = {'sample_id': rows[i]['id'], 'true_label': rows[i]['label'],
                                 'input_sha256': input_digest(prepared[j]), 'logits': z.tolist(),
                                 'probabilities': probabilities(z[None, :], temperature)[0].tolist(),
                                 'pred_label': int(z.argmax()), 'option_order': order, **stats[i],
                                 'reused_head_prediction': False}
                    if features is not None:
                        result[i]['candidate_features'] = features[j].tolist()
                print(f'{split_name}/{rule}: {min(start+batch_size,len(indices))}/{len(indices)}', flush=True)
        finally:
            if hook:
                hook.remove()
        outputs[rule] = result
        timings[rule] = {'forward_rows': len(indices), 'seconds': time.monotonic()-began}
        write_json(directory / f'{split_name}-{rule}.json', {'split': split_name, 'rule': rule,
                   'rows': result, 'test_accessed': False})
    return outputs, timings, temperature


def load_model(torch, upstream, cp, device):
    model, tok, cfg = upstream.load_checkpoint(str(ROOT / cp['checkpoint']))
    if upstream.uses_parallel_layout(cfg):
        raise ValueError('Study requires non-parallel layout')
    model.float().eval().to(device)
    if any(not torch.isfinite(p).all().item() for p in model.parameters()):
        raise FloatingPointError('Nonfinite checkpoint parameters')
    return model, tok, cfg


def sliding_windows(torch, cp, rows, stats, spec, directory):
    from laya.agent import Agent, _option_logits
    agent = Agent(str(ROOT / cp['checkpoint']), device=spec['device'], backend='eager',
                  expected_sha256={'model.safetensors': cp['weight_sha256']})
    agent.model.float().eval()
    agent.amp_enabled = False
    agent.dtype = torch.float32
    if agent.device != torch.device(spec['device']):
        raise RuntimeError('Study refuses upstream device fallback')
    if any(not torch.isfinite(p).all().item() for p in agent.model.parameters()):
        raise FloatingPointError('Nonfinite sliding-window checkpoint parameters')
    forward_device = next(agent.model.parameters()).device
    if forward_device.type != torch.device(spec['device']).type:
        raise RuntimeError('Study refuses upstream device fallback')
    def require_forward_device(module, args):
        if args[0].device != forward_device:
            raise RuntimeError('Study refuses upstream device fallback')
    agent.model.register_forward_pre_hook(require_forward_device)
    original_decode = agent._decode_answers
    def decode(logits, act, items, ids, internal, offset, lang=None):
        answers = original_decode(logits, act, items, ids, internal, offset, lang)
        raw = _option_logits(logits, items, offset)
        for j, qid in enumerate(ids):
            # Upstream displays four decimals. Preserve full precision for metrics,
            # without changing its most-confident-window selection or answer.
            answers[qid]['raw_probabilities'] = probabilities(np.asarray(raw[j])[None, :], agent.temperature[0])[0].tolist()
        return answers
    agent._decode_answers = decode
    report = {}
    for window in WINDOWS:
        results = []
        for row, stat in zip(rows, stats):
            if not stat['truncated']:
                continue
            began = time.monotonic()
            prediction = agent.predict_long(row['text'], {'sentiment': train_laya.QUESTION},
                window=window, stride=window//2, batch_size=spec['batch_size'])
            answer = prediction['answers']['sentiment']
            probability = answer['raw_probabilities']
            if answer['choice'] != ('negative', 'positive')[int(np.argmax(probability))]:
                raise ValueError('Upstream long prediction and full probabilities disagree')
            results.append({'sample_id': row['id'], 'true_label': row['label'],
                            'probabilities': probability, 'window': answer['window'],
                            'window_count': prediction['usage']['windows'],
                            'seconds': time.monotonic()-began})
        if results:
            labels = [r['true_label'] for r in results]
            prob = [r['probabilities'] for r in results]
            report[str(window)] = {'metrics': stratified_metrics(labels, prob, [True]*len(labels))['all'],
                'window_count': sum(r['window_count'] for r in results),
                'seconds': sum(r['seconds'] for r in results), 'stride': window//2,
                'probability_scope': 'native deciding-window probability; not document calibrated'}
            base = [r for r in stats if r['truncated']]
            tail_doc = json.loads((directory / 'validation-head_tail.json').read_text())
            tail = [r for r in tail_doc['rows'] if r['truncated']]
            if [r['sample_id'] for r in tail] != [r['sample_id'] for r in results]:
                raise ValueError('Long-window and head-tail IDs differ')
            baseline_prob = [r['probabilities'] for r in tail]
            report[str(window)]['versus_head_tail'] = compare_views([r['sample_id'] for r in base],
                labels, baseline_prob, prob, [True]*len(labels))
            report[str(window)]['baseline_head_tail_accuracy'] = stratified_metrics(labels,
                baseline_prob, [True]*len(labels))['all']['accuracy']
            report[str(window)]['accuracy_target_met'] = report[str(window)]['metrics']['accuracy'] >= .975
        else:
            report[str(window)] = {'count': 0}
        write_json(directory / f'validation-long-{window}.json', {'rows': results, 'test_accessed': False})
        print(f'predict_long window={window}: {len(results)} truncated samples', flush=True)
    report['stop_n5'] = all(report[str(w)].get('metrics', {}).get('accuracy', 0)
                          < report[str(w)].get('baseline_head_tail_accuracy', 0) for w in WINDOWS)
    report['latency_acceptable'] = None
    report['latency_note'] = 'No latency budget was declared; report timing without declaring success.'
    return report


def evaluate_views(outputs, fit):
    base = outputs['head']
    labels, ids = [r['true_label'] for r in base], [r['sample_id'] for r in base]
    mask = np.array([r['truncated'] for r in base], dtype=bool)
    views = {name: np.array([r['probabilities'] for r in rows]) for name, rows in outputs.items()}
    views['gated_equal'] = gated_equal_fusion(views['head'], views['head_tail'], mask)
    logits = np.array([r['logits'] for r in base])
    features = np.array([r['candidate_features'] for r in base])
    views['candidate_rerank'] = probabilities(logits+fit['coefficient']*features, fit['temperature'])
    report = {'single_and_fused': {name: stratified_metrics(labels, p, mask) for name, p in views.items()},
              'versus_head': {name: compare_views(ids, labels, views['head'], p, mask)
                              for name, p in views.items() if name != 'head'}}
    delta = report['single_and_fused']['gated_equal']['all']['accuracy']-report['single_and_fused']['head']['all']['accuracy']
    strata = report['single_and_fused']
    stratified_delta = strata['gated_equal']['truncated'].get('accuracy', 0)-strata['head']['truncated'].get('accuracy', 0)
    p = report['versus_head']['gated_equal']['paired']['mcnemar_exact_two_sided_p']
    report['exp1'] = {'success': delta >= .001-1e-12 and stratified_delta >= .005-1e-12,
                     'negative': p > .2 and delta < .001-1e-12,
                     'delta_accuracy_pp': delta*100, 'delta_truncated_pp': stratified_delta*100,
                     'mcnemar_p': p, 'window_selection_on_development': False}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--validate-config', action='store_true')
    args = parser.parse_args()
    spec = validate_study(json.loads(args.config.read_text()))
    if args.validate_config:
        print(json.dumps({'state': 'config_valid', 'training_performed': False, 'data_accessed': False,
                          'model_accessed': False, 'accelerator_checked': False}))
        return
    out = ROOT / 'reports' / spec['run_id']
    if out.exists():
        raise FileExistsError('Use a new run_id; reports are immutable')
    train_laya.block_network()
    profile = prepare_runtime(dict(spec, training={'amp': False}))
    import torch
    import laya.train as upstream
    if version('laya') != '0.4.1':
        raise ValueError('Study requires pinned Laya 0.4.1')
    runtime = apply_runtime(torch, profile)
    train_laya.require_device(torch, spec['device'])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    split_path = ROOT / 'reports/split_manifest.json'
    split = json.loads(split_path.read_text())
    for name in ('train', 'validation'):
        if file_digest(ROOT / f'data/processed/{name}.jsonl') != split['jsonl_sha256'][name]:
            raise ValueError('Frozen source hash mismatch')
    manifests = {cp['name']: check_checkpoint(cp, split) for cp in spec['checkpoints']}
    if len({tuple(m['calibration_ids']) for m in manifests.values()}) != 1:
        raise ValueError('All models must use identical original calibration IDs')
    train_rows = list(read_jsonl(ROOT / 'data/processed/train.jsonl'))
    by_id = {row['id']: row for row in train_rows}
    # Validate exact partitions before any inference, without loading dev text/labels.
    dev_placeholders = [{'id': sid, 'label': 0} for sid in split['ids']['validation']]
    for manifest in manifests.values():
        select_splits(manifest, split, train_rows, dev_placeholders)
    calibration = [by_id[i] for i in next(iter(manifests.values()))['calibration_ids']]
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'protocol.json', {'spec': spec, 'config_sha256': file_digest(args.config),
        'script_sha256': file_digest(Path(__file__)), 'views_sha256': file_digest(ROOT / 'scripts/laya_decision_views.py'),
        'uv_lock_sha256': file_digest(ROOT / 'uv.lock'), 'split_manifest_sha256': file_digest(split_path),
        'runtime': runtime, 'dtype': 'float32', 'amp': False, 'tf32': False, 'test_accessed': False,
        'versions': {k: version(k) for k in ('torch', 'laya', 'transformers', 'numpy')},
        'validation_scope': 'one frozen study batch; no window/weight selection on development'})
    fits, reports, development_outputs = {}, {}, {}
    write_json(out / 'status.json', {'state': 'running', 'training_performed': False, 'test_accessed': False})
    try:
        with torch.inference_mode():
            for cp in spec['checkpoints']:
                folder = out / cp['name']
                folder.mkdir()
                model, tok, cfg = load_model(torch, upstream, cp, spec['device'])
                outputs, timing, temperature = export_views(torch, upstream, model, tok, cfg, calibration,
                    spec['device'], spec['batch_size'], folder, 'calibration')
                rows = outputs['head']
                fits[cp['name']] = fit_candidate_coefficient([r['logits'] for r in rows],
                    [r['candidate_features'] for r in rows], [r['true_label'] for r in rows], temperature)
                write_json(folder / 'frozen-fit.json', fits[cp['name']])
                reports[cp['name']] = {'calibration_timing': timing}
                del model, tok
                gc.collect()
            # All fitted coefficients are already frozen on disk at this point.
            validation = list(read_jsonl(ROOT / 'data/processed/validation.jsonl'))
            for cp in spec['checkpoints']:
                folder = out / cp['name']
                select_splits(manifests[cp['name']], split, train_rows, validation)
                model, tok, cfg = load_model(torch, upstream, cp, spec['device'])
                outputs, timing, _ = export_views(torch, upstream, model, tok, cfg, validation,
                    spec['device'], spec['batch_size'], folder, 'validation')
                reports[cp['name']].update(evaluate_views(outputs, fits[cp['name']]), validation_timing=timing)
                development_outputs[cp['name']] = outputs
                del model, tok
                gc.collect()
                reports[cp['name']]['exp2'] = sliding_windows(torch, cp, validation, outputs['head'], spec, folder)
                if file_digest(ROOT / cp['checkpoint'] / 'model.safetensors') != cp['weight_sha256']:
                    raise ValueError('Checkpoint changed during study')
        summary = {'state': 'completed', 'models': reports, 'training_performed': False,
                   'test_accessed': False, 'validation_used_for_fitting': False, 'n1_calibration_fits': fits}
        if set(development_outputs) == {'prefix-trained', 'random-window-trained'}:
            comparisons = {}
            for rule in ('head', 'head_tail'):
                a, b = (development_outputs[name][rule] for name in ('prefix-trained', 'random-window-trained'))
                if [(r['sample_id'], r['true_label'], r['truncated']) for r in a] != [
                        (r['sample_id'], r['true_label'], r['truncated']) for r in b]:
                    raise ValueError('EXP-3 paired evaluation IDs/labels/strata differ')
                comparisons[rule] = compare_views([r['sample_id'] for r in a], [r['true_label'] for r in a],
                    [r['probabilities'] for r in a], [r['probabilities'] for r in b], [r['truncated'] for r in a])
            sa, sb = (reports[name]['single_and_fused'] for name in ('prefix-trained', 'random-window-trained'))
            delta = sb['head_tail']['all']['accuracy']-sa['head_tail']['all']['accuracy']
            short_delta = sb['head_tail']['untruncated']['accuracy']-sa['head_tail']['untruncated']['accuracy']
            summary['exp3_training_factor'] = {'primary_inference': 'head_tail_in_both_arms',
                'secondary_inference': 'head_in_both_arms', 'paired': comparisons,
                'delta_accuracy_pp': delta*100, 'delta_untruncated_pp': short_delta*100,
                'success': delta >= .002-1e-12 and short_delta >= -1e-12,
                'early_stopping_on_development': False, 'primary_checkpoint': 'final_epoch'}
        write_json(out / 'summary.json', summary)
        write_json(out / 'status.json', {'state': 'completed', 'training_performed': False, 'test_accessed': False})
    except BaseException as exc:
        write_json(out / 'status.json', {'state': 'failed', 'error': repr(exc), 'test_accessed': False})
        raise


if __name__ == '__main__':
    main()

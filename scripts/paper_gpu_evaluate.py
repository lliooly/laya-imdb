"""Calibration then gated development evaluation for fixed supplementary runs."""
import argparse
import gc
import json
from pathlib import Path
import time

import paper_gpu_supplement as study
from common import ROOT, file_digest, read_jsonl, write_json


def load(kind, seed):
    import torch
    device = study.setup(seed)
    run, report, checkpoint = study.paths(kind, seed)
    folder = checkpoint/'epoch-002'
    verified = study.read(report/'reload-verification.json')
    if verified['bundle_sha256'] != file_digest(folder/'bundle.json'):
        raise ValueError('Unverified model cannot be evaluated')
    study.check_bundle(folder)
    if kind == 'deberta':
        import deberta_bf16_training as core
        from transformers import AutoTokenizer
        model = core.load_saved(folder).to(device).eval()
        tok = AutoTokenizer.from_pretrained(str(ROOT/'.cache/deberta-v3-large'), local_files_only=True, use_fast=False)
        config = None
    else:
        import laya.train as upstream
        model, tok, config = upstream.load_checkpoint(str(folder))
        model = model.to(device).eval()
    return run, report, folder, model, tok, config, device


def forward(kind, model, tok, config, rows, device):
    import numpy as np
    if kind == 'deberta':
        from deberta_inputs import encode
        from deberta_bf16_training import predict
        items = [dict(id=r['id'], label=r['label'], **encode(tok, r['text'])) for r in rows]
        logits, seconds = predict(model, items, tok.pad_token_id, device, batch_size=8)
        return dict(logits=logits, truncated=[r['truncated'] for r in items], forward_rows=len(rows), seconds=seconds)
    import laya.train as upstream
    from export_laya_posthoc import prepare_rules
    began = time.monotonic()
    head, tail, stats, _, _ = prepare_rules(tok, rows)
    head_z = np.asarray([r[1] for r in upstream.calibration_records(model, tok, head, device, 512, 192, batch_size=8)])
    tail_z = head_z.copy()
    changed = [i for i, r in enumerate(stats) if r['truncated']]
    if changed:
        tail_z[changed] = [r[1] for r in upstream.calibration_records(model, tok, [tail[i] for i in changed],
                                                                  device, 512, 192, batch_size=8)]
    return dict(head_logits=head_z.tolist(), tail_logits=tail_z.tolist(),
                truncated=[r['truncated'] for r in stats], forward_rows=len(rows)+len(changed),
                seconds=time.monotonic()-began)


def probabilities(kind, output, temperature):
    import numpy as np
    from laya_decision_views import probabilities as softmax, gated_equal_fusion
    if kind == 'deberta':
        return softmax(output['logits'], temperature)
    return gated_equal_fusion(softmax(output['head_logits'], temperature),
                              softmax(output['tail_logits'], temperature), np.asarray(output['truncated'], bool))


def export(kind, seed, split_name):
    import numpy as np
    from comparison_metrics import fit_temperature
    run, report, folder, model, tok, config, device = load(kind, seed)
    target = report/(split_name+'.json')
    if target.exists():
        raise FileExistsError(target)
    source, split = study.verified_training_rows()
    if split_name == 'calibration':
        rows = [source[i] for i in split['calibration_ids']]
    else:
        frozen = study.read(ROOT/'reports'/study.STUDY/'all-models-frozen.json')
        if frozen['model_bundle_sha256'][run] != file_digest(folder/'bundle.json'):
            raise ValueError('Final model changed after all-model freeze')
        rows = list(read_jsonl(ROOT/'data/processed/validation.jsonl'))
        manifest = study.read(ROOT/'reports/split_manifest.json')
        if (file_digest(ROOT/'data/processed/validation.jsonl') != manifest['jsonl_sha256']['validation']
                or [r['id'] for r in rows] != split['validation']['ids']):
            raise ValueError('Development split changed')
    output = forward(kind, model, tok, config, rows, device)
    if split_name == 'calibration':
        if kind == 'deberta':
            z = np.asarray(output['logits'])
            temperature = fit_temperature(z[:, 1]-z[:, 0], [r['label'] for r in rows], bounds=(.5, 5.))
        else:
            temperature = float(config['temperature'][0])
    else:
        calibration_path = report/'calibration.json'
        if frozen['calibration_sha256'][run] != file_digest(calibration_path):
            raise ValueError('Calibration changed after freeze')
        temperature = study.read(calibration_path)['temperature']
    p = probabilities(kind, output, temperature)
    write_json(target, dict(run_id=run, kind=kind, seed=seed, split=split_name, temperature=temperature,
        sample_ids=[r['id'] for r in rows], true_labels=[r['label'] for r in rows],
        positive_probabilities=p[:, 1].tolist(), model_sha256=file_digest(folder/'model.safetensors'),
        test_accessed=False, **output))


def freeze():
    from laya_probability_fusion import fit_mixing_coefficient
    import numpy as np
    out = ROOT/'reports'/study.STUDY/'all-models-frozen.json'
    if out.exists():
        raise FileExistsError(out)
    models, calibrations, groups = {}, {}, []
    for repeat in range(3):
        data = {}
        for kind in study.SEEDS:
            seed = study.SEEDS[kind][repeat]
            run, report, checkpoint = study.paths(kind, seed)
            if (report/'development.json').exists():
                raise ValueError('Development was accessed before the group freeze')
            verified = study.read(report/'reload-verification.json')
            models[run] = file_digest(checkpoint/'epoch-002/bundle.json')
            if verified['bundle_sha256'] != models[run]:
                raise ValueError('Independent reload verification is stale')
            calibrations[run] = file_digest(report/'calibration.json')
            data[kind] = study.read(report/'calibration.json')
        a, b = data['baseline'], data['combined']
        if any(data[k]['sample_ids'] != a['sample_ids'] or data[k]['true_labels'] != a['true_labels'] for k in data):
            raise ValueError('Calibration IDs or labels differ')
        binary = lambda d: np.column_stack((1-np.asarray(d['positive_probabilities']), d['positive_probabilities']))
        fit = fit_mixing_coefficient(binary(a), binary(b), a['true_labels'])
        groups.append(dict(repeat=repeat+1, seeds={k: study.SEEDS[k][repeat] for k in study.SEEDS},
                           laya_mixture=fit, outer_laya_deberta_weights=[.5, .5]))
    write_json(out, dict(model_bundle_sha256=models, calibration_sha256=calibrations, groups=groups,
        frozen_utc=study.now(), all_nine_runs_verified=True, epoch=2, test_accessed=False,
        development_used_for_selection=False, source_sha256=file_digest(Path(__file__))))


def aggregate():
    import numpy as np
    from comparison_metrics import metrics, paired
    from laya_decision_views import probability_margin
    frozen = study.read(ROOT/'reports'/study.STUDY/'all-models-frozen.json')
    results = []
    for group in frozen['groups']:
        data = {k: study.read(study.paths(k, s)[1]/'development.json') for k, s in group['seeds'].items()}
        reference = data['baseline']
        if any(d['sample_ids'] != reference['sample_ids'] or d['true_labels'] != reference['true_labels'] for d in data.values()):
            raise ValueError('Development predictions are not aligned')
        y = np.asarray(reference['true_labels'])
        p = {k: np.asarray(d['positive_probabilities']) for k, d in data.items()}
        weight = group['laya_mixture']['coefficient']
        p['laya'] = weight*p['baseline']+(1-weight)*p['combined']
        p['fusion'] = .5*p['laya']+.5*p['deberta']
        margins = {k: probability_margin(np.column_stack((1-v, v))) for k, v in p.items()}
        scores = {k: metrics(y, m) for k, m in margins.items()}
        result = dict(repeat=group['repeat'], seeds=group['seeds'], laya_coefficient=weight, metrics=scores,
                      fusion_vs_deberta=paired(y, margins['deberta'], margins['fusion']),
                      fusion_vs_laya=paired(y, margins['laya'], margins['fusion']))
        results.append(result)
        write_json(ROOT/'reports'/study.STUDY/f'repeat-{group["repeat"]}-predictions.json',
                   dict(sample_ids=reference['sample_ids'], true_labels=y.tolist(),
                        positive_probabilities={k: v.tolist() for k, v in p.items()}, test_accessed=False))
    statistics = {}
    for name in ('baseline', 'combined', 'deberta', 'laya', 'fusion'):
        a = [r['metrics'][name]['accuracy'] for r in results]
        statistics[name] = dict(accuracy_mean=float(np.mean(a)), accuracy_sample_sd=float(np.std(a, ddof=1)),
                                accuracy_range=[min(a), max(a)])
    for base in ('deberta', 'laya'):
        delta = [100*(r['metrics']['fusion']['accuracy']-r['metrics'][base]['accuracy']) for r in results]
        statistics['fusion_minus_'+base] = dict(mean_pp=float(np.mean(delta)), sample_sd_pp=float(np.std(delta, ddof=1)),
                                              range_pp=[min(delta), max(delta)], all_deltas_pp=delta)
    summary = dict(state='completed', study_id=study.STUDY, results=results, seed_statistics=statistics,
        official_test_accessed=False, evaluation_count_per_repeat=2500, independent_sample_count=2500,
        independent_new_evaluation_available=False, reused_development_exploratory=True,
        best_seed_selection=False, limitations=['three repetitions only', 'previously reused development set',
            'no new independent evaluation data', 'same development IDs reused across repetitions'])
    write_json(ROOT/'reports'/study.STUDY/'summary.json', summary)
    lines = ['# 论文补充：三个完整系统重复', '',
             '每组重新训练 Laya baseline、Laya combined 和 DeBERTa，共九次两轮训练。固定第二轮，不选择最好种子。', '',
             '| 重复 | Laya | DeBERTa | 最终融合 | 融合 − DeBERTa |', '|---|---:|---:|---:|---:|']
    for r in results:
        m = r['metrics']
        lines.append(f'| {r["repeat"]} | {100*m["laya"]["accuracy"]:.2f}% | {100*m["deberta"]["accuracy"]:.2f}% | {100*m["fusion"]["accuracy"]:.2f}% | {100*(m["fusion"]["accuracy"]-m["deberta"]["accuracy"]):+.2f} pp |')
    lines += ['', '上述成绩均来自同一已多次使用的 2500 条开发记录，属于探索性稳定性证据。三组不能当作 7500 个独立样本，也不能替代新的独立测试。', '']
    (ROOT/'reports'/study.STUDY/'REPORT.md').write_text('\n'.join(lines))


def benchmark():
    """Real forwards for the first newly retrained system; no cached probabilities."""
    import numpy as np
    import torch
    import statistics
    out = ROOT/'reports'/study.STUDY/'benchmark.json'
    if out.exists():
        raise FileExistsError(out)
    frozen = study.read(ROOT/'reports'/study.STUDY/'all-models-frozen.json')
    group = frozen['groups'][0]
    rows = list(read_jsonl(ROOT/'data/processed/validation.jsonl'))[:64]
    results = []
    for mode, kinds in [('deberta', ['deberta']), ('laya', ['baseline', 'combined']),
                         ('fusion', ['baseline', 'combined', 'deberta'])]:
        gc.collect()
        torch.cuda.empty_cache()
        began = time.perf_counter()
        loaded = {kind: load(kind, group['seeds'][kind]) for kind in kinds}
        torch.cuda.synchronize()
        loading_seconds = time.perf_counter()-began
        temperatures = {k: study.read(v[1]/'calibration.json')['temperature'] for k, v in loaded.items()}
        def invoke():
            probabilities_by_kind, counts = {}, {}
            for kind, value in loaded.items():
                _, _, _, model, tok, config, device = value
                output = forward(kind, model, tok, config, rows, device)
                probabilities_by_kind[kind] = probabilities(kind, output, temperatures[kind])
                counts[kind] = output['forward_rows']
            p = probabilities_by_kind
            if mode == 'deberta':
                prediction = p['deberta']
            else:
                w = group['laya_mixture']['coefficient']
                prediction = w*p['baseline']+(1-w)*p['combined']
                if mode == 'fusion':
                    prediction = .5*prediction+.5*p['deberta']
            if prediction.shape != (64, 2) or not np.isfinite(prediction).all():
                raise ValueError('Invalid benchmark predictions')
            return counts
        for _ in range(2):
            invoke()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations, counts = [], []
        for _ in range(5):
            torch.cuda.synchronize()
            began = time.perf_counter()
            counts.append(invoke())
            torch.cuda.synchronize()
            durations.append(time.perf_counter()-began)
        median = statistics.median(durations)
        results.append(dict(mode=mode, seconds=durations, median_seconds=median,
            reviews_per_second=len(rows)/median, amortized_ms_per_review=1000*median/len(rows),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(), model_load_seconds=loading_seconds,
            forward_rows=counts, resident_models=kinds, true_model_forward=True))
        del invoke, loaded
        gc.collect()
        torch.cuda.empty_cache()
    write_json(out, dict(results=results, runtime=study.runtime(), common_sample_ids=[r['id'] for r in rows],
        sample_selection='first 64 fixed development IDs; independent of predictions and labels',
        sample_count=64, batch_size=8, warmup_repetitions=2, measured_repetitions=5,
        scope='CPU tokenization and preparation, transfers, real forwards, calibrated fusion; excludes model loading',
        memory_scope='PyTorch allocated/reserved on this GPU including mode-specific resident weights',
        checkpoint_scope='first newly retrained seed group, not historical deployed weight files',
        single_request_latency=False, test_accessed=False))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('calibration', 'development', 'freeze', 'aggregate', 'benchmark'))
    p.add_argument('--kind', choices=tuple(study.SEEDS))
    p.add_argument('--seed', type=int)
    a = p.parse_args()
    if a.mode == 'freeze':
        freeze()
    elif a.mode == 'aggregate':
        aggregate()
    elif a.mode == 'benchmark':
        benchmark()
    else:
        export(a.kind, a.seed, a.mode)


if __name__ == '__main__':
    main()

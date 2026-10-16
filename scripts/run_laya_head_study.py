"""Bounded sequential CUDA admission, six fixed runs, then gated development."""
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from common import ROOT, file_digest, write_json
from run_laya_head_control import STUDY, PAIRS, name


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def check_admission():
    records = []
    for kind in ('original', 'linear'):
        reports = [ROOT/'reports'/name(42, kind, r) for r in (1, 2)]
        for leaf in ('updates.jsonl', 'training_audit.json', 'epoch_checkpoints.json'):
            if (reports[0]/leaf).read_bytes() != (reports[1]/leaf).read_bytes():
                raise ValueError('Independent admission repeats differ: '+kind+'/'+leaf)
        if gzip.open(reports[0]/'actual_inputs.jsonl.gz', 'rb').read() != gzip.open(reports[1]/'actual_inputs.jsonl.gz', 'rb').read():
            raise ValueError('Admission repeat input records differ')
        a = read(reports[0]/'training_audit.json')
        if (a['updates'], a['gradient_exposures']) != (32, 256):
            raise ValueError('Admission did not complete 32 actual optimizer updates')
        for report in reports:
            frozen = read(report/'frozen.json')
            if not frozen['independent_reload_logits_bitwise_equal']:
                raise ValueError('Independent checkpoint reload did not match')
            records.append({'run_id': report.name,
                            'input_stream_sha256': a['input_stream_sha256'],
                            'initial_encoder_sha256': read(report/'manifest.json')['initial_state_sha256']['encoder'],
                            'frozen_sha256': file_digest(report/'frozen.json')})
        # Exact duplicate repeat weights share storage; retain both paths and all audit evidence.
        for epoch in (1, 2):
            paths = [ROOT/'checkpoints'/r.name/f'epoch-{epoch:03d}'/'model.safetensors' for r in reports]
            if file_digest(paths[0]) != file_digest(paths[1]):
                raise ValueError('Admission repeat checkpoint bytes differ')
            replacement = paths[1].with_suffix('.verified-link')
            os.link(paths[0], replacement)
            os.replace(replacement, paths[1])
    if len({r['initial_encoder_sha256'] for r in records}) != 1 or len({r['input_stream_sha256'] for r in records}) != 1:
        raise ValueError('Heads do not share the same source encoder and actual gradient inputs')
    return {'state': 'passed', 'runs': records, 'repeats_bitwise_exact': True,
            'development_body_read': False, 'test_accessed': False}


def aggregate(report):
    import numpy as np
    from laya_frozen_probe import paired_accuracy
    from run_laya_risk_coverage import wilson
    payloads = {name(s, h): read(ROOT/'reports'/name(s, h)/'validation.json') for s, h in PAIRS}
    reference = payloads[name(42, 'original')]['rows']
    if any(p['rows'] != reference for p in payloads.values()):
        raise ValueError('Development labels, IDs, token inputs or truncation groups differ')
    y = np.array([r['true_label'] for r in reference])
    truncated = np.array([r['truncated'] for r in reference])
    if (len(y), int(truncated.sum())) != (2500, 407):
        raise ValueError('Unexpected fixed development groups')
    predictions = {n: np.argmax(p['logits'], axis=1) for n, p in payloads.items()}
    groups = {'all': np.ones(len(y), dtype=bool), 'truncated': truncated, 'untruncated': ~truncated}
    results = {}
    for group, keep in groups.items():
        yy = y[keep]
        deltas = np.column_stack([(predictions[name(s, 'original')][keep] == yy).astype(float)-
                                  (predictions[name(s, 'linear')][keep] == yy) for s in (42, 43, 44)])
        accuracies = {h: [float((predictions[name(s, h)][keep] == yy).mean()) for s in (42, 43, 44)]
                      for h in ('original', 'linear')}
        per_seed = []
        for s in (42, 43, 44):
            entry = {'seed': s, **paired_accuracy(yy, predictions[name(s, 'linear')][keep],
                    predictions[name(s, 'original')][keep])}
            for h in ('original', 'linear'):
                correct = int((predictions[name(s, h)][keep] == yy).sum())
                entry[h] = {'correct': correct, 'count': len(yy), 'accuracy': correct/len(yy),
                            'accuracy_wilson95': wilson(correct, len(yy))}
            per_seed.append(entry)
        rng = np.random.default_rng(20261015)
        strata = [np.flatnonzero(yy == label) for label in (0, 1)]
        draws = np.empty(10000)
        for i in range(len(draws)):
            indices = np.concatenate([g[rng.integers(len(g), size=len(g))] for g in strata if len(g)])
            draws[i] = 100*deltas[indices].mean()
        seed_effect = 100*deltas.mean(axis=0)
        results[group] = {'count': int(keep.sum()), 'per_seed': per_seed,
            'accuracy_seed_statistics': {h: {'mean': float(np.mean(a)),
                'sample_sd': float(np.std(a, ddof=1)), 'range': [min(a), max(a)]} for h, a in accuracies.items()},
            'mean_original_minus_linear_pp': float(seed_effect.mean()),
            'seed_sample_sd_pp': float(seed_effect.std(ddof=1)),
            'seed_range_pp': [float(seed_effect.min()), float(seed_effect.max())],
            'leave_one_seed_out_means_pp': {str(s): float(np.delete(seed_effect, i).mean()) for i, s in enumerate((42, 43, 44))},
            'shared_id_stratified_bootstrap95_mean_delta_pp': np.quantile(draws, [.025, .975]).tolist(),
            'uncertainty_scope': 'conditional on these six fitted models; excludes training-seed population uncertainty'}
    summary = {'state': 'completed', 'study_id': STUDY, 'finished_utc': now(),
        'primary_direction': 'original minus linear', 'groups': results,
        'run_metrics': {n: {'raw': p['raw_metrics'], 'calibrated': p['calibrated_metrics'],
                           'temperature': p['temperature']} for n, p in payloads.items()},
        'bootstrap_repetitions': 10000, 'bootstrap_seed': 20261015,
        'seed_sign_flip_minimum_two_sided_p': .25, 'practical_investment_reference_pp': .20,
        'equivalence_claim_allowed': False, 'test_accessed': False}
    write_json(report/'summary.json', summary)
    lines = ['# Laya 原始头与线性头配对训练报告', '',
        '六个模型均使用同一已发布 Laya 编码器，每个头重新初始化，训练 2 个 epoch。每个 seed 的两个头实际梯度输入逐条一致。', '',
        '全部模型完成 FP32 独立进程重载、400 条校准集温度拟合和冻结后，才读取开发集正文。正式测试集未访问。', '',
        '| seed | 原始头开发准确率 | 线性头开发准确率 | 原始头 − 线性头 |', '|---|---:|---:|---:|']
    for seed in (42, 43, 44):
        counts = [payloads[name(seed, h)]['raw_metrics']['all']['correct'] for h in ('original', 'linear')]
        lines.append(f'| {seed} | {100*counts[0]/2500:.2f}% | {100*counts[1]/2500:.2f}% | {100*(counts[0]-counts[1])/2500:+.2f} pp |')
    lines.extend(['', '| 分组 | 样本数 | 平均差值 | seed 样本标准差 | 条件性样本 bootstrap 95% 区间 |', '|---|---:|---:|---:|---|'])
    for group, value in results.items():
        lo, hi = value['shared_id_stratified_bootstrap95_mean_delta_pp']
        lines.append(f'| {group} | {value["count"]} | {value["mean_original_minus_linear_pp"]:+.3f} pp | {value["seed_sample_sd_pp"]:.3f} pp | [{lo:+.3f}, {hi:+.3f}] pp |')
    lines.extend(['', '区间只表达固定六个模型下的开发样本不确定性，不包含训练种子总体的不确定性。三个 seed 的双侧符号翻转检验最小 p 为 0.25，不能据此宣称稳定优越或等效。', '',
        '0.20 pp 仅为投入参考值，不是等效界。各 seed 的配对区间、McNemar 名义检验、去掉一个 seed 后的均值、原始与温度校准后的 NLL/ECE 等指标见 summary.json。', '',
        '来源与复现：configs 中的冻结协议绑定源代码及环境锁；各 run 的 manifest、actual_inputs.jsonl.gz、updates.jsonl、training_audit.json、epoch_checkpoints.json 和 frozen.json 保存训练、输入、检查点及校准证据。', ''])
    (report/'REPORT.md').write_text('\n'.join(lines))
    return summary


def main():
    report = ROOT/'reports'/STUDY
    report.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    status = {'state': 'running', 'phase': 'admission', 'started_utc': now(), 'pid': os.getpid(),
              'formal_models_started': 0, 'formal_models_frozen': 0, 'test_accessed': False}
    def save(**updates):
        status.update(updates, updated_utc=now(), seconds=time.monotonic()-start)
        write_json(report/'status.json', status)
    def run(mode, seed, kind, repeat=None, limit=5400):
        remaining = 36000-(time.monotonic()-start)
        if remaining <= 0:
            raise TimeoutError('Ten-hour total budget exceeded')
        command = [sys.executable, '-u', str(ROOT/'scripts/run_laya_head_control.py'),
                   '--mode', mode, '--seed', str(seed), '--head-kind', kind]
        if repeat:
            command.extend(['--repeat', str(repeat)])
            if mode == 'reload':command.append('--admission-reload')
        save(current_run=name(seed, kind, repeat), current_mode=mode)
        print(now(), 'START', ' '.join(command), flush=True)
        process = subprocess.Popen(command, cwd=ROOT, start_new_session=True)
        try:
            code = process.wait(timeout=max(.01, min(limit, remaining)))
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(process.pid, signal.SIGKILL);process.wait()
            raise
        if code:
            raise RuntimeError(f'{mode} {seed} {kind} exited {code}; no automatic retry')
    try:
        busy = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
        if busy:
            raise RuntimeError('GPU already has compute processes: '+busy)
        admission_start = time.monotonic()
        for kind in ('original', 'linear'):
            for repeat in (1, 2):
                for mode in ('admission', 'reload'):
                    available = 1800-(time.monotonic()-admission_start)
                    if available <= 0:raise TimeoutError('Admission exceeded thirty minutes')
                    run(mode, 42, kind, repeat, available)
        admission = check_admission()
        admission['seconds'] = time.monotonic()-admission_start
        write_json(report/'admission.json', admission)
        available = shutil.disk_usage(ROOT).free
        if available < 30*1024**3:
            raise OSError(f'Need 30 GiB free before formal runs; available {available}')
        write_json(report/'formal-storage-admission.json', {'free_bytes': available, 'minimum_bytes': 30*1024**3})
        save(phase='formal-training')
        for index, (seed, kind) in enumerate(PAIRS):
            run_start = time.monotonic()
            save(formal_models_started=index+1)
            run('train', seed, kind)
            run('reload', seed, kind, limit=5400-(time.monotonic()-run_start))
            save(formal_models_frozen=index+1)
        encoder_hashes = set()
        for seed in (42, 43, 44):
            paired = [ROOT/'reports'/name(seed, kind) for kind in ('original', 'linear')]
            audits = [read(p/'training_audit.json') for p in paired]
            if any((a['updates'], a['gradient_exposures']) != (5526, 44200) for a in audits) or audits[0]['input_stream_sha256'] != audits[1]['input_stream_sha256']:
                raise ValueError('Formal paired gradient inputs or update counts differ')
            for p in paired:
                encoder_hashes.add(read(p/'manifest.json')['initial_state_sha256']['encoder'])
        if len(encoder_hashes) != 1:
            raise ValueError('Formal initial encoder states differ')
        run_ids = [name(seed, kind) for seed, kind in PAIRS]
        write_json(report/'all-models-frozen.json', {'run_ids': run_ids,
            'frozen_sha256': {n: file_digest(ROOT/'reports'/n/'frozen.json') for n in run_ids},
            'frozen_utc': now(), 'contract_sha256': file_digest(ROOT/'configs'/f'{STUDY}.json'),
            'development_body_read': False, 'test_accessed': False})
        save(phase='gated-development')
        for seed, kind in PAIRS:run('evaluate', seed, kind)
        save(phase='aggregation')
        aggregate(report)
        save(state='completed', phase='completed', finished_utc=now())
    except BaseException as exc:
        save(state='failed', error=repr(exc), finished_utc=now())
        raise


if __name__ == '__main__':
    main()

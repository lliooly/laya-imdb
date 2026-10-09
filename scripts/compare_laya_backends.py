"""Offline same-checkpoint Laya backend parity and synchronized inference benchmark."""
import argparse
import json
import platform
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

import day2_runtime  # Disable reduced-precision FP32 matmul before MLX import.
import train_laya
train_laya.block_network()
import numpy as np
from common import ROOT, file_digest, read_jsonl, write_json
from comparison_metrics import metrics, sigmoid

OUT = ROOT / 'reports/laya-backend-comparison'
CHECKPOINT = ROOT / 'checkpoints/laya-l512-s42-ce'
KEYS = ('input_ids', 'attention_mask', 'marker_pos', 'marker_mask', 'qtype')


def source():
    path = ROOT / 'data/processed/validation.jsonl'
    split = json.loads((ROOT / 'reports/split_manifest.json').read_text())
    assert file_digest(path) == split['jsonl_sha256']['validation']
    rows = list(read_jsonl(path))
    assert len(rows) == 2500
    return rows


def worker(backend, phase, cache_gib=2):
    rows = source()
    import torch
    import laya.train as t
    from transformers import AutoTokenizer
    cfg = json.loads((CHECKPOINT / 'rl_agent_config.json').read_text())
    tok = AutoTokenizer.from_pretrained(str(CHECKPOINT / 'tokenizer'), local_files_only=True)

    def reference_prepare(chunk):
        typed = [{'state': r['text'], 'questions': {'sentiment': train_laya.QUESTION},
                  'expected': {'sentiment': ('negative', 'positive')[r['label']]}} for r in chunk]
        items, skipped = t.items_from_rows(tok, typed, cfg['max_len'], cfg['head_max_len'])
        assert not skipped and len(items) == len(chunk)
        return [t.encode_item(tok, it, cfg['max_len'], cfg['head_max_len']) for it in items]

    started = time.perf_counter()
    if backend == 'mps-fp32':
        model, tok, _ = t.load_checkpoint(str(CHECKPOINT))
        model.eval().to('mps')
        torch.mps.synchronize()

        def prepare(chunk):
            return reference_prepare(chunk)

        def collate(items):
            return {k: v for k, v in t.collate_items([items], tok.pad_token_id).items() if k in KEYS}

        def forward(batch):
            with torch.inference_mode():
                result = model(*(batch[k].to('mps') for k in KEYS))
                torch.mps.synchronize()
                return result[0].cpu().numpy().astype(np.float64)

        def memory():
            return {'mps_current_allocated_bytes': torch.mps.current_allocated_memory(),
                    'mps_current_driver_bytes': torch.mps.driver_allocated_memory(),
                    'note': 'MPS current values are not peaks and cannot be compared to MLX allocator peak.'}
    else:
        import mlx.core as mx
        if cache_gib:
            mx.set_cache_limit(cache_gib * 1024**3)  # Bound variable-length cache growth.
        from laya_mlx import Agent
        from laya_mlx.agent import collate_items
        agent = Agent(str(CHECKPOINT), dtype='float32' if backend == 'mlx-fp32' else 'float16',
                      device='gpu', compile=False, cache_prompts=False)
        mx.reset_peak_memory()

        def prepare(chunk):
            return [agent.prepare(r['text'], {'sentiment': train_laya.QUESTION})[0][0] for r in chunk]

        def collate(items):
            return collate_items(items, agent.tok.pad_token_id)

        def forward(batch):
            return np.asarray(agent.forward(batch)[0]).astype(np.float64)

        def memory():
            return {'mlx_allocator_peak_since_load_bytes': mx.get_peak_memory(),
                    'mlx_active_bytes': mx.get_active_memory(), 'mlx_cache_bytes': mx.get_cache_memory(),
                    'note': 'MLX allocator values exclude some driver/system allocations.'}
    load_seconds = time.perf_counter() - started
    inputs = prepare(rows)
    reference = reference_prepare(rows)
    for i, (actual, expected) in enumerate(zip(inputs, reference)):
        for k in ('ids', 'markers', 'qtype'):
            assert actual[k] == expected[k], (backend, i, k)
    # Verify padded batch values too; integer widths differ between runtimes.
    for start in range(0, len(rows), 8):
        a = collate(inputs[start:start+8])
        b = t.collate_items([reference[start:start+8]], tok.pad_token_id)
        for k in KEYS:
            assert np.array_equal(np.asarray(a[k]), b[k].numpy()), (backend, start, k)
    print(backend, 'all 2500 prepared inputs match upstream', flush=True)
    meta = {'backend': backend, 'phase': phase, 'load_seconds': load_seconds,
            'identical_inputs': 2500, 'native_temperature': cfg['temperature'][0],
            'compiled': False, 'tf32': False,
            'mlx_cache_limit_bytes': cache_gib * 1024**3 if backend.startswith('mlx') and cache_gib else None, 'checkpoint_sha256': file_digest(CHECKPOINT / 'model.safetensors'),
            'source_sha256': file_digest(ROOT / 'data/processed/validation.jsonl'),
            'versions': {name: version(name) for name in ('laya', 'laya-mlx', 'mlx', 'torch', 'transformers', 'numpy')},
            'platform': platform.platform(), 'machine': platform.machine()}
    if phase == 'eval':
        z = []
        start_time = time.perf_counter()
        with (OUT / (backend + '-predictions.jsonl')).open('w') as f:
            for start in range(0, len(rows), 8):
                logits = forward(collate(inputs[start:start+8]))
                assert np.isfinite(logits).all()
                z.extend(logits.tolist())
                for r, v in zip(rows[start:start+8], logits):
                    f.write(json.dumps({'sample_id': r['id'], 'true_label': r['label'], 'logits': v.tolist(),
                                        'prob_positive': float(sigmoid((v[1]-v[0])/cfg['temperature'][0])),
                                        'pred_label': int(np.argmax(v))})+'\n')
                if start % 400 == 0:
                    print(backend, min(start+8, len(rows)), '/2500', round(time.perf_counter()-start_time, 1), 's', flush=True)
        z = np.asarray(z)
        meta['evaluation_elapsed_seconds'] = time.perf_counter()-start_time
        meta['raw_metrics'] = metrics([r['label'] for r in rows], z[:, 1]-z[:, 0])
        meta['native_calibrated_metrics'] = metrics([r['label'] for r in rows], z[:, 1]-z[:, 0], cfg['temperature'][0])
    else:
        lengths = np.asarray([len(it['ids']) for it in inputs])
        groups = {'short': np.flatnonzero(lengths <= 128),
                  'medium': np.flatnonzero((lengths > 128) & (lengths <= 320)),
                  'long': np.flatnonzero(lengths == 512)}
        results = []
        for group, indices in groups.items():
            assert len(indices) >= 8
            selected = indices[np.linspace(0, len(indices)-1, 8, dtype=int)].tolist()
            for size in (1, 8):
                for mode in ('prepared_forward', 'end_to_end'):
                    # Rotate through real comments; all backends use identical IDs/order.
                    chunks = [[selected[(i*size+j) % 8] for j in range(size)] for i in range(55)]
                    batches = [collate([inputs[j] for j in chunk]) for chunk in chunks]
                    samples = []
                    for i, (chunk, batch) in enumerate(zip(chunks, batches)):
                        began = time.perf_counter()
                        if mode == 'end_to_end':
                            batch = collate(prepare([rows[j] for j in chunk]))
                        logits = forward(batch)
                        # Include common native-temperature softmax/result materialization.
                        probs = sigmoid((logits[:, 1]-logits[:, 0])/cfg['temperature'][0])
                        _ = [{'sample_id': rows[j]['id'], 'prob_positive': float(p), 'pred_label': int(p > .5)}
                             for j, p in zip(chunk, probs)]
                        elapsed = (time.perf_counter()-began)*1000
                        if i >= 5:
                            samples.append(elapsed)
                    results.append({'length_group': group, 'batch_size': size, 'mode': mode,
                                    'sample_ids': [rows[j]['id'] for j in selected],
                                    'token_lengths': lengths[selected].tolist(), 'warmup': 5, 'repetitions': 50,
                                    'milliseconds': samples, 'p50_ms': float(np.median(samples)),
                                    'p95_ms': float(np.percentile(samples, 95)),
                                    'reviews_per_second': float(size*1000/np.mean(samples))})
                    print(backend, group, size, mode, round(np.median(samples), 2), 'ms', flush=True)
        meta['benchmark'] = results
    meta['memory'] = memory()
    write_json(OUT / (backend+'-'+phase+'.json'), meta)


def summarize():
    backends = ('mps-fp32', 'mlx-fp32', 'mlx-fp16')
    records = {b: list(read_jsonl(OUT / (b+'-predictions.jsonl'))) for b in backends}
    ref = records['mps-fp32']
    old = list(read_jsonl(ROOT / 'reports/model-comparison/laya_ce_full/raw.jsonl'))
    assert [r['sample_id'] for r in ref] == [r['sample_id'] for r in old]
    assert all(r['pred_label'] == int(np.argmax(o['logits'])) for r, o in zip(ref, old))
    comparisons = {}
    for b, rows in records.items():
        assert [r['sample_id'] for r in rows] == [r['sample_id'] for r in ref]
        a = np.array([r['logits'] for r in ref]); z = np.array([r['logits'] for r in rows])
        p = np.array([r['prob_positive'] for r in rows]); pr = np.array([r['prob_positive'] for r in ref])
        comparisons[b] = {'prediction_disagreements': sum(r['pred_label'] != s['pred_label'] for r,s in zip(rows,ref)),
                          'max_abs_logit_difference': float(np.max(np.abs(a-z))),
                          'max_abs_calibrated_probability_difference': float(np.max(np.abs(pr-p))),
                          'mean_abs_calibrated_probability_difference': float(np.mean(np.abs(pr-p)))}
    write_json(OUT/'parity.json', comparisons)
    eval_cache = json.loads((OUT/'mlx-fp32-eval.json').read_text()).get('mlx_cache_limit_bytes')
    cache_description = f'数值核对 MLX cache limit: {eval_cache or "runtime default"}；分组测速 cache limit 见各 backend 的 bench JSON。'
    lines = ['# Laya-CE 同权重推理对照', '',
             '同一份微调权重、相同 2,500 条开发验证样本；未读取测试集、未重新训练。输入 token、mask、选项顺序和位置逐条核对。', '',
             'FP32 关闭 MLX TF32；全部 eager、无 prompt cache。温度使用原 checkpoint 值，没有重新拟合。复现脚本默认两个阶段均使用 2 GiB cache。', cache_description, '',
             '| Backend | Accuracy | Macro-F1 | 预测差异（对 MPS） | 最大校准概率差 |',
             '|---|---:|---:|---:|---:|']
    for b in backends:
        m = json.loads((OUT/(b+'-eval.json')).read_text())['raw_metrics']; c = comparisons[b]
        lines.append(f"| {b} | {m['accuracy']:.4%} | {m['macro_f1']:.6f} | {c['prediction_disagreements']} | {c['max_abs_calibrated_probability_difference']:.6g} |")
    lines += ['', '## 延迟和吞吐', '',
              '每个后端/阶段独立进程，顺序运行。每组 8 条真实影评，5 次预热、50 次同步计时；长度为包含问题与选项后的实际序列长度。P50 为每批延迟，吞吐为 batch_size / 平均耗时。', '',
              'prepared_forward 包括 CPU 输入转设备、模型、同步、输出转 CPU 和公共结果格式化；end_to_end 另加各后端的分词、模板及 padding。两者均不含模型加载和首次预热。', '',
              '| Length | Batch | Scope | MPS FP32 ms | MLX FP32 ms | MLX FP16 ms | FP32 speedup | FP16 speedup* |',
              '|---|---:|---|---:|---:|---:|---:|---:|']
    bench = {b: json.loads((OUT/(b+'-bench.json')).read_text())['benchmark'] for b in backends}
    for entries in zip(*(bench[b] for b in backends)):
        a,x,h = entries
        lines.append(f"| {a['length_group']} | {a['batch_size']} | {a['mode']} | {a['p50_ms']:.2f} | {x['p50_ms']:.2f} | {h['p50_ms']:.2f} | {a['p50_ms']/x['p50_ms']:.2f}× | {a['p50_ms']/h['p50_ms']:.2f}× |")
    lines += ['', '*FP16 加速同时包含框架与精度变化，不能全部归因为 MLX。原始 JSON 保留 P95、吞吐及全部计时。', '',
              '## 缓存诊断', '',
              '初始 MLX 默认设置全量评估结束时缓存超过 51 GB；原始结果保存在 default-cache-diagnostic。它与 FP32 评估变慢同时出现，但未做受控全量重复实验，不能单独证明缓存是原因。分组测速另设 2 GiB 缓存上限；不把默认设置评估耗时与受限缓存分组测速混为同一实验。', '',
              '## 限制', '',
              '- 单设备单次运行，受温度、系统负载和运行顺序影响，不能当作跨设备结论。',
              '- MLX allocator peak 与 MPS current driver memory 口径不同，仅记录，不比较内存优劣。',
              '- 这是推理对照，不能推断训练速度。该 MLX 包当前不提供微调入口。',
              '- 验证集此前参与开发与 checkpoint 选择，准确率不是最终测试结果。', '',
              '复现：`uv run --offline --locked --extra laya --extra metal --extra laya-mlx python -u scripts/compare_laya_backends.py`', '']
    import csv
    with (OUT/'benchmark.csv').open('w') as f:
        fields = ['backend', 'length_group', 'batch_size', 'mode', 'p50_ms', 'p95_ms', 'reviews_per_second']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for b in backends:
            for row in bench[b]:
                writer.writerow({'backend': b, **{k: row[k] for k in fields if k != 'backend'}})
    for b in backends:
        with (OUT/(b+'-predictions.csv')).open('w') as f:
            fields = ['sample_id', 'true_label', 'prob_positive', 'pred_label']
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows({k: row[k] for k in fields} for row in records[b])
    write_json(OUT/'manifest.json', {
        'hardware': json.loads((ROOT/'reports/environment.json').read_text())['hardware'],
        'validation_sha256': file_digest(ROOT/'data/processed/validation.jsonl'),
        'checkpoint_sha256': file_digest(CHECKPOINT/'model.safetensors'),
        'uv_lock_sha256': file_digest(ROOT/'uv.lock'),
        'script_sha256': file_digest(Path(__file__)),
        'test_accessed': False, 'training_performed': False,
        'numerical_evaluation_cache_limit_bytes': eval_cache,
        'benchmark_cache_limits': {b: json.loads((OUT/(b+'-bench.json')).read_text()).get('mlx_cache_limit_bytes') for b in backends},
        'execution_order': ['eval/'+b for b in backends]+['bench/'+b for b in backends],
        'note': 'Initial full evaluation used runtime-default MLX cache. Benchmark used 2 GiB; final reproduction script defaults to 2 GiB in both phases.'})
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    write_json(OUT/'checksums.json', {str(p.relative_to(OUT)): file_digest(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'checksums.json'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mlx-cache-gib', type=int, default=2, help='MLX cache limit; 0 keeps runtime default')
    parser.add_argument('--summarize-only', action='store_true')
    parser.add_argument('--backend', choices=('mps-fp32','mlx-fp32','mlx-fp16'))
    parser.add_argument('--phase', choices=('eval','bench'))
    args = parser.parse_args()
    if args.mlx_cache_gib < 0:
        parser.error('cache limit must be nonnegative')
    OUT.mkdir(parents=True, exist_ok=True)
    if args.backend and not args.phase:
        parser.error('--backend requires --phase')
    if args.summarize_only:
        summarize()
    elif args.backend:
        worker(args.backend, args.phase, args.mlx_cache_gib)
    else:
        for phase in ('eval','bench'):
            for backend in ('mps-fp32','mlx-fp32','mlx-fp16'):
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--backend', backend, '--phase', phase, '--mlx-cache-gib', str(args.mlx_cache_gib)], check=True)
        summarize()


if __name__ == '__main__':
    main()

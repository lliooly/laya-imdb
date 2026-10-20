"""Supplement frozen paper evidence without changing historical runs or fitting."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from comparison_metrics import metrics, paired, sigmoid

ROOT = Path(__file__).resolve().parents[1]
FINAL = 'reports/deberta-laya-final-cpu-evaluation-20261018-v1'
TRAIN = 'deberta-heterogeneous-bf16-budget7-s42-20261017-v1'
TEMPERATURE = 1.1374971945707235
MODELS = ('original_laya_main', 'deberta_cpu', 'fusion_50_50')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def probability_metrics(y, p):
    p = np.asarray(p, dtype=float)
    if p.shape != np.asarray(y).shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid aligned probabilities')
    p = np.clip(p, 1e-15, 1-1e-15)
    return metrics(y, np.log(p)-np.log1p(-p))


def text_groups(records):
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[row['text']].append(i)
    if any(len({records[i]['label'] for i in ix}) != 1 for ix in groups.values()):
        raise ValueError('Conflicting labels need a separate, predeclared group policy')
    return sorted(groups.values(), key=lambda ix: min(records[i]['id'] for i in ix))


def representative_indices(records, groups):
    return np.array([min(ix, key=lambda i: records[i]['id']) for ix in groups], dtype=int)


def cluster_interval(records, groups, a_correct, b_correct, replicates=20000, seed=20261019):
    """Resample text groups within label strata; retain original row weights."""
    if not groups:
        raise ValueError('Empty cohort')
    rng = np.random.default_rng(seed)
    numerators = np.zeros(replicates, dtype=np.int64)
    denominators = np.zeros(replicates, dtype=np.int64)
    delta = np.asarray(b_correct, dtype=int)-np.asarray(a_correct, dtype=int)
    for label in (0, 1):
        categories = Counter((len(ix), int(delta[ix].sum())) for ix in groups if records[ix[0]]['label'] == label)
        if not categories:
            continue
        values = list(categories)
        counts = np.array(list(categories.values()))
        draws = rng.multinomial(int(counts.sum()), counts/counts.sum(), size=replicates)
        denominators += draws @ np.array([size for size, _ in values])
        numerators += draws @ np.array([value for _, value in values])
    return dict(lower=float(np.quantile(numerators/denominators, .025)),
                upper=float(np.quantile(numerators/denominators, .975)),
                unit='cleaned_text_group', weighting='original_rows', label_stratified=True,
                replicates=replicates, seed=seed, group_count=len(groups),
                conditions='fixed models/calibration; excludes selection and training uncertainty')


def analyze(out):
    out.mkdir(parents=True, exist_ok=False)
    inputs = [f'{FINAL}/test-fused-predictions.json', 'data/processed/train.jsonl',
              'data/processed/validation.jsonl', 'data/processed/test.jsonl',
              'docs/superpowers/specs/2026-10-19-paper-supplementary-validation-design.md',
              'scripts/paper_supplementary_validation.py', 'scripts/comparison_metrics.py']
    hashes = {p: digest(ROOT/p) for p in inputs}
    pred = read(ROOT/inputs[0]); test = rows(ROOT/'data/processed/test.jsonl')
    train_text = {r['text'] for p in ('train', 'validation') for r in rows(ROOT/f'data/processed/{p}.jsonl')}
    if pred['sample_ids'] != [r['id'] for r in test] or pred['true_labels'] != [r['label'] for r in test]:
        raise ValueError('Prediction IDs/labels differ from original source')
    y = np.array(pred['true_labels']); probabilities = {k: np.array(pred['positive_probabilities'][k]) for k in MODELS}
    groups = text_groups(test); unique = representative_indices(test, groups)
    clean = np.array([i for i, r in enumerate(test) if r['text'] not in train_text])
    clean_set = set(clean.tolist()); clean_groups = [ix for ix in groups if ix[0] in clean_set]
    clean_unique = representative_indices(test, clean_groups)
    cohorts = {'official_full': np.arange(len(test)), 'exclude_training_pool_overlap': clean,
               'unique_test_text': unique, 'exclude_overlap_and_unique': clean_unique}
    result = dict(run_id=out.name, original_test_previously_accessed=True,
                  cached_test_predictions_and_text_read=True, new_test_forward_pass=False,
                  new_training=False, new_fitting=False, source_sha256=hashes,
                  excluded_test_ids=[r['id'] for i, r in enumerate(test) if i not in clean_set],
                  unique_text_representative='lexicographically minimum sample_id, independent of labels/predictions',
                  cohorts={})
    for name, ix in cohorts.items():
        entry = dict(count=len(ix), labels=dict(Counter(map(int, y[ix]))),
                     metrics={k: probability_metrics(y[ix], p[ix]) for k, p in probabilities.items()}, paired={})
        for base in MODELS[:2]:
            comparison = paired(y[ix], probabilities[base][ix]-.5, probabilities['fusion_50_50'][ix]-.5)
            comparison['net_correct'] = comparison['b_only_correct']-comparison['a_only_correct']
            comparison['interpretation'] = 'descriptive; historical selection remains; duplicate rows are not independent'
            entry['paired']['fusion_vs_'+base] = comparison
        if name in ('official_full', 'exclude_training_pool_overlap'):
            entry['cluster_intervals'] = {base: cluster_interval(test, groups if name == 'official_full' else clean_groups,
                (probabilities[base] > .5) == y, (probabilities['fusion_50_50'] > .5) == y) for base in MODELS[:2]}
        result['cohorts'][name] = entry
    if not all(digest(ROOT/p) == sha for p, sha in hashes.items()):
        raise ValueError('Evidence changed during analysis')
    write(out/'summary.json', result)
    report = ['# 冻结官方测试的重复文本敏感性分析', '',
              '全部结果复用原预测。排除与完整原官方训练文本相同的测试记录；不改标签，不新训练或拟合。', '',
              '| 口径 | n | Laya | DeBERTa | 固定融合 | 融合−DeBERTa 净正确数 |',
              '|---|---:|---:|---:|---:|---:|']
    titles = ['完整官方测试', '排除训练侧同文', '测试按唯一文本计分', '排除同文且唯一化']
    for title, (name, entry) in zip(titles, result['cohorts'].items()):
        m = entry['metrics']; report.append('| '+title+' | '+str(entry['count'])+' | '+
            ' | '.join(f'{m[k]["accuracy"]:.4%}' for k in MODELS)+' | '+
            str(entry['paired']['fusion_vs_deberta_cpu']['net_correct'])+' |')
    report += ['', '## 按文本组重采样的配对区间', '',
               '抽样保持组内记录和模型间配对；分母随被抽中组的大小变化。百分位区间固定已训练模型和校准参数。', '']
    for name in ('official_full', 'exclude_training_pool_overlap'):
        for base, ci in result['cohorts'][name]['cluster_intervals'].items():
            report.append(f'- {name}，fusion − {base}：95% 区间 [{ci["lower"]*100:+.4f}, {ci["upper"]*100:+.4f}] 个百分点。')
    report += ['', '这些子集来自已经使用的官方测试，不是新独立测试。去重未消除历史反馈、模型选择偏差或训练方差。',
               '代表记录由 sample ID 固定，未按概率、标签是否正确或效果择优。完整官方成绩保持原口径。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    print('\n'.join(report), flush=True)


def cpu_parity(out, resume=False):
    model_path = ROOT/'checkpoints'/TRAIN/'epoch-002'
    contract_path = out/'contract.json'
    if not resume:
        out.mkdir(parents=True, exist_ok=False)
        from transformers import AutoTokenizer
        from deberta_inputs import encode
        source = rows(ROOT/'data/processed/validation.jsonl')
        ref = read(ROOT/'reports'/TRAIN/'evaluation/development.json')
        if [r['id'] for r in source] != ref['sample_ids'] or [r['label'] for r in source] != ref['true_labels']:
            raise ValueError('Development identities differ')
        tok = AutoTokenizer.from_pretrained(str(model_path/'assets/tokenizer'), local_files_only=True, use_fast=False)
        items = []
        for row in source:
            if hashlib.sha256(row['text'].encode()).hexdigest() != row['text_sha256']:
                raise ValueError('Text digest differs')
            item = encode(tok, row['text']); item.update(sample_id=row['id'], label=row['label'])
            item['input_sha256'] = identity(item['input_ids']); items.append(item)
        files = ['data/processed/validation.jsonl', f'reports/{TRAIN}/evaluation/development.json',
                 'reports/laya-probability-fusion-s43-s42-20261014-v1/validation-fused.json',
                 'scripts/paper_supplementary_validation.py', 'scripts/deberta_training.py', 'scripts/deberta_inputs.py',
                 'scripts/comparison_metrics.py', str((model_path/'model.safetensors').relative_to(ROOT)),
                 str((model_path/'config.json').relative_to(ROOT))]
        files += [str(p.relative_to(ROOT)) for p in sorted((model_path/'assets/tokenizer').iterdir()) if p.is_file()]
        sources = {p: digest(ROOT/p) for p in files}
        historical = read(ROOT/'configs/deberta-laya-final-cpu-evaluation-20261018-v1.json')['sources_sha256']
        weight = str((model_path/'model.safetensors').relative_to(ROOT))
        if sources[weight] != historical[weight]:
            raise ValueError('Checkpoint is not the frozen paper model')
        contract = dict(run_id=out.name, count=2500, temperature=TEMPERATURE, threads=12, batch_size=8,
                        backend='cpu_fp32', test_accessed=False, new_fitting=False, new_training=False,
                        source_sha256=sources, pad_token_id=tok.pad_token_id, inputs_sha256=identity(items))
        write(out/'inputs.json', items); write(contract_path, contract)
    contract = read(contract_path); contract_sha = digest(contract_path); items = read(out/'inputs.json')
    if identity(items) != contract['inputs_sha256'] or not all(digest(ROOT/p) == v for p, v in contract['source_sha256'].items()):
        raise ValueError('Frozen inputs or sources changed')
    import torch
    from deberta_training import seed_all, load_saved, predict
    torch.set_num_threads(contract['threads']); seed_all()
    model = load_saved(model_path)
    order = sorted(range(len(items)), key=lambda i: len(items[i]['input_ids']))
    logits = [None]*len(items); started = time.monotonic()
    for offset in range(0, len(order), 64):
        indices = order[offset:offset+64]; part = [items[i] for i in indices]
        chunk_key = identity([contract_sha, [(r['sample_id'], r['input_sha256']) for r in part]])
        path = out/'chunks'/f'{offset:06d}.json'
        if path.exists():
            saved = read(path)
            if saved['identity'] != chunk_key or identity(saved['logits']) != saved['content_sha256']:
                raise ValueError('Invalid saved inference chunk')
            z = saved['logits']
        else:
            z, seconds = predict(model, part, contract['pad_token_id'], torch.device('cpu'), batch_size=8)
            write(path, dict(identity=chunk_key, logits=z, seconds=seconds, content_sha256=identity(z)))
        if np.asarray(z).shape != (len(part), 2) or not np.isfinite(z).all():
            raise ValueError('Invalid output logits')
        for i, row in zip(indices, z, strict=True): logits[i] = row
        progress = dict(state='running', completed=min(offset+64, len(order)), total=len(order),
                        elapsed_seconds=time.monotonic()-started, test_accessed=False, contract_sha256=contract_sha)
        write(out/'status.json', progress)
        print('CPU_PARITY', progress['completed'], '/', len(order), flush=True)
    actual = np.array(logits); ref = read(ROOT/'reports'/TRAIN/'evaluation/development.json')
    if [r['sample_id'] for r in items] != ref['sample_ids']:
        raise ValueError('Final output order differs')
    expected = np.array(ref['logits']); y = np.array(ref['true_labels'])
    a, b = sigmoid((expected[:, 1]-expected[:, 0])/TEMPERATURE), sigmoid((actual[:, 1]-actual[:, 0])/TEMPERATURE)
    laya = read(ROOT/'reports/laya-probability-fusion-s43-s42-20261014-v1/validation-fused.json')['rows']
    if [r['sample_id'] for r in laya] != ref['sample_ids']:
        raise ValueError('Laya IDs differ')
    lp = np.array([r['probabilities'][1] for r in laya])
    compared = dict(cuda_bf16=a, cpu_fp32=b, fusion_cuda_bf16=(lp+a)/2, fusion_cpu_fp32=(lp+b)/2)
    result = dict(run_id=out.name, count=len(y), temperature=TEMPERATURE, source_sha256=contract['source_sha256'],
        test_accessed=False, new_fitting=False, new_training=False, contract_sha256=contract_sha,
        max_abs_logit_difference=float(abs(expected-actual).max()),
        max_abs_probability_difference=float(abs(a-b).max()), mean_abs_probability_difference=float(abs(a-b).mean()),
        class_disagreement_count=int(((a>.5) != (b>.5)).sum()),
        fusion_disagreement_count=int((((lp+a)/2>.5) != ((lp+b)/2>.5)).sum()),
        metrics={k: probability_metrics(y, p) for k, p in compared.items()},
        discordant_ids=[sid for sid, p, q in zip(ref['sample_ids'], a, b) if (p>.5) != (q>.5)],
        elapsed_seconds=time.monotonic()-started, scope='development parity; not exact equivalence or independent confirmation')
    if not all(digest(ROOT/p) == v for p, v in contract['source_sha256'].items()):
        raise ValueError('Evidence changed during inference')
    write(out/'predictions.json', dict(sample_ids=ref['sample_ids'], logits=logits, contract_sha256=contract_sha))
    write(out/'summary.json', result); write(out/'status.json', dict(state='completed', count=2500, test_accessed=False))
    report = ['# DeBERTa 完整开发集 CPU FP32 / CUDA BF16 核验', '',
        f'固定同一模型和温度，开发样本 n=2500；类别差异 {result["class_disagreement_count"]} 条，融合类别差异 {result["fusion_disagreement_count"]} 条。',
        f'最大概率绝对差 {result["max_abs_probability_difference"]:.8f}；平均绝对差 {result["mean_abs_probability_difference"]:.8f}。', '',
        '| 方案 | 正确数 | Accuracy | NLL | ECE |', '|---|---:|---:|---:|---:|']
    for name, value in result['metrics'].items():
        report.append(f'| {name} | {value["correct"]} | {value["accuracy"]:.4%} | {value["nll"]:.8f} | {value["ece_15_equal_width_top_label"]:.4%} |')
    report += ['', '未新训练或重新校准，未执行官方测试前向。核验扩展到完整开发集，不意味着不同后端逐位等价。']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n'); print('\n'.join(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['analyze', 'cpu-parity'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.mode == 'analyze':
        if args.resume: parser.error('Analysis is immutable; use a new output directory')
        analyze(args.output)
    else:
        cpu_parity(args.output, args.resume)

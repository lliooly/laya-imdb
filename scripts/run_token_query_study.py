"""Frozen full-token mean/dense/top32 readout experiment with sealed evaluation."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from common import ROOT, file_digest, read_jsonl, write_json
from comparison_metrics import fit_temperature, metrics, sigmoid

KINDS = ('mean', 'dense', 'top32')

RUN = 'laya-token-query-mps-fp32-20261018-v1'
OUT, ART = ROOT/'reports'/RUN, ROOT/'artifacts'/RUN
CONFIG = ROOT/'configs'/(RUN+'.json')
OLD = 'laya-frozen-probe-baseline-s43-mps-fp32-20261015-v1'
READS = set()


def read(path):
    return json.loads(Path(path).read_text())


def save(name, value):
    write_json(OUT/name, value)


def emit(stage, **kwargs):
    print(json.dumps(dict(stage=stage, **kwargs)), flush=True)


def now():
    return datetime.now(timezone.utc).isoformat()


def guard(stage):
    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).absolute()
        mode, flags = args[1], args[2]
        reading = ('r' in mode or '+' in mode) if isinstance(mode, str) else not (flags & os.O_WRONLY)
        if not reading:
            return
        name = str(path)
        if any(s in name for s in ('/data/raw/', '/data/processed/test', '/laya-frozen-official-test-', '/deberta-laya-final-cpu-evaluation-')):
            raise PermissionError('Official test prohibited: '+name)
        if stage in ('extract', 'train') and path.suffix in ('.json', '.jsonl', '.npz', '.npy') and (
            path.name.startswith('validation') or path.name.endswith('-development.json') or
            path.name in ('predictions.json', 'development.json', 'development-heads.json')):
            raise PermissionError('Development prohibited before freezing: '+name)
        if name.startswith(str(ROOT)+'/') and '/.venv/' not in name:
            READS.add(str(path.relative_to(ROOT)))
    sys.addaudithook(audit)


def config():
    c = read(CONFIG)
    assert c['run_id'] == RUN and c['kinds'] == list(KINDS)
    assert c['seeds'] == [42, 43, 44] and c['epochs'] == 5 and c['batch_size'] == 32
    assert c['learning_rate'] == .0003 and c['weight_decay'] == .01
    assert c['width'] == 128 and c['dropout'] == .1 and c['primary'] == 'dense'
    return c


def source_spec():
    return read(ROOT/'configs'/(OLD+'.json'))


def verify_bindings():
    p = read(OUT/'protocol.json')
    for path, sha in p['source_sha256'].items():
        assert file_digest(ROOT/path) == sha, path
    return p


def old_cache(split):
    meta = read(ROOT/'reports'/OLD/(split+'-extraction.json'))
    cache = ROOT/meta['cache']
    index = cache.with_name(split+'-index.json')
    assert file_digest(cache) == meta['cache_sha256']
    assert file_digest(index) == meta['index_sha256']
    records = read(index)['rows']
    with np.load(cache, allow_pickle=False) as z:
        result = {k: z[k].copy() for k in ('encoder', 'labels', 'truncated', 'full_logits')}
    assert [r['true_label'] for r in records] == result['labels'].tolist()
    return records, result


def load_source():
    import laya.train as upstream
    import train_laya
    spec = source_spec()
    train_laya.block_network()
    for path, sha in spec['asset_sha256'].items():
        assert file_digest(ROOT/spec['checkpoint']/path) == sha
    model, tok, cfg = upstream.load_checkpoint(str(ROOT/spec['checkpoint']))
    assert (cfg['max_len'], cfg['head_max_len']) == (512, 192)
    assert not upstream.uses_parallel_layout(cfg)
    model.float().eval().requires_grad_(False).to('mps')
    assert all(not p.requires_grad and torch.isfinite(p).all() for p in model.parameters())
    return upstream, model, tok


def extraction(upstream, model, tok, rows, split, batch_size):
    from export_laya_posthoc import input_digest, prepare_rules
    if (ART/(split+'-tokens.npy')).exists():
        raise FileExistsError('Partial/completed cache exists; preserve it')
    records, original = old_cache(split)
    assert [r['id'] for r in rows] == [r['sample_id'] for r in records]
    assert [r['label'] for r in rows] == original['labels'].tolist()
    _, tail, _, _, _ = prepare_rules(tok, [dict(row, label=0) for row in rows])
    prepared = [upstream.encode_item(tok, item, 512, 192, [0, 1], parallel=False) for item in tail]
    assert all(input_digest(item) == r['input_sha256'] for item, r in zip(prepared, records, strict=True))
    lengths = np.asarray([len(x['ids']) for x in prepared], dtype=np.int64)
    offsets = np.concatenate(([0], lengths.cumsum()))
    required = int(offsets[-1])*1024*4
    if shutil.disk_usage(ART).free < required+8*1024**3:
        raise OSError('Insufficient disk space for lossless token cache')
    path = ART/(split+'-tokens.npy')
    tokens = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32, shape=(int(offsets[-1]), 1024))
    markers = np.asarray([x['markers'] for x in prepared], dtype=np.int64)
    started, maximum = time.perf_counter(), 0.
    # Match historical batch size to minimize backend shape-dependent differences.
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chunk = prepared[start:start+batch_size]
            batch = upstream.collate_items([chunk], tok.pad_token_id)
            ids, mask = batch['input_ids'].to('mps'), batch['attention_mask'].to('mps')
            m = batch['marker_pos'].to('mps')
            assert batch['marker_mask'].all() and (torch.gather(ids, 1, m) == tok.mask_token_id).all()
            h = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state.float().cpu().numpy()
            if not np.isfinite(h).all():
                raise FloatingPointError('Nonfinite encoder output')
            for j, item in enumerate(chunk):
                row = start+j
                a, b = offsets[row:row+2]
                tokens[a:b] = h[j, :b-a]
                delta = float(np.max(np.abs((h[j, markers[row, 1]]-h[j, markers[row, 0]])-original['encoder'][row])))
                maximum = max(maximum, delta)
            if maximum > .001:
                raise ValueError('Historical marker parity failed: '+str(maximum))
            if start % 400 == 0 or start+batch_size >= len(rows):
                tokens.flush()
                progress = dict(split=split, rows=min(start+batch_size, len(rows)), count=len(rows),
                                seconds=time.perf_counter()-started, max_marker_error=maximum)
                save('progress.json', progress)
                emit('extract', **progress)
    tokens.flush()
    del tokens
    index = dict(split=split, sample_ids=[r['id'] for r in rows], labels=original['labels'].tolist(),
        offsets=offsets.tolist(), markers=markers.tolist(), truncated=original['truncated'].tolist(),
        input_sha256=[r['input_sha256'] for r in records], token_count=int(offsets[-1]), dtype='float32',
        shape=[int(offsets[-1]), 1024], max_marker_error=maximum, seconds=time.perf_counter()-started,
        cache_sha256=file_digest(path), index_source_sha256=file_digest(ROOT/'artifacts'/OLD/(split+'-index.json')),
        test_accessed=False)
    write_json(ART/(split+'-index.json'), index)
    save(split+'-extraction.json', {k:v for k,v in index.items() if k not in ('sample_ids','labels','offsets','markers','input_sha256','truncated')})
    emit('cache_complete', split=split, seconds=index['seconds'], bytes=required)


def extract():
    from run_laya_frozen_probe import admit_training
    spec, c = source_spec(), config()
    if OUT.exists() or ART.exists():
        raise FileExistsError('Preserve historical/partial run')
    OUT.mkdir(parents=True)
    ART.mkdir(parents=True)
    sources = ['configs/'+RUN+'.json', 'scripts/run_token_query_study.py', 'scripts/token_query_head.py',
        'scripts/export_laya_posthoc.py', 'scripts/comparison_metrics.py', 'scripts/common.py',
        'scripts/bow_nb_branch.py', 'scripts/run_laya_frozen_probe.py', 'scripts/train_laya.py',
        'uv.lock', 'configs/'+OLD+'.json', spec['manifest'], 'reports/split_manifest.json', 'data/processed/train.jsonl']
    hashes = {p:file_digest(ROOT/p) for p in sources}
    manifest, ids, selected = admit_training(spec)
    save('protocol.json', dict(config=c, created_utc=now(), source_sha256=hashes, checkpoint=spec['checkpoint'],
        checkpoint_sha256=spec['asset_sha256']['model.safetensors'],
        versions={k:version(k) for k in ('torch','transformers','numpy','laya')},
        interpretation='Frozen fine-tuned encoder pilot; not end-to-end or DSA replication',
        device='mps', precision='FP32 cache/parameters/compute, no AMP', primary_fixed_before_development='dense',
        partition_counts=[22100,400,2500], official_test_access_allowed=False, historical_development_reused=True,
        development_data_sha256=spec['source_sha256']['data/processed/validation.jsonl']))
    repro = OUT/'reproduction'
    repro.mkdir()
    for source in sources:
        if source.startswith('scripts/') or source == 'configs/'+RUN+'.json':
            shutil.copyfile(ROOT/source, repro/Path(source).name)
    upstream, model, tok = load_source()
    for split in ('calibration', 'gradient_train'):
        extraction(upstream, model, tok, selected[split], split, c['extraction_batch_size'])
    save('extraction-complete.json', dict(completed_utc=now(), splits=['gradient_train','calibration'], test_accessed=False))


def load_tokens(split, verify=True):
    index = read(ART/(split+'-index.json'))
    path = ART/(split+'-tokens.npy')
    if verify:
        assert file_digest(path) == index['cache_sha256']
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    assert list(values.shape) == index['shape'] and values.dtype == np.float32
    return values, index


def batch(values, index, indices, device='mps'):
    offsets = index['offsets']
    lengths = [offsets[i+1]-offsets[i] for i in indices]
    h = np.zeros((len(indices), max(lengths), values.shape[-1]), dtype=np.float32)
    mask = np.zeros(h.shape[:2], dtype=bool)
    for j, i in enumerate(indices):
        h[j, :lengths[j]] = values[offsets[i]:offsets[i+1]]
        mask[j, :lengths[j]] = True
    markers = np.asarray(index['markers'])[indices]
    return torch.from_numpy(h).to(device), torch.from_numpy(mask).to(device), torch.from_numpy(markers).to(device)


def predict(model, values, index, batch_size=32):
    model.eval()
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(index['labels']), batch_size):
            ii = np.arange(start, min(start+batch_size, len(index['labels'])))
            outputs.append(model(*batch(values, index, ii)).float().cpu().numpy())
    return np.concatenate(outputs)


def init(kind, seed, c):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return TokenQueryHead(kind, width=c['width'], dropout=c['dropout']).float()


def fit_mixture(a, b, y):
    delta = b-a
    def derivative(w):
        p = np.clip(a+w*delta, 1e-12, 1-1e-12)
        return float(np.mean((p-y)*delta/(p*(1-p))))
    lo, hi = 0., .25
    if derivative(lo) >= 0:
        w = lo
    elif derivative(hi) <= 0:
        w = hi
    else:
        for _ in range(80):
            mid = (lo+hi)/2
            if derivative(mid) > 0:
                hi = mid
            else:
                lo = mid
        w = (lo+hi)/2
    return dict(sparse_weight=w, bounds=[0., .25], objective='calibration400_mean_NLL',
                derivative_at_solution=derivative(w))


def train():
    from safetensors.torch import load_file, save_file
    from run_final_cpu_alternatives_v2 import cal_neural
    from laya_head_control import state_digest
    c, protocol = config(), verify_bindings()
    if (OUT/'heads-frozen.json').exists():
        raise FileExistsError('Heads already frozen')
    assert (OUT/'extraction-complete.json').exists()
    tx, ti = load_tokens('gradient_train')
    cx, ci = load_tokens('calibration')
    assert len(ti['labels']) == 22100 and len(ci['labels']) == 400
    assert not set(ti['sample_ids']) & set(ci['sample_ids'])
    pa, pb, cy, cal_ids = cal_neural()
    assert ci['sample_ids'] == cal_ids and np.array_equal(cy, ci['labels'])
    all_results, records, initializations = {}, {}, {}
    labels = np.asarray(ti['labels'])
    for seed in c['seeds']:
        for kind in KINDS:
            name = f'{kind}-s{seed}'
            path = ART/(name+'.safetensors')
            if path.exists():
                raise FileExistsError('Partial head exists: '+name)
            torch.manual_seed(500000+seed)
            model = init(kind, seed, c)
            initial = state_digest(model.state_dict())
            if seed in initializations:
                assert initial == initializations[seed]
            initializations[seed] = initial
            model.to('mps')
            opt = torch.optim.AdamW(model.parameters(), lr=c['learning_rate'], weight_decay=c['weight_decay'])
            history, began = [], time.perf_counter()
            for epoch in range(c['epochs']):
                model.train()
                order = np.random.default_rng(20261018+seed*100+epoch).permutation(len(labels))
                total, updates, maximum_grad = 0., 0, 0.
                for start in range(0, len(labels), c['batch_size']):
                    ii = order[start:start+c['batch_size']]
                    opt.zero_grad(set_to_none=True)
                    z = model(*batch(tx, ti, ii))
                    loss = torch.nn.functional.cross_entropy(z, torch.as_tensor(labels[ii], device='mps'))
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite query loss')
                    loss.backward()
                    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                        raise FloatingPointError('Missing/nonfinite query gradient')
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    maximum_grad = max(maximum_grad, float(norm))
                    opt.step()
                    total += float(loss.detach())*len(ii)
                    updates += 1
                    if start % 6400 == 0:
                        emit('training', candidate=name, epoch=epoch+1, rows=start+len(ii), seconds=time.perf_counter()-began)
                record = dict(epoch=epoch+1, mean_loss=total/len(labels), updates=updates, exposures=len(labels), max_grad_norm=maximum_grad)
                history.append(record)
                emit('epoch_complete', candidate=name, **record, seconds=time.perf_counter()-began)
            z = predict(model, cx, ci)
            state = {k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
            if any(not torch.isfinite(v).all() for v in state.values()):
                raise FloatingPointError('Nonfinite saved head')
            save_file(state, path)
            restored = init(kind, seed, c).to('mps')
            restored.load_state_dict(load_file(path), strict=True)
            np.testing.assert_array_equal(z, predict(restored, cx, ci))
            margin = z[:,1]-z[:,0]
            temperature = fit_temperature(margin, cy, bounds=(.5,5.))
            all_results[name] = dict(calibration_logits=z.tolist(), temperature=temperature)
            records[name] = dict(history=history, initial_state_sha256=initial, weights_sha256=file_digest(path),
                parameters=sum(p.numel() for p in model.parameters()), seconds=time.perf_counter()-began, reload_exact=True)
            save('training-progress.json', dict(records=records, calibration=all_results, development_opened=False))
            del model, restored, opt
            torch.mps.empty_cache()
    fits, probabilities = {}, {}
    baseline = (pa+pb)/2
    for kind in KINDS:
        pp = [sigmoid((np.asarray(all_results[f'{kind}-s{s}']['calibration_logits'])[:,1]-
                        np.asarray(all_results[f'{kind}-s{s}']['calibration_logits'])[:,0])/
                        all_results[f'{kind}-s{s}']['temperature']) for s in c['seeds']]
        p = np.mean(pp, axis=0)
        mixture = fit_mixture(baseline, p, cy)
        fits[kind] = mixture
        probabilities[kind] = p.tolist()
    frozen = dict(run_id=RUN, frozen_utc=now(), config_sha256=file_digest(CONFIG), primary='dense',
        records=records, calibration=all_results, mixtures=fits, calibration_branch_probabilities=probabilities,
        calibration_ids=cal_ids, calibration_labels=cy.tolist(), baseline=baseline.tolist(),
        source_protocol_sha256=file_digest(OUT/'protocol.json'), development_opened=False, test_accessed=False,
        token_index_sha256={s:file_digest(ART/(s+'-index.json')) for s in ('gradient_train','calibration')})
    save('heads-frozen.json', frozen)
    emit('all_heads_frozen', models=len(records), mixtures=fits)


def check_frozen():
    p = verify_bindings()
    f = read(OUT/'heads-frozen.json')
    assert f['source_protocol_sha256'] == file_digest(OUT/'protocol.json')
    assert f['config_sha256'] == file_digest(CONFIG) and not f['development_opened'] and not f['test_accessed']
    for name, record in f['records'].items():
        assert file_digest(ART/(name+'.safetensors')) == record['weights_sha256']
    return f, p


def develop():
    from safetensors.torch import load_file
    f, p = check_frozen()
    if (OUT/'development-heads.json').exists():
        raise FileExistsError('Development already predicted')
    frozen_sha = file_digest(OUT/'heads-frozen.json')
    path = ROOT/'data/processed/validation.jsonl'
    assert file_digest(path) == p['development_data_sha256']
    rows = list(read_jsonl(path))
    spec = source_spec()
    assert [r['id'] for r in rows] == read(ROOT/spec['manifest'])['validation']['ids']
    upstream, model, tok = load_source()
    extraction(upstream, model, tok, rows, 'validation', config()['extraction_batch_size'])
    del upstream, model, tok
    torch.mps.empty_cache()
    x, index = load_tokens('validation')
    outputs = {}
    for seed in config()['seeds']:
        for kind in KINDS:
            name = f'{kind}-s{seed}'
            model = init(kind, seed, config()).to('mps')
            model.load_state_dict(load_file(ART/(name+'.safetensors')), strict=True)
            z = predict(model, x, index)
            outputs[name] = z.tolist()
            del model
    assert frozen_sha == file_digest(OUT/'heads-frozen.json')
    save('development-heads.json', dict(sample_ids=index['sample_ids'], true_labels=index['labels'],
        truncated=index['truncated'], logits=outputs, frozen_sha256=frozen_sha, test_accessed=False))
    emit('development_predictions_complete', count=len(rows))


def evaluate():
    from run_bow_nb_study import compare
    from run_final_cpu_alternatives_v2 import holm, met, NEURAL, PREVIOUS
    f, p = check_frozen()
    if (OUT/'summary.json').exists():
        raise FileExistsError('Evaluation already complete')
    d = read(OUT/'development-heads.json')
    assert d['frozen_sha256'] == file_digest(OUT/'heads-frozen.json')
    provenance = read(ROOT/PREVIOUS/'protocol.json')
    path = ROOT/NEURAL/'predictions.json'
    assert file_digest(path) == provenance['development_cache_expected_sha256']
    neural = read(path)
    assert not neural['test_accessed'] and neural['sample_ids'] == d['sample_ids']
    y, long = np.asarray(d['true_labels']), np.asarray(d['truncated'], dtype=bool)
    assert y.tolist() == neural['true_labels'] and len(y) == 2500 and long.sum() == 407
    baseline = np.asarray(neural['positive_probabilities']['fusion_50_50'])
    assert ((baseline>.5)==y).sum() == 2432
    native_records, native = old_cache('validation')
    native_p = sigmoid(native['full_logits'][:,1]-native['full_logits'][:,0])
    results, predictions, single_seeds = {}, {}, {}
    for kind in KINDS:
        blocks = []
        for seed in config()['seeds']:
            name = f'{kind}-s{seed}'
            z = np.asarray(d['logits'][name])
            t = f['calibration'][name]['temperature']
            pp = sigmoid((z[:,1]-z[:,0])/t)
            single_seeds[name] = met(y, pp)
            blocks.append(pp)
        pp = np.mean(blocks, axis=0)
        w = f['mixtures'][kind]['sparse_weight']
        blend = (1-w)*baseline+w*pp
        predictions[kind] = dict(branch=pp.tolist(), fusion=blend.tolist())
        results[kind] = dict(branch=met(y, pp), fusion=met(y, blend), weight=w,
            branch_vs_native=compare(y, native_p, pp), fusion_vs_baseline=compare(y, baseline, blend),
            groups={name:dict(branch=met(y[mask], pp[mask]), fusion=met(y[mask], blend[mask]))
                    for name,mask in [('long',long),('short',~long)]})
    adjusted = holm({k:r['fusion_vs_baseline']['mcnemar_exact_two_sided_p'] for k,r in results.items()})
    for kind in KINDS:
        r = results[kind]
        r['fusion_holm_p'] = adjusted[kind]
        pair = r['fusion_vs_baseline']
        r['continue_rule_passed'] = pair['net']>=5 and pair['conditional_stratified_bootstrap95_pp'][0]>0 and adjusted[kind]<.05
    comparisons = {}
    for kind in ('dense','top32'):
        comparisons[kind+'_vs_mean'] = compare(y, np.asarray(predictions['mean']['branch']), np.asarray(predictions[kind]['branch']))
    summary = dict(run_id=RUN, primary='dense', baseline=met(y,baseline), native_encoder_head=met(y,native_p),
        results=results, single_seeds=single_seeds, controlled_comparisons=comparisons,
        limitations=['frozen fine-tuned encoder','historically reused dev and calibration','not DSA replication','no end-to-end or domain pretraining'],
        test_accessed=False, frozen_sha256=file_digest(OUT/'heads-frozen.json'))
    save('development-probabilities.json', dict(sample_ids=d['sample_ids'],true_labels=y.tolist(),baseline=baseline.tolist(), predictions=predictions))
    save('summary.json',summary)
    lines=['# 全序列查询头与 Top-32 稀疏选择实验','',
        '固定已微调的 baseline-s43 编码器，完整 FP32 token 缓存；每组 3 种子、固定 5 轮。未做编码器端到端更新，未做领域预训练，不是 DSA 复现。', '',
        '主假设预先固定为 dense。全部训练、温度和融合权重在开发集读取前冻结；本轮不访问官方测试。历史开发集和校准集已反复使用。','',
        '| 方法 | 三种子分支 Accuracy | 校准分支权重 | 融合 Accuracy | 修复 / 新错 / 净增 | Holm p |','|---|---:|---:|---:|---|---:|']
    for k,r in results.items():
        pair=r['fusion_vs_baseline']
        lines.append(f"| {k} | {r['branch']['accuracy']*100:.2f}% | {r['weight']*100:.4f}% | {r['fusion']['accuracy']*100:.2f}% | {pair['repaired']} / {pair['damaged']} / {pair['net']:+d} | {r['fusion_holm_p']:.4g} |")
    lines += ['', '神经融合参照为97.28%；原s43单模型参照96.84%。均值组与查询组使用相同参数、相同初始化、相同训练顺序与预算；原历史头没有重新训练，不能视为等预算端到端比较。', '',
        '完整统计、NLL/ECE、长短分组、单种子结果及查询相对均值对照见 summary.json。未根据开发成绩选择种子或重新训练。继续规则：融合净增≥5、配对bootstrap下界>0、三候选Holm p<.05。', '',
        '复现：uv run --locked --offline --no-sync python scripts/run_token_query_study.py extract / train / develop；evaluate 使用 environments/bow-cpu 锁定环境。MPS阶段需本机GPU可见。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    emit('evaluation_complete', results={k:dict(branch=r['branch']['accuracy'],fusion=r['fusion']['accuracy'],weight=r['weight']) for k,r in results.items()})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('extract','train','develop','evaluate'))
    args=parser.parse_args()
    guard(args.stage)
    if args.stage!='evaluate':
        global torch, TokenQueryHead
        import torch
        from token_query_head import TokenQueryHead
        torch.set_num_threads(4)
        if not torch.backends.mps.is_available():
            raise RuntimeError('Apple GPU unavailable; do not silently switch precision/device')
    functions=dict(extract=extract,train=train,develop=develop,evaluate=evaluate)
    try:
        functions[args.stage]()
    finally:
        if OUT.exists():
            save(args.stage+'-access-audit.json',dict(read_paths=sorted(READS),test_accessed=False))


if __name__=='__main__':
    main()

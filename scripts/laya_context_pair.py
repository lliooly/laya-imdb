"""Frozen paired context experiment, epoch bundles, admission and gated evaluation."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time

from common import file_digest, read_jsonl, write_json

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'laya-context-pair-s42-modal-20261016-v1'
VOLUME = 'end-imdb-laya-robust-deferred-20261015-v2'
BASE = 'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json'
SECONDS = 13500
CAP_USD = 12


def read(path):
    return json.loads(Path(path).read_text())


def config(root=ROOT):
    c = read(root/'configs'/f'{STUDY}.json')
    if (c['study_id'] != STUDY or c['additional_reserve_usd'] != CAP_USD
            or c['maximum_seconds'] != SECONDS or c['lengths'] != [512, 2048]
            or c['seed'] != 42):
        raise ValueError('Frozen context study scope differs')
    for rel, sha in c['source_sha256'].items():
        if file_digest(root/rel) != sha:
            raise ValueError('Bound source changed: '+rel)
    return c


def run_name(cap):
    if cap not in (512, 2048):
        raise ValueError('Unexpected context length')
    return f'context-{cap}-s42'


def bundle(cap, epoch, root=ROOT):
    return root/'checkpoints'/STUDY/run_name(cap)/f'epoch-{epoch+1:03d}'


def receipt(cap, epoch, root=ROOT):
    return root/'reports'/STUDY/'cloud-delivery'/f'{run_name(cap)}-epoch-{epoch+1:03d}.json'


def identity(cap, root=ROOT):
    return {'study_id': STUDY, 'run_id': run_name(cap), 'max_len': cap, 'seed': 42,
            'contract_sha256': file_digest(root/'configs'/f'{STUDY}.json')}


def training_config(cap, micro_batch=8):
    import laya.train as upstream
    from run_laya_head_control import TRAINING
    return upstream.TrainConfig(**dict(TRAINING, seed=42, max_len=cap,
                                     micro_batch=micro_batch, log_every=64))


def tokenizer_and_rows():
    from transformers import AutoTokenizer
    lock = read(ROOT/'configs/laya_model_lock.json')
    for rel, sha in lock['sha256'].items():
        if file_digest(ROOT/lock['local_dir']/rel) != sha:
            raise ValueError('Published model asset differs: '+rel)
    tok = AutoTokenizer.from_pretrained(str(ROOT/lock['local_dir']/'tokenizer'), local_files_only=True)
    split = read(ROOT/BASE)
    rows = list(read_jsonl(ROOT/'data/processed/train.jsonl'))
    if [r['id'] for r in rows] != split['train']['ids']:
        raise ValueError('Training order differs')
    groups = [set(split[k]) for k in ('gradient_train_ids', 'calibration_ids')]
    dev = set(split['validation']['ids'])
    if ([len(s) for s in groups] != [22100, 400] or len(dev) != 2500
            or groups[0]&groups[1] or any(g&dev for g in groups)):
        raise ValueError('Invalid frozen partition')
    return tok, {r['id']: r for r in rows}, split, lock


def items_for(tok, rows, cap):
    import laya.train as upstream
    from laya.common import build_head
    from laya_decision_views import full_tokens, window_tokens
    from train_laya import QUESTION
    typed = [{'state': r['text'], 'questions': {'sentiment': QUESTION},
              'expected': {'sentiment': ('negative', 'positive')[r['label']]}} for r in rows]
    items, skipped = upstream.items_from_rows(tok, typed, cap, 192, label_smoothing=0.)
    if skipped or len(items) != len(rows):
        raise ValueError('Input omission')
    lengths = []
    for item, full in zip(items, full_tokens(tok, [r['text'] for r in rows]), strict=True):
        head, _, _ = build_head(tok, item['q'], 192)
        if len(head) != 45:
            raise ValueError('Template overhead differs')
        item['state_ids'] = window_tokens(full, 'head_tail', cap-46)
        lengths.append(len(full))
    return items, lengths


def files_under(folder):
    return [{'path': str(p.relative_to(folder)), 'bytes': p.stat().st_size, 'sha256': file_digest(p)}
            for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'bundle.json']


def seal(folder, ident, stage):
    manifest = {'format': 'laya-context-bundle-v1', 'identity': ident, 'stage': stage,
                'bundle': str(folder.relative_to(ROOT)), 'files': files_under(folder)}
    write_json(folder/'bundle.json', manifest)
    return manifest


def check_bundle(folder):
    from laya_checkpoint_delivery import within
    manifest = read(folder/'bundle.json')
    if manifest['format'] != 'laya-context-bundle-v1' or manifest['identity']['study_id'] != STUDY:
        raise ValueError('Bundle belongs to another study')
    if len({f['path'] for f in manifest['files']}) != len(manifest['files']):
        raise ValueError('Repeated bundle file')
    for f in manifest['files']:
        p = within(folder, f['path'])
        if not p.is_file() or p.stat().st_size != f['bytes'] or file_digest(p) != f['sha256']:
            raise ValueError('Sealed bundle corrupted: '+f['path'])
    return manifest


def accepted(cap, epoch, root=ROOT):
    folder = bundle(cap, epoch, root)
    r = read(receipt(cap, epoch, root))
    if (r['state'] != 'verified_committed_volume' or r['identity'] != identity(cap, root)
            or r['bundle_sha256'] != file_digest(folder/'bundle.json')
            or not r['verification']['strict_load']):
        raise ValueError('Cloud acceptance differs')
    return r


def assets(folder):
    from importlib.metadata import version
    import platform
    import torch
    c = read(ROOT/'configs'/f'{STUDY}.json')
    for rel in c['source_sha256']:
        if rel.startswith(('scripts/', 'configs/', 'environments/')):
            out = folder/'assets/source'/rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT/rel, out)
    shutil.copyfile(ROOT/'configs'/f'{STUDY}.json', folder/'assets/contract.json')
    lock = read(ROOT/'configs/laya_model_lock.json')
    shutil.copytree(ROOT/lock['local_dir']/'tokenizer', folder/'assets/tokenizer')
    write_json(folder/'assets/runtime.json', {'python': platform.python_version(),
        'packages': {p:version(p) for p in ['torch','transformers','laya','numpy','safetensors','tokenizers']},
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        'strict_deterministic':torch.are_deterministic_algorithms_enabled(),
        'amp':False,'tf32':torch.backends.cuda.matmul.allow_tf32})


def save_boundary(model, tok, items, cfg, opt, sched, rng, ident, epoch, count, history, folder, journal=None):
    import torch
    import laya.train as upstream
    from laya_context_training import snapshot
    from laya_head_control import save_fp32
    from laya_robust_training import finite_tree
    device = next(model.parameters()).device
    stage = folder.with_name(folder.name+'.writing')
    if folder.exists():raise FileExistsError(folder)
    meta = save_fp32(model, stage)
    state = snapshot(model, opt, sched, rng, device, ident, epoch, count, history, cfg.micro_batch)
    if not finite_tree(state):
        raise FloatingPointError('Invalid saved training state')
    torch.save(state, stage/'optimizer_rng.pt')
    # Reuse actual gradient examples only; reference evaluation consumes no RNG.
    prepared = [upstream.encode_item(tok, it, cfg.max_len, 192, [0, 1], False) for it in items[:2]]
    batch = upstream.collate_items([prepared], tok.pad_token_id)
    model.eval()
    with torch.inference_mode():
        logits = upstream._forward(model, batch, device, False, False).cpu().tolist()
    write_json(stage/'reference.json', {'inputs': prepared, 'pad_token_id': tok.pad_token_id, 'logits': logits})
    write_json(stage/'training.json', {'identity': ident, 'next_epoch': epoch+1,
               'initial_state_sha256': history[0]['initial_state_sha256'], 'history': history,
               'updates': state['updates'], 'exposures': state['exposures'], 'test_accessed': False})
    if journal:
        shutil.move(str(journal), stage/'audit')
        assets(stage)
    # Atomic rename publishes an entirely closed, immutable directory.
    seal(stage, ident, f'epoch-{epoch+1:03d}')
    value = read(stage/'bundle.json')
    value['bundle'] = str(folder.relative_to(ROOT))
    write_json(stage/'bundle.json', value)
    if folder.exists():
        raise FileExistsError(folder)
    stage.rename(folder)
    return meta


def verify(folder, output):
    import torch
    import laya.train as upstream
    from laya_context_training import FORMAT
    from laya_head_control import load_fp32
    from laya_robust_training import finite_tree
    if folder.resolve() in output.resolve().parents:
        raise ValueError('Verification receipts must be outside the immutable bundle')
    torch.set_num_threads(4)
    m = check_bundle(folder)
    model, meta = load_fp32(folder)
    state = torch.load(folder/'optimizer_rng.pt', map_location='cpu', weights_only=True)
    training = read(folder/'training.json')
    if (state['format'] != FORMAT or state['identity'] != m['identity']
            or state['next_epoch'] != training['next_epoch'] or state['history'] != training['history']
            or state['updates'] != training['updates'] or state['exposures'] != training['exposures']
            or state['model_state_sha256'] != meta['state_sha256'] or not finite_tree(state)):
        raise ValueError('Training snapshot identity differs')
    del state
    ref = read(folder/'reference.json')
    batch = upstream.collate_items([ref['inputs']], ref['pad_token_id'])
    model.eval()
    with torch.inference_mode():
        actual = upstream._forward(model, batch, torch.device('cpu'), False, False).cpu()
    expected = torch.tensor(ref['logits'])
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    result = {'strict_load': True, 'optimizer_rng_finite': True, 'state_sha256': meta['state_sha256'],
              'max_abs_logit_difference': float((actual-expected).abs().max()), 'rtol': 1e-4, 'atol': 1e-4,
              'bundle_sha256': file_digest(folder/'bundle.json'), 'device': 'cpu'}
    write_json(output, result)
    print(json.dumps(result), flush=True)


def train_epoch(cap, epoch, deadline):
    import torch
    from laya_context_training import setup, run_epoch, restore, Journal
    from laya_head_control import load_fp32, state_digest
    from laya_long_context_pilot import source_model
    tok, by_id, split, lock = tokenizer_and_rows()
    ids = split['gradient_train_ids']
    items, _ = items_for(tok, [by_id[i] for i in ids], cap)
    cfg = training_config(cap)
    ident = identity(cap)
    folder = bundle(cap, epoch)
    if folder.exists() or folder.with_name(folder.name+'.writing').exists():
        raise FileExistsError('Refuse an implicit rerun or partial overwrite')
    if epoch:
        accepted(cap, 0)
        model, _ = load_fp32(bundle(cap, 0))
    else:
        model = source_model(ROOT/lock['local_dir'])
    initial = state_digest(model.state_dict())
    if not epoch and initial != '1b7f6f9916d53639303b90df3f3e9e77112912d16eb36af8c51fa88190b6c7e4':
        raise ValueError('Source differs from the verified pilot initialization')
    opt, sched, rng = setup(model, cfg, len(items), torch.device('cuda'))
    history = []
    if epoch:
        state = torch.load(bundle(cap, 0)/'optimizer_rng.pt', map_location='cpu', weights_only=True)
        history = restore(state, model, opt, sched, rng, torch.device('cuda'), ident, len(items), 8, 1)
        del state
        initial = history[0]['initial_state_sha256']
    journal = Journal(ROOT/'artifacts'/STUDY/f'{run_name(cap)}-epoch-{epoch+1:03d}-audit')
    try:
        loss, audit = run_epoch(model, tok, items, ids, cfg, torch.device('cuda'), cap,
                               epoch, opt, sched, rng, journal, deadline)
    finally:
        journal.close()
    history.append(dict(audit, loss=loss, initial_state_sha256=initial))
    save_boundary(model, tok, items, cfg, opt, sched, rng, ident, epoch, len(items), history, folder, journal.folder)


def admission(deadline):
    import torch
    from laya_context_training import setup, run_epoch, restore
    from laya_head_control import load_fp32, state_digest
    from laya_long_context_pilot import source_model
    from laya_robust_training import capture_rng
    tok, by_id, split, lock = tokenizer_and_rows()
    from laya_decision_views import full_tokens
    ids = split['gradient_train_ids']
    rows = [by_id[i] for i in ids]
    sizes = [len(t) for t in full_tokens(tok, [r['text'] for r in rows])]
    selected = sorted(range(len(ids)), key=lambda i: (-sizes[i], i))[:8]
    ids = [ids[i] for i in selected]
    items, _ = items_for(tok, [by_id[i] for i in ids], 2048)
    cfg = training_config(2048, 8)
    ident = dict(identity(2048), admission=True)
    model = source_model(ROOT/lock['local_dir'])
    initial = state_digest(model.state_dict())
    opt, sched, rng = setup(model, cfg, len(items), torch.device('cuda'))
    loss, audit = run_epoch(model, tok, items, ids, cfg, torch.device('cuda'), 2048, 0, opt, sched, rng, deadline=deadline)
    history = [dict(audit, loss=loss, initial_state_sha256=initial)]
    folder = ROOT/'checkpoints'/STUDY/'admission-2048'
    save_boundary(model, tok, items, cfg, opt, sched, rng, ident, 0, len(items), history, folder)
    loss_a, _ = run_epoch(model, tok, items, ids, cfg, torch.device('cuda'), 2048, 1, opt, sched, rng, deadline=deadline)
    expected = state_digest(model.state_dict())
    expected_rng = capture_rng(torch.device('cuda'))
    expected_option = rng.getstate()
    del model, opt, sched
    gc.collect(); torch.cuda.empty_cache()
    model, _ = load_fp32(folder)
    opt, sched, rng = setup(model, cfg, len(items), torch.device('cuda'))
    state = torch.load(folder/'optimizer_rng.pt', map_location='cpu', weights_only=True)
    restore(state, model, opt, sched, rng, torch.device('cuda'), ident, len(items), 8, 1)
    del state
    loss_b, _ = run_epoch(model, tok, items, ids, cfg, torch.device('cuda'), 2048, 1, opt, sched, rng, deadline=deadline)
    actual = state_digest(model.state_dict())
    actual_rng = capture_rng(torch.device('cuda'))
    def rng_sha(value):
        import hashlib
        def plain(x):
            if isinstance(x, torch.Tensor): return x.tolist()
            if isinstance(x, dict): return {k: plain(v) for k,v in x.items()}
            if isinstance(x, (tuple,list)): return [plain(v) for v in x]
            return x
        return hashlib.sha256(json.dumps(plain(value), sort_keys=True).encode()).hexdigest()
    if (expected != actual or loss_a != loss_b or expected_option != rng.getstate()
            or rng_sha(expected_rng) != rng_sha(actual_rng)):
        raise ValueError('Actual 2048 CUDA epoch recovery differs from continuous training')
    write_json(ROOT/'reports'/STUDY/'admission.json', {'state': 'passed', 'max_len': 2048,
               'sample_ids': ids, 'initial_state_sha256': initial, 'final_state_sha256': actual,
               'exact_resume_weights_rng_losses': True, 'test_accessed': False, 'development_accessed': False})


def freeze(root=ROOT):
    if read(root/'reports'/STUDY/'admission.json')['state'] != 'passed':
        raise ValueError('Admission absent')
    final = []
    for cap in (512, 2048):
        for epoch in (0, 1):
            accepted(cap, epoch, root)
        train = read(bundle(cap, 1, root)/'training.json')
        if train['updates'] != 5526 or train['exposures'] != 44200:
            raise ValueError('Final exposure differs')
        final.append(train)
    if (final[0]['initial_state_sha256'] != final[1]['initial_state_sha256']
            or any(a['schedule_sha256'] != b['schedule_sha256'] or a['updates'] != 2763
                   or b['updates'] != 2763 or a['exposures'] != 22100 or b['exposures'] != 22100
                   for a,b in zip(final[0]['history'], final[1]['history'], strict=True))):
        raise ValueError('Paired initialization, inputs or updates differ')
    gate = {'state': 'both_frozen', 'study_id': STUDY, 'test_accessed': False,
            'final_bundle_sha256': {str(c): file_digest(bundle(c, 1, root)/'bundle.json') for c in (512, 2048)}}
    path = root/'reports'/STUDY/'both-frozen.json'
    if path.exists() and read(path) != gate:
        raise ValueError('Frozen models changed')
    write_json(path, gate)
    return gate


def wilson(correct, count):
    p, z = correct/count, 1.959963984540054
    center = (p+z*z/(2*count))/(1+z*z/count)
    half = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [center-half, center+half]


def evaluate(deadline):
    import numpy as np
    import torch
    import laya.train as upstream
    from laya_head_control import load_fp32
    from comparison_metrics import metrics, paired, fit_temperature
    gate = freeze()
    report = ROOT/'reports'/STUDY
    if (report/'evaluation/bundle.json').exists():
        raise FileExistsError('Evaluation already sealed')
    tok, by_id, split, _ = tokenizer_and_rows()
    dev = list(read_jsonl(ROOT/'data/processed/validation.jsonl'))
    if [r['id'] for r in dev] != split['validation']['ids']:
        raise ValueError('Development order changed')
    out = report/'evaluation.writing'
    out.mkdir(exist_ok=False)
    result = {'study_id': STUDY, 'state': 'evaluated', 'test_accessed': False,
              'selection': 'fixed_epoch_2_single_seed_42', 'models': {}, 'frozen_gate': gate,
              'limitations': ['single_seed', 'development_reused', 'conditional_sample_uncertainty_only']}
    margins = {}
    lengths = None
    labels = np.asarray([r['label'] for r in dev])
    for cap in (512, 2048):
        model, _ = load_fp32(bundle(cap, 1))
        model.to('cuda').eval()
        values = {}
        for part, rows in [('calibration', [by_id[i] for i in split['calibration_ids']]), ('development', dev)]:
            items, sizes = items_for(tok, rows, cap)
            logits = []
            with torch.inference_mode():
                for start in range(0, len(items), 8):
                    if time.time() >= deadline-120:
                        raise TimeoutError('Deadline before evaluation completion')
                    prepared = [upstream.encode_item(tok, it, cap, 192, [0, 1], False) for it in items[start:start+8]]
                    batch = upstream.collate_items([prepared], tok.pad_token_id)
                    z = upstream._forward(model, batch, torch.device('cuda'), False, False).cpu().numpy()
                    if z.shape != (len(prepared), 2) or not np.isfinite(z).all():
                        raise FloatingPointError('Invalid prediction logits')
                    logits.extend(z.tolist())
            values[part] = np.asarray(logits)
            write_json(out/f'{cap}-{part}.json', {'sample_ids': [r['id'] for r in rows],
                       'true_labels': [r['label'] for r in rows], 'review_token_lengths': sizes,
                       'logits': logits, 'test_accessed': False})
            if part == 'development':
                lengths = np.asarray(sizes)
        calibration_labels = [by_id[i]['label'] for i in split['calibration_ids']]
        temperature = fit_temperature(values['calibration'][:, 1]-values['calibration'][:, 0], calibration_labels, bounds=(.5, 5.))
        m = values['development'][:, 1]-values['development'][:, 0]
        margins[cap] = m
        groups = {}
        for name, keep in [('all', np.ones(len(dev), dtype=bool)), ('truncated_at_512', lengths>466),
                           ('complete_at_512', lengths<=466)]:
            raw = metrics(labels[keep], m[keep])
            groups[name] = {'raw': raw, 'calibrated': metrics(labels[keep], m[keep], temperature),
                            'accuracy_wilson95': wilson(raw['correct'], raw['count'])}
        result['models'][str(cap)] = {'temperature': temperature, 'groups': groups,
                                     'native_truncated_count': int((lengths>cap-46).sum())}
        del model
        gc.collect(); torch.cuda.empty_cache()
    result['paired'] = {}
    for name, keep in [('all', np.ones(len(dev), dtype=bool)), ('truncated_at_512', lengths>466),
                       ('complete_at_512', lengths<=466)]:
        a, b, y = margins[512][keep], margins[2048][keep], labels[keep]
        delta = ((b>0)==y).astype(int)-((a>0)==y).astype(int)
        rng = np.random.default_rng(20261016)
        # Multinomial resampling of {-1,0,1} is the exact paired empirical bootstrap.
        counts = np.asarray([(delta==v).sum() for v in (-1,0,1)])
        draw = rng.multinomial(len(delta), counts/len(delta), size=20000)
        ci = np.quantile((draw[:,2]-draw[:,0])/len(delta), [.025,.975]).tolist()
        result['paired'][name] = dict(paired(y, a, b), delta_accuracy=float(delta.mean()),
                                      delta_accuracy_conditional_bootstrap95=ci, count=len(delta))
    write_json(out/'summary.json', result)
    lines = ['# 512 / 2048 正式对照结果', '', '固定 epoch 2，seed 42；仅开发集，未读取官方测试集。', '',
             '| 总长度 | Accuracy | Macro-F1 | AUROC | 校准 NLL | 校准 ECE |', '|---|---:|---:|---:|---:|---:|']
    for cap in (512,2048):
        m = result['models'][str(cap)]['groups']['all']
        lines.append(f"| {cap} | {m['raw']['accuracy']*100:.2f}% | {m['raw']['macro_f1']:.6f} | {m['raw']['auroc']:.6f} | {m['calibrated']['nll']:.6f} | {m['calibrated']['ece_15_equal_width_top_label']:.6f} |")
    lines += ['', '## 配对比较与不确定性', '']
    for name, p in result['paired'].items():
        ci = [x*100 for x in p['delta_accuracy_conditional_bootstrap95']]
        lines.append(f"- {name}：n={p['count']}，Δ={p['delta_accuracy']*100:+.2f} pp，条件配对 bootstrap 95% 区间 [{ci[0]:+.2f}, {ci[1]:+.2f}] pp；修复 {p['b_only_correct']}、退化 {p['a_only_correct']}，精确 McNemar p={p['mcnemar_exact_two_sided_p']:.6g}。")
    lines += ['', '该区间仅反映固定模型的样本不确定性，不覆盖训练种子变异。开发集已被多轮使用，存在选择乐观性；不得据此宣称未见测试集或跨种子稳定收益。共同截断组按 review tokens >466 定义。原始/校准完整指标、Wilson 区间和逐条预测见 summary.json 与预测文件。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    assets(out)
    seal(out, {'study_id': STUDY}, 'evaluated')
    m = read(out/'bundle.json'); m['bundle'] = str((report/'evaluation').relative_to(ROOT))
    write_json(out/'bundle.json', m)
    out.rename(report/'evaluation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['admission','train','evaluate','verify'])
    parser.add_argument('--cap', type=int, choices=[512,2048])
    parser.add_argument('--epoch', type=int, choices=[0,1])
    parser.add_argument('--deadline', type=float, default=0)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.mode == 'verify':
        verify(args.bundle, args.output)
        return
    config()
    from laya_long_context_pilot import strict_runtime
    strict_runtime()
    from train_laya import block_network
    block_network()
    if args.mode == 'admission': admission(args.deadline)
    elif args.mode == 'train': train_epoch(args.cap, args.epoch, args.deadline)
    else: evaluate(args.deadline)


if __name__ == '__main__':
    main()

"""One fixed L2 run; calibration reload and gated development run in new processes."""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import time

from common import ROOT, file_digest, read_jsonl, write_json


STUDY = 'laya-head-control-cu126-20261015-v2'
PAIRS = [(42, 'original'), (42, 'linear'), (43, 'linear'),
         (43, 'original'), (44, 'original'), (44, 'linear')]
BASE_MANIFEST = 'reports/laya-rtx4090-ce-baseline-s43-epochs2-fp32-v1/manifest.json'
TRAINING = dict(epochs=2, micro_batch=8, grad_accum=1, encoder_lr=2.5e-5,
                head_lr=1e-4, min_lr=1e-6, weight_decay=.01, grad_clip=1.,
                loss='soft-ce', shuffle_options=['choice'], freeze_encoder=False,
                calib_max=400, calib_frac=.1, calib_seed=42, max_len=512,
                head_max_len=192, amp=False, gradient_checkpointing=True,
                log_every=64, label_smoothing=0.)


def now():
    return datetime.now(timezone.utc).isoformat()


def name(seed, kind, repeat=None):
    result = f'laya-head-control-{kind}-s{seed}-strict-fp32-20261015-v2'
    return result if repeat is None else result+f'-admission-r{repeat}'


def verify_contract():
    contract = json.loads((ROOT/'configs'/f'{STUDY}.json').read_text())
    if contract['study_id'] != STUDY or contract['training'] != TRAINING or contract['pairs'] != [list(x) for x in PAIRS]:
        raise ValueError('Fixed experiment protocol changed')
    for rel, expected in contract['source_sha256'].items():
        if file_digest(ROOT/rel) != expected:
            raise ValueError('Bound source changed: '+rel)
    lock = json.loads((ROOT/'configs/laya_model_lock.json').read_text())
    for rel, expected in lock['sha256'].items():
        if file_digest(ROOT/lock['local_dir']/rel) != expected:
            raise ValueError('Published asset changed: '+rel)
    return contract, lock


def load_training():
    import laya.train as upstream
    split = json.loads((ROOT/'reports/split_manifest.json').read_text())
    manifest = json.loads((ROOT/BASE_MANIFEST).read_text())
    for part in ('train', 'validation'):
        if file_digest(ROOT/f'data/processed/{part}.jsonl') != split['jsonl_sha256'][part] or manifest[part]['ids'] != split['ids'][part]:
            raise ValueError('Frozen data source or partition order changed')
    rows = list(read_jsonl(ROOT/'data/processed/train.jsonl'))
    if [r['id'] for r in rows] != split['ids']['train']:
        raise ValueError('Frozen training ID order changed')
    ids, calibration = upstream.split_calibration([r['id'] for r in rows], 400, .1, 42)
    if ids != manifest['gradient_train_ids'] or calibration != manifest['calibration_ids']:
        raise ValueError('Calibration partition changed')
    groups = [set(ids), set(calibration), set(split['ids']['validation'])]
    if [len(s) for s in groups] != [22100, 400, 2500] or any(groups[i]&groups[j] for i in range(3) for j in range(i+1, 3)):
        raise ValueError('Unexpected partition counts or overlap')
    by_id = {r['id']: r for r in rows}
    if any(type(r['label']) is not int or r['label'] not in (0, 1) for r in rows):
        raise ValueError('Invalid original training labels')
    return rows, ids, calibration, by_id


def tokenizer(lock):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(ROOT/lock['local_dir']/'tokenizer'), local_files_only=True)


def predict(model, tok, rows, device):
    import numpy as np
    import torch
    import laya.train as upstream
    from export_laya_posthoc import prepare_rules, input_digest
    model.eval()
    blocks, records = [], []
    with torch.inference_mode():
        for start in range(0, len(rows), 8):
            chunk = rows[start:start+8]
            _, items, stats, _, _ = prepare_rules(tok, [dict(r, label=0) for r in chunk])
            prepared = [upstream.encode_item(tok, it, 512, 192, [0, 1], False) for it in items]
            batch = upstream.collate_items([prepared], tok.pad_token_id)
            if not torch.equal(torch.gather(batch['input_ids'], 1, batch['marker_pos']), torch.full_like(batch['marker_pos'], tok.mask_token_id)):
                raise ValueError('Expected two real MASK markers')
            logits = upstream._forward(model, batch, device, False, False).cpu().numpy()
            if logits.shape != (len(chunk), 2) or not np.isfinite(logits).all():
                raise FloatingPointError('Invalid prediction logits')
            blocks.append(logits.copy())
            records.extend({'sample_id': r['id'], 'true_label': r['label'],
                            'input_sha256': input_digest(p), **stat}
                           for r, p, stat in zip(chunk, prepared, stats, strict=True))
    return {'rows': records, 'logits': np.concatenate(blocks).tolist(), 'test_accessed': False}


class Tee:
    def __init__(self, first, second):
        self.first, self.second = first, second
    def write(self, text):
        self.first.write(text);self.second.write(text);self.second.flush()
    def flush(self):
        self.first.flush();self.second.flush()


def train(args, contract, lock, device):
    import numpy as np
    import torch
    import laya.train as upstream
    from cuda_numerics import guard_training
    from export_laya_posthoc import prepare_rules
    from laya_head_control import from_source, state_digest, save_fp32, audit_training
    rows, ids, calibration, by_id = load_training()
    if args.mode == 'admission':
        chosen = {y: [i for i in ids if by_id[i]['label'] == y][:64] for y in (0, 1)}
        selected = set(chosen[0]+chosen[1])
        ids = [i for i in ids if i in selected]
        if len(ids) != 128:
            raise ValueError('Admission must contain 64 examples per label')
    repeat = args.repeat if args.mode == 'admission' else None
    run = name(args.seed, args.head_kind, repeat)
    output, report = ROOT/'checkpoints'/run, ROOT/'reports'/run
    if output.exists() or report.exists():
        raise FileExistsError('Use a new run_id; do not overwrite '+run)
    output.mkdir(parents=True);report.mkdir(parents=True)
    started = time.monotonic()
    status = {'state': 'running', 'run_id': run, 'started_utc': now(), 'mode': args.mode}
    write_json(report/'status.json', status)
    try:
        with (report/'console.log').open('x') as log, redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(f'{run}: loading same published encoder; initializing {args.head_kind} head', flush=True)
            tok = tokenizer(lock)
            _, items, stats, _, _ = prepare_rules(tok, [by_id[i] for i in ids])
            model = from_source(ROOT/lock['local_dir'], args.head_kind, args.seed)
            initial = {'encoder': state_digest(model.encoder.state_dict()),
                       'head': state_digest({n: p for n, p in model.state_dict().items() if not n.startswith('encoder.')})}
            cfg = upstream.TrainConfig(**dict(TRAINING, seed=args.seed))
            cfg.validate()
            write_json(report/'manifest.json', {'run_id': run, 'study_id': STUDY, 'mode': args.mode,
                'head_kind': args.head_kind, 'seed': args.seed, 'initialization_seed': 1000003+args.seed,
                'gradient_ids': ids, 'calibration_ids': calibration, 'gradient_truncated': sum(s['truncated'] for s in stats),
                'training': dict(TRAINING, seed=args.seed), 'initial_state_sha256': initial,
                'parameter_count': sum(p.numel() for p in model.parameters()),
                'contract_sha256': file_digest(ROOT/'configs'/f'{STUDY}.json'),
                'source_sha256': contract['source_sha256'],
                'versions': {n: version(n) for n in ('torch', 'laya', 'transformers', 'numpy', 'safetensors')},
                'cuda_build': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
                'strict_runtime': True, 'tf32_enabled': False, 'cuda_finite_guards': True,
                'encoder_reference_compile': False, 'attention_implementation': 'sdpa',
                'development_body_read': False, 'test_accessed': False})
            saved = []
            def epoch_end(index, loss):
                before_cpu, before_cuda = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                training = model.training;model.eval()
                meta = save_fp32(model, output/f'epoch-{index+1:03d}')
                model.train(training)
                if not torch.equal(before_cpu, torch.get_rng_state()) or not torch.equal(before_cuda, torch.cuda.get_rng_state()):
                    raise RuntimeError('Saving consumed training RNG')
                saved.append({'epoch': index+1, 'mean_training_loss': loss, **meta})
                write_json(report/'epoch_checkpoints.json', saved)
            model.to(device)
            torch.cuda.reset_peak_memory_stats()
            with guard_training(torch, upstream), audit_training(upstream, model, tok, items, ids, cfg, report, args.mode == 'admission'):
                losses = upstream.train_model(model, tok, items, cfg, device, 512, 192, on_epoch_end=epoch_end, parallel=False)
            final = {'encoder': state_digest(model.encoder.state_dict()),
                     'head': state_digest({n: p for n, p in model.state_dict().items() if not n.startswith('encoder.')})}
            if any(initial[n] == final[n] for n in initial):
                raise RuntimeError('Encoder or head did not update')
            reference = predict(model, tok, [by_id[i] for i in calibration], device)
            reference['state_sha256'] = saved[-1]['state_sha256']
            write_json(report/'calibration-reference.json', reference)
            write_json(report/'result.json', {'state': 'trained_pending_independent_reload', 'run_id': run,
                'losses': losses, 'initial_state_sha256': initial, 'final_state_sha256': final,
                'checkpoint': str((output/'epoch-002').relative_to(ROOT)),
                'checkpoint_state_sha256': saved[-1]['state_sha256'],
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                'seconds': time.monotonic()-started, 'test_accessed': False})
            status.update(state='trained_pending_independent_reload', finished_utc=now(), seconds=time.monotonic()-started)
    except BaseException as exc:
        status.update(state='failed', error=repr(exc), seconds=time.monotonic()-started)
        raise
    finally:
        write_json(report/'status.json', status)


def reload_or_evaluate(args, contract, lock, device):
    import numpy as np
    from laya_head_control import load_fp32, fit_temperature
    repeat = args.repeat if args.admission_reload else None
    run = name(args.seed, args.head_kind, repeat)
    report = ROOT/'reports'/run
    result = json.loads((report/'result.json').read_text())
    if json.loads((report/'manifest.json').read_text())['contract_sha256'] != file_digest(ROOT/'configs'/f'{STUDY}.json'):
        raise ValueError('Contract differs from the one used for training')
    if args.mode == 'evaluate':
        gate_path = ROOT/'reports'/STUDY/'all-models-frozen.json'
        gate = json.loads(gate_path.read_text())
        if gate['contract_sha256'] != file_digest(ROOT/'configs'/f'{STUDY}.json'):
            raise ValueError('Contract differs from the common freeze gate')
        expected_names = [name(s, h) for s, h in PAIRS]
        if gate['run_ids'] != expected_names or len(gate['frozen_sha256']) != 6:
            raise ValueError('All six models must be frozen before development access')
        for n in expected_names:
            if file_digest(ROOT/'reports'/n/'frozen.json') != gate['frozen_sha256'][n]:
                raise ValueError('A frozen model changed')
        if (report/'validation.json').exists():
            raise FileExistsError('Development predictions already exist')
    elif (report/'frozen.json').exists():
        raise FileExistsError('Readout already frozen')
    model, metadata = load_fp32(ROOT/result['checkpoint'])
    if metadata['state_sha256'] != result['checkpoint_state_sha256']:
        raise ValueError('Saved state differs from the trained final model')
    model.to(device).eval()
    tok = tokenizer(lock)
    if args.mode == 'reload':
        _, _, calibration, by_id = load_training()
        reference = json.loads((report/'calibration-reference.json').read_text())
        current = predict(model, tok, [by_id[i] for i in calibration], device)
        if current['rows'] != reference['rows'] or not np.array_equal(current['logits'], reference['logits']):
            raise ValueError('Independent FP32 reload changed calibration inputs or logits')
        z = np.asarray(current['logits'], dtype=np.float64)
        fit = fit_temperature(z[:, 1]-z[:, 0], [r['true_label'] for r in current['rows']])
        write_json(report/'calibration-reloaded.json', current)
        frozen = {'state': 'frozen', 'run_id': run, 'frozen_utc': now(), 'head_kind': args.head_kind,
            'seed': args.seed, 'checkpoint': result['checkpoint'], 'checkpoint_metadata': metadata,
            'temperature_fit': fit, 'contract_sha256': file_digest(ROOT/'configs'/f'{STUDY}.json'),
            'calibration_sha256': file_digest(report/'calibration-reloaded.json'),
            'independent_reload_logits_bitwise_equal': True, 'development_body_read': False, 'test_accessed': False}
        write_json(report/'frozen.json', frozen)
        write_json(report/'status.json', {'state': 'frozen', 'run_id': run, 'finished_utc': now()})
        print(f'{run}: independent FP32 reload exact; calibrated and frozen', flush=True)
    else:
        from laya_decision_views import probabilities
        from laya_head_control import logit_metrics
        frozen = json.loads((report/'frozen.json').read_text())
        rows = list(read_jsonl(ROOT/'data/processed/validation.jsonl'))
        split = json.loads((ROOT/'reports/split_manifest.json').read_text())
        if [r['id'] for r in rows] != split['ids']['validation']:
            raise ValueError('Development ID order changed')
        prediction = predict(model, tok, rows, device)
        z = np.asarray(prediction['logits'])
        y = np.array([r['true_label'] for r in prediction['rows']])
        truncated = np.array([r['truncated'] for r in prediction['rows']])
        t = frozen['temperature_fit']['temperature']
        raw_prob, calibrated_prob = probabilities(z), probabilities(z, t)
        prediction.update(run_id=run, head_kind=args.head_kind, seed=args.seed, temperature=t,
            raw_probabilities=raw_prob.tolist(), calibrated_probabilities=calibrated_prob.tolist(),
            raw_metrics=logit_metrics(y, z, truncated),
            calibrated_metrics=logit_metrics(y, z, truncated, t))
        write_json(report/'validation.json', prediction)
        print(f'{run}: development correct={prediction["calibrated_metrics"]["all"]["correct"]}/2500', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('train', 'admission', 'reload', 'evaluate'), required=True)
    p.add_argument('--head-kind', choices=('original', 'linear'), required=True)
    p.add_argument('--seed', type=int, choices=(42, 43, 44), required=True)
    p.add_argument('--repeat', type=int, choices=(1, 2), default=1)
    p.add_argument('--admission-reload', action='store_true')
    args = p.parse_args()
    if args.mode == 'admission' and args.seed != 42:
        raise ValueError('Admission uses seed42 only')
    if args.admission_reload and (args.mode != 'reload' or args.seed != 42):
        raise ValueError('Invalid admission reload')
    # Prepare strict runtime before importing Torch or any module importing Laya.
    from laya_runtime import prepare_runtime, apply_runtime
    profile = prepare_runtime({'device': 'cuda', 'runtime_profile': 'strict', 'training': {'amp': False}})
    import train_laya
    train_laya.block_network()
    import torch
    torch.set_num_threads(4)
    apply_runtime(torch, profile)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train_laya.require_device(torch, 'cuda')
    if version('torch') != '2.14.1+cu126' or version('laya') != '0.4.1':
        raise ValueError('Require the registered CUDA 12.6 environment')
    contract, lock = verify_contract()
    device = torch.device('cuda:0')
    if args.mode in ('train', 'admission'):
        train(args, contract, lock, device)
    else:
        reload_or_evaluate(args, contract, lock, device)


if __name__ == '__main__':
    main()

"""Fixed robust-loss runs; epoch and final delivery are mandatory progress gates."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys
import time

from common import ROOT, file_digest, read_jsonl
from laya_checkpoint_delivery import atomic_json, seal_bundle, wait_for_receipt, request_sha, verify_files, validate_receipt, within

STUDY = 'laya-robust-loss-cu126-20261015-v1'
PAIRS = [(42, 'ce'), (42, 'mixed'), (43, 'mixed'), (43, 'ce'), (44, 'ce'), (44, 'mixed')]


class VerifiedEpochBoundary(Exception):
    """Admission deliberately exits only after durable first-epoch delivery."""


def now():
    return datetime.now(timezone.utc).isoformat()


def name(seed, arm, repeat=None):
    run = f'laya-robust-loss-{arm}-s{seed}-strict-fp32-20261015-v1'
    return run if repeat is None else run + f'-admission-r{repeat}'


def contract_path():
    return ROOT/'configs'/f'{STUDY}.json'


def verify_contract():
    from run_laya_head_control import TRAINING
    d = json.loads(contract_path().read_text())
    if (d['study_id'] != STUDY or d['training'] != TRAINING or d['pairs'] != [list(p) for p in PAIRS]
            or d['loss'] != {'q': .7, 'gce_mixture_weight': .5}
            or d['runtime'] != {'device': 'cuda:0', 'profile': 'strict', 'dtype': 'float32', 'amp': False, 'tf32': False}):
        raise ValueError('Fixed experiment contract differs')
    for rel, sha in d['source_sha256'].items():
        if file_digest(ROOT/rel) != sha:
            raise ValueError('Bound source changed: ' + rel)
    lock = json.loads((ROOT/'configs/laya_model_lock.json').read_text())
    for rel, sha in lock['sha256'].items():
        if file_digest(ROOT/lock['local_dir']/rel) != sha:
            raise ValueError('Published asset changed: ' + rel)
    return d, lock


def identity(seed, arm, repeat):
    return {'study_id': STUDY, 'run_id': name(seed, arm, repeat), 'seed': seed, 'arm': arm,
            'contract_sha256': file_digest(contract_path()), 'mode': 'formal' if repeat is None else 'admission'}


def assets(folder, contract, lock):
    target = folder/'assets'
    target.mkdir()
    atomic_json(target/'source-lock.json', contract)
    shutil.copyfile(ROOT/'configs/laya_model_lock.json', target/'laya_model_lock.json')
    shutil.copyfile(ROOT/'uv.lock', target/'uv.lock')
    shutil.copyfile(ROOT/'environments/laya-cu126/uv.lock', target/'cuda-uv.lock')
    shutil.copytree(ROOT/lock['local_dir']/'tokenizer', target/'tokenizer')
    for rel in contract['source_sha256']:
        # Bind data by SHA, archive executable source/locks; do not duplicate training text.
        if rel.startswith('data/'):
            continue
        dst = target/'source'/rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/rel, dst)
    shutil.copyfile(contract_path(), target/'study-contract.json')
    dst = target/'source'/'configs'/contract_path().name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(contract_path(), dst)


def request_for(run, stage):
    return json.loads((ROOT/'delivery/requests'/f'{run}.{stage}.json').read_text())


def require_delivered(run, stage, timeout):
    req = request_for(run, stage)
    verify_files(ROOT, req)
    production_receipt(req, timeout)
    return req


def production_receipt(req, timeout):
    receipt = wait_for_receipt(ROOT, req, timeout=timeout)
    if receipt.get('transport') != 'ssh' or receipt.get('minimum_free_bytes', 0) < 100*1024**3:
        raise ValueError('Production requires off-host SSH delivery with 100 GiB local reserve')
    return receipt


def train(args, contract, lock, device):
    import numpy as np
    import torch
    import laya.train as upstream
    from laya_head_control import from_source, save_fp32, load_fp32, state_digest
    from laya_robust_training import train_model, read_resume
    from laya_robust_audit import TrainingAudit
    from run_laya_head_control import load_training, tokenizer, predict, TRAINING
    from export_laya_posthoc import prepare_rules
    ident = identity(args.seed, args.arm, args.repeat)
    run = ident['run_id']
    report, output = ROOT/'reports'/run, ROOT/'checkpoints'/run
    if not args.resume:
        if output.exists() or report.exists():
            raise FileExistsError('Use a new run ID or explicitly recover the same experiment')
        report.mkdir(parents=True)
    elif not (output/'epoch-001').is_dir():
        raise ValueError('No saved epoch boundary to recover')
    atomic_json(report/'status.json', {'state': 'preparing', 'identity': ident, 'updated_utc': now()})
    start = time.monotonic()
    try:
        rows, ids, calibration, by_id = load_training()
        if args.repeat:
            chosen = {label: [i for i in ids if by_id[i]['label'] == label][:64] for label in (0, 1)}
            selected = set(chosen[0] + chosen[1])
            ids = [i for i in ids if i in selected]
        tok = tokenizer(lock)
        _, items, stats, _, _ = prepare_rules(tok, [by_id[i] for i in ids])
        cfg = upstream.TrainConfig(**dict(TRAINING, seed=args.seed))
        resuming = args.resume
        if resuming:
            require_delivered(run, 'epoch-001', args.delivery_timeout)
            model, _ = load_fp32(output/'epoch-001')
            manifest = json.loads((output/'epoch-001/manifest.json').read_text())
            if manifest['identity'] != ident or manifest['gradient_ids'] != ids:
                raise ValueError('Saved experiment identity or gradient partition changed')
        else:
            model = from_source(ROOT/lock['local_dir'], 'original', args.seed)
            manifest = {'identity': ident, 'head_kind': 'original', 'training': asdict(cfg),
                'gradient_ids': ids, 'calibration_ids': calibration,
                'gradient_truncated': sum(s['truncated'] for s in stats),
                'initial_state_sha256': {'encoder': state_digest(model.encoder.state_dict()),
                    'head': state_digest({n: p for n, p in model.state_dict().items() if not n.startswith('encoder.')})},
                'source_sha256': contract['source_sha256'], 'versions': {n: version(n) for n in ('torch','laya','transformers','numpy','safetensors')},
                'gpu': torch.cuda.get_device_name(), 'cuda_build': torch.version.cuda,
                'strict_runtime': True, 'test_accessed': False, 'development_body_read': False}
            atomic_json(report/'manifest.json', manifest)
        final_exists = (output/'epoch-002').exists()
        if final_exists:
            if not resuming:
                raise FileExistsError('Final checkpoint already exists')
            require_delivered(run, 'epoch-002', args.delivery_timeout)
            model, _ = load_fp32(output/'epoch-002')
        else:
            recovery = read_resume(output/'epoch-001/resume.pt', ident, state_digest(model.state_dict())) if resuming else None
            attempt = 1
            while (report/f'attempt-{attempt:03d}').exists():
                attempt += 1
            audit = TrainingAudit(upstream, model, tok, items, ids, cfg, report/f'attempt-{attempt:03d}', bool(args.repeat),
                                  output/'epoch-001' if resuming else None)
            model.to(device)
            torch.cuda.reset_peak_memory_stats()
            def epoch_end(index, loss, context):
                audit.flush()
                stage = f'epoch-{index+1:03d}'
                target = output/stage
                metadata = save_fp32(model, target)
                atomic_json(target/'manifest.json', manifest)
                for leaf in ('actual_inputs.jsonl', 'updates.jsonl'):
                    shutil.copyfile(audit.folder/leaf, target/leaf)
                if index == 0:
                    torch.save(context.snapshot(), target/'resume.pt')
                else:
                    final = {'encoder': state_digest(model.encoder.state_dict()),
                             'head': state_digest({n:p for n,p in model.state_dict().items() if not n.startswith('encoder.')})}
                    if any(final[k] == manifest['initial_state_sha256'][k] for k in final):
                        raise ValueError('Encoder or head did not update')
                    evidence = audit.finish()
                    atomic_json(target/'training_audit.json', evidence)
                    result = {'identity': ident, 'state': 'trained_pending_delivery', 'checkpoint': str(target.relative_to(ROOT)),
                        'checkpoint_state_sha256': metadata['state_sha256'], 'losses': list(context.history),
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                        'restored_epoch1': resuming,
                        'seconds_including_prior_delivery': time.monotonic()-start, 'test_accessed': False}
                    atomic_json(report/'result.json', result)
                    atomic_json(target/'result.json', result)
                atomic_json(target/'epoch.json', {'epoch': index+1, 'loss': loss, 'updates': context.updates,
                    'exposures': context.exposures, 'input_stream_sha256': audit.hash.hexdigest()})
                assets(target, contract, lock)
                request = seal_bundle(ROOT, str(target.relative_to(ROOT)), ident, stage)
                atomic_json(report/'status.json', {'state': 'waiting_local_delivery', 'identity': ident, 'stage': stage, 'updated_utc': now()})
                production_receipt(request, args.delivery_timeout)
                if index == 0 and args.admission_checkpoint_exit:
                    raise VerifiedEpochBoundary()
            try:
                train_model(model, tok, items, cfg, device, args.arm, ident, epoch_end, audit, recovery)
            finally:
                audit.close()
        model.to(device).eval()
        # Reference may be generated after recovering a fully saved epoch-002; record this.
        reference_path = report/'calibration-reference.json'
        if not reference_path.exists():
            reference = predict(model, tok, [by_id[i] for i in calibration], device)
            reference['state_sha256'] = state_digest(model.state_dict())
            reference['generated_after_epoch2_recovery'] = bool(final_exists)
            atomic_json(reference_path, reference)
        atomic_json(report/'status.json', {'state': 'trained_delivered_pending_independent_reload', 'identity': ident, 'updated_utc': now()})
    except VerifiedEpochBoundary:
        atomic_json(report/'status.json', {'state': 'awaiting_explicit_epoch_resume', 'identity': ident, 'updated_utc': now()})
        return
    except BaseException as exc:
        from laya_checkpoint_delivery import DeliveryBlocked
        atomic_json(report/'status.json', {'state': 'blocked_delivery' if isinstance(exc, DeliveryBlocked) else 'failed',
            'identity': ident, 'error': repr(exc), 'updated_utc': now()})
        raise


def reload_or_evaluate(args, contract, lock, device):
    import numpy as np
    from laya_head_control import load_fp32, fit_temperature, logit_metrics
    from run_laya_head_control import load_training, tokenizer, predict
    ident = identity(args.seed, args.arm, args.repeat)
    run = ident['run_id']; report = ROOT/'reports'/run; output = ROOT/'checkpoints'/run
    require_delivered(run, 'epoch-002', args.delivery_timeout)
    model, metadata = load_fp32(output/'epoch-002')
    model.to(device).eval(); tok = tokenizer(lock)
    if args.mode == 'reload':
        if (output/'frozen').exists():
            # Do not regenerate a sealed freeze; resend its request/receipt if needed.
            require_delivered(run, 'frozen', args.delivery_timeout)
            return
        _, _, calibration, by_id = load_training()
        reference = json.loads((report/'calibration-reference.json').read_text())
        current = predict(model, tok, [by_id[i] for i in calibration], device)
        if (reference['state_sha256'] != metadata['state_sha256'] or reference['rows'] != current['rows']
                or not np.array_equal(reference['logits'], current['logits'])):
            raise ValueError('Independent FP32 reload changed calibration inputs/logits')
        z = np.asarray(current['logits'], dtype=np.float64)
        frozen = {'identity': ident, 'state': 'frozen', 'frozen_utc': now(),
                  'checkpoint_state_sha256': metadata['state_sha256'],
                  'temperature_fit': fit_temperature(z[:,1]-z[:,0], [r['true_label'] for r in current['rows']]),
                  'independent_reload_logits_bitwise_equal': True, 'development_body_read': False, 'test_accessed': False}
        stage = output/'frozen'; stage.mkdir()
        atomic_json(stage/'frozen.json', frozen)
        atomic_json(stage/'calibration-reference.json', reference)
        atomic_json(stage/'calibration-reloaded.json', current)
        prior = request_for(run, 'epoch-002')
        request = seal_bundle(ROOT, str(stage.relative_to(ROOT)), ident, 'frozen',
                              checkpoint_bundle=str((output/'epoch-002').relative_to(ROOT)), epoch_request_sha256=request_sha(prior))
        production_receipt(request, args.delivery_timeout)
        atomic_json(report/'status.json', {'state': 'frozen_delivered', 'identity': ident, 'updated_utc': now()})
    else:
        if args.repeat:
            raise ValueError('Admission must never access development data')
        gate = json.loads((ROOT/'reports'/STUDY/'all-models-frozen.json').read_text())
        if gate['contract_sha256'] != file_digest(contract_path()) or gate['run_ids'] != [name(s,a) for s,a in PAIRS]:
            raise ValueError('Invalid six-model freeze gate')
        for seed, arm in PAIRS:
            candidate = name(seed, arm)
            req = request_for(candidate, 'frozen')
            if gate['freeze_request_sha256'][candidate] != request_sha(req):
                raise ValueError('A frozen model changed')
            receipt = production_receipt(req, 0)
            validate_receipt(req, receipt)
        if (report/'validation.json').exists():
            require_delivered(run, 'evaluated', args.delivery_timeout)
            return
        frozen = json.loads((output/'frozen/frozen.json').read_text())
        rows = list(read_jsonl(ROOT/'data/processed/validation.jsonl'))
        split = json.loads((ROOT/'reports/split_manifest.json').read_text())
        if [r['id'] for r in rows] != split['ids']['validation']:
            raise ValueError('Development partition changed')
        prediction = predict(model, tok, rows, device)
        logits = np.asarray(prediction['logits']); labels = [r['true_label'] for r in prediction['rows']]
        truncated = [r['truncated'] for r in prediction['rows']]; t = frozen['temperature_fit']['temperature']
        prediction.update(identity=ident, temperature=t, raw_metrics=logit_metrics(labels, logits, truncated),
                          calibrated_metrics=logit_metrics(labels, logits, truncated, t))
        stage = output/'evaluated'; stage.mkdir()
        atomic_json(stage/'validation.json', prediction)
        atomic_json(stage/'frozen.json', frozen)
        prior = request_for(run, 'epoch-002')
        req = seal_bundle(ROOT, str(stage.relative_to(ROOT)), ident, 'evaluated',
                          checkpoint_bundle=str((output/'epoch-002').relative_to(ROOT)), epoch_request_sha256=request_sha(prior))
        atomic_json(report/'validation.json', prediction)
        production_receipt(req, args.delivery_timeout)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=('train','reload','evaluate'), required=True)
    p.add_argument('--arm', choices=('ce','mixed'), required=True)
    p.add_argument('--seed', type=int, choices=(42,43,44), required=True)
    p.add_argument('--repeat', type=int, choices=(1,2))
    p.add_argument('--resume', action='store_true')
    p.add_argument('--delivery-timeout', type=float, default=10800)
    p.add_argument('--admission-checkpoint-exit', action='store_true', help='Admission only: exit after verified epoch-001')
    args = p.parse_args()
    if args.repeat and (args.seed != 42 or args.mode == 'evaluate'):
        p.error('Admission uses seed42 and never evaluates development')
    if args.resume and args.mode != 'train':
        p.error('--resume only recovers an interrupted training/delivery stage')
    if args.admission_checkpoint_exit and (args.repeat is None or args.mode != 'train'):
        p.error('Controlled boundary exit is restricted to CUDA admission')
    from laya_runtime import prepare_runtime, apply_runtime
    profile = prepare_runtime({'device':'cuda', 'runtime_profile':'strict', 'training':{'amp':False}})
    import train_laya
    train_laya.block_network()
    import torch
    torch.set_num_threads(4); apply_runtime(torch, profile)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    train_laya.require_device(torch, 'cuda')
    if version('torch') != '2.14.1+cu126' or version('laya') != '0.4.1':
        raise ValueError('Require locked CUDA 12.6 environment')
    contract, lock = verify_contract()
    if args.repeat is None and args.mode == 'train':
        from run_laya_robust_study import check_admission
        admission = json.loads((ROOT/'reports'/STUDY/'admission.json').read_text())
        if admission.get('state') != 'passed' or admission.get('contract_sha256') != file_digest(contract_path()):
            raise ValueError('Formal entry requires current CUDA and local delivery admission')
        check_admission()
    if args.mode == 'train':
        train(args, contract, lock, torch.device('cuda:0'))
    else:
        reload_or_evaluate(args, contract, lock, torch.device('cuda:0'))


if __name__ == '__main__':
    main()

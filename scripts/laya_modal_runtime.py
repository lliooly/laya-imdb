"""Modal transport adapter; the frozen SSH protocol and scientific loop stay intact."""
import argparse
import json

from common import ROOT, file_digest
from laya_checkpoint_delivery import wait_for_receipt

STUDY = 'laya-robust-loss-modal-20261015-v3'
VOLUME = 'end-imdb-laya-robust-20261015-v3'
CAMPAIGN = 'laya-robust-loss-modal-20261015'
MINIMUM_FREE = 100 * 1024**3


def name(seed, arm, repeat=None):
    result = f'laya-robust-loss-modal-{arm}-s{seed}-strict-fp32-20261015-v3'
    return result if repeat is None else result + f'-admission-r{repeat}'


def contract_path():
    return ROOT/'configs'/f'{STUDY}.json'


def production_receipt(request, timeout):
    receipt = wait_for_receipt(ROOT, request, timeout=timeout)
    if (receipt.get('transport') != 'modal-volume'
            or receipt.get('modal_volume') != VOLUME
            or receipt.get('minimum_free_bytes', 0) < MINIMUM_FREE):
        raise ValueError('Require verified off-cloud Modal delivery and 100 GiB local reserve')
    return receipt


def poll_call(call):
    """Modal 1.2.6 uses builtin TimeoutError for an unfinished output poll.

    modal.exception.TimeoutError is a different type (remote execution timeout).
    """
    try:
        return True, call.get(timeout=0)
    except TimeoutError:
        return False, None


def install():
    import run_laya_robust_loss as runner
    import run_laya_robust_study as study
    # Both modules imported their functions by name; replace the transport boundary
    # explicitly in both. No monkeypatch of tensors, loss, RNG or optimizer occurs.
    for module in (runner, study):
        module.STUDY = STUDY
        module.name = name
        module.contract_path = contract_path
        module.production_receipt = production_receipt
    return runner, study


def prepare():
    original = ROOT/'configs/laya-robust-loss-cu126-20261015-v1.json'
    d = json.loads(original.read_text())
    for rel, sha in d['source_sha256'].items():
        if file_digest(ROOT/rel) != sha:
            raise ValueError('Frozen source changed: ' + rel)
    d.update(study_id=STUDY, platform='modal', modal_volume=VOLUME,
             parent_contract_sha256=file_digest(original),
             cuda_admission_passed=False, state='prepared_pending_modal_admission',
             modal_budget={'confirmed_free_credit_usd': 30, 'maximum_compute_usd': 24,
                           'conservative_usd_per_hour': 3, 'gpu': 'L40S',
                           'cpu': 4, 'memory_mib': 32768,
                           'automatic_retries': 0, 'maximum_concurrent_gpus': 1})
    for p in [original, *sorted((ROOT/'scripts').glob('*modal*.py')),
              *sorted((ROOT/'tests').glob('*modal*.py')),
              ROOT/'environments/laya-modal/pyproject.toml', ROOT/'environments/laya-modal/uv.lock']:
        d['source_sha256'][str(p.relative_to(ROOT))] = file_digest(p)
    from laya_checkpoint_delivery import atomic_json
    if contract_path().exists():
        if json.loads(contract_path().read_text()) != d:
            raise FileExistsError('Modal source contract changed; create a new version')
    else:
        atomic_json(contract_path(), d)
    return d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepare', action='store_true')
    args, remaining = p.parse_known_args()
    if args.prepare:
        if remaining:
            p.error('Unexpected arguments')
        print(json.dumps({'study': STUDY, 'bound_files': len(prepare()['source_sha256'])}))
        return
    import sys
    sys.argv = [sys.argv[0], *remaining]
    runner, _ = install()
    runner.main()


if __name__ == '__main__':
    main()

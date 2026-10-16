"""Guards for completing only the preregistered five remaining evaluations."""
from pathlib import Path

from laya_checkpoint_delivery import digest, request_sha
from laya_modal_deferred import name
from laya_modal_recovery import STUDY, VOLUME, accepted, read, verify_sources

COMPLETION = 'laya-robust-loss-modal-evaluation-20261016-v1'
JOBS = tuple({'seed': s, 'arm': a, 'evaluate': True}
             for s, a in ((42, 'mixed'), (43, 'mixed'), (43, 'ce'), (44, 'ce'), (44, 'mixed')))
CAP = 25.0
RESERVATION = 1.0
RATE = 3.0


def reserve(ledger):
    """A single extra dollar from the originally authorized $30, never a reset."""
    if any(j['state'] == 'reserved' for j in ledger['jobs']):
        raise ValueError('Unresolved compute reservation; refuse duplicate launch')
    if any(j['run_id'] == COMPLETION for j in ledger['jobs']):
        raise ValueError('Evaluation completion already submitted')
    if ledger['upper_bound_used_usd'] != 24.0 or ledger['maximum_compute_usd'] != 24.0:
        raise ValueError('Previous budget changed; require reconciliation')
    record = {'run_id': COMPLETION, 'state': 'reserved', 'mode': 'evaluate_only',
              'previous_budget_usd': 24.0, 'reserved_usd': RESERVATION,
              'total_compute_cap_usd': CAP, 'original_user_confirmed_free_usd': 30.0}
    ledger['jobs'].append(record)
    ledger['maximum_compute_usd'] = ledger['upper_bound_used_usd'] = CAP
    return record


def verify_scope(root):
    root = Path(root)
    config = read(root/'configs'/f'{COMPLETION}.json')
    if (config['completion_id'] != COMPLETION or config['scientific_study_id'] != STUDY
            or config['modal_volume'] != VOLUME or config['jobs'] != list(JOBS)
            or config['maximum_additional_compute_usd'] != RESERVATION
            or config['maximum_total_compute_usd'] != CAP):
        raise ValueError('Evaluation-only scope or budget changed')
    for rel, sha in config['operational_source_sha256'].items():
        if digest(root/rel) != sha:
            raise ValueError('Evaluation operational source changed: '+rel)
    verify_sources(root, read(root/'configs/laya-robust-loss-modal-recovery-20261016-v1.json'))
    gate = read(root/'reports'/STUDY/'all-models-frozen.json')
    expected = [name(s, a) for s, a in ((42, 'ce'), (42, 'mixed'), (43, 'mixed'), (43, 'ce'), (44, 'ce'), (44, 'mixed'))]
    if gate['run_ids'] != expected or gate['contract_sha256'] != config['scientific_contract_sha256']:
        raise ValueError('Six-model common freeze gate changed')
    for run in expected:
        for stage in ('epoch-002', 'frozen'):
            if not accepted(root, run, stage):
                raise ValueError('Unverified final model: '+run)
        req = read(root/'delivery/requests'/f'{run}.frozen.json')
        if request_sha(req) != gate['freeze_request_sha256'][run]:
            raise ValueError('Frozen model differs from common gate')
    return config

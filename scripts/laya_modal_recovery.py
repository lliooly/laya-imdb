"""Crash recovery for the unchanged deferred-v2 scientific experiment."""
import json
from pathlib import Path
import time
import uuid

from laya_checkpoint_delivery import (atomic_json, digest, request_sha,
                                      validate_request, verify_files)
from laya_modal_deferred import STUDY, VOLUME, name, validate_cloud_receipt

RECOVERY = 'laya-robust-loss-modal-recovery-20261016-v1'
RATE = 3.0
CAP = 24.0


def read(path):
    return json.loads(Path(path).read_text())


def sealed(root, run, stage):
    path = Path(root)/'delivery/requests'/f'{run}.{stage}.json'
    if not path.exists():
        return None
    req = read(path)
    validate_request(req)
    contract = Path(root)/'configs'/f'{STUDY}.json'
    if (req['identity']['run_id'] != run or req['stage'] != stage
            or req['identity'].get('study_id') != STUDY
            or req['identity']['contract_sha256'] != digest(contract)):
        raise ValueError('Recovery request belongs to another experiment')
    verify_files(root, req)
    return req


def accepted(root, run, stage):
    req = sealed(root, run, stage)
    if req is None:
        return False
    for channel in ('acks', 'receipts'):
        path = Path(root)/'delivery'/channel/f'{request_sha(req)}.json'
        if not path.exists():
            return False
        validate_cloud_receipt(req, read(path))
    return True


def job_done(root, job):
    run = name(job['seed'], job['arm'])
    stage = 'evaluated' if job['evaluate'] else 'frozen'
    if not accepted(root, run, stage):
        return False
    # A tiny frozen/evaluation bundle alone cannot establish the weight bytes.
    if not accepted(root, run, 'epoch-002'):
        raise ValueError('Final receipt lacks its accepted full checkpoint')
    req = read(Path(root)/'delivery/requests'/f'{run}.{stage}.json')
    prior = read(Path(root)/'delivery/requests'/f'{run}.epoch-002.json')
    if req['epoch_request_sha256'] != request_sha(prior):
        raise ValueError('Final bundle refers to another checkpoint')
    return True


def quarantine(root, paths, reason):
    """Retain incomplete writes; never move a sealed bundle or overwrite history."""
    root = Path(root)
    target = root/'artifacts'/RECOVERY/'interrupted-writes'/uuid.uuid4().hex
    moved = []
    for path in paths:
        path = Path(path)
        if path.exists():
            dest = target/path.relative_to(root)
            dest.parent.mkdir(parents=True, exist_ok=True)
            path.rename(dest)
            moved.append(str(path.relative_to(root)))
    if moved:
        atomic_json(target/'recovery.json', {'reason': reason, 'paths': moved,
                                           'preserved_unix': time.time()})
    return moved


def plan(root, job):
    """Choose the next valid epoch boundary without rerunning completed models."""
    root = Path(root)
    run = name(job['seed'], job['arm'])
    output, report = root/'checkpoints'/run, root/'reports'/run
    if job_done(root, job):
        return []
    if job['evaluate']:
        if not accepted(root, run, 'frozen'):
            raise ValueError('Cannot evaluate an unfrozen model')
        if sealed(root, run, 'evaluated') is not None:
            raise ValueError('Repair the sealed evaluation receipt before planning')
        quarantine(root, [output/'evaluated', report/'validation.json'], 'incomplete evaluation')
        return [('evaluate', False)]
    final = sealed(root, run, 'epoch-002')
    if final is not None:
        if not accepted(root, run, 'epoch-002'):
            raise ValueError('Repair the final epoch receipt before planning')
        if sealed(root, run, 'frozen') is not None:
            raise ValueError('Repair the sealed frozen receipt before planning')
        quarantine(root, [output/'frozen'], 'incomplete independent reload')
        # The frozen driver recovers a completed epoch-002 and produces a missing
        # calibration reference without taking any additional optimizer steps.
        return [('train', True), ('reload', False)]
    first = sealed(root, run, 'epoch-001')
    if first is not None:
        if not accepted(root, run, 'epoch-001'):
            raise ValueError('Repair first epoch receipt before planning')
        quarantine(root, [output/'epoch-002', output/'frozen',
                          report/'calibration-reference.json', report/'result.json'],
                   'resume from accepted epoch-001; retain incomplete later writes')
        return [('train', True), ('reload', False)]
    quarantine(root, [output, report], 'interrupted before first sealed epoch; deterministic restart')
    return [('train', False), ('reload', False)]


class BudgetExpired(RuntimeError):
    pass


def initial_state(config, now):
    seconds = (CAP-config['previous_upper_usd'])/RATE*3600
    if seconds < 1800:
        raise BudgetExpired('Less than 30 minutes remain under the original compute cap')
    return {'schema': 'laya-modal-recovery-v1', 'recovery_id': RECOVERY,
            'scientific_study': STUDY, 'started_unix': now,
            'deadline_unix': now+seconds-120, 'previous_upper_usd': config['previous_upper_usd'],
            'completed': {}, 'active': None, 'state': 'ready', 'local_delivery_verified': False}


def check_state(state, config):
    if (state['schema'] != 'laya-modal-recovery-v1' or state['recovery_id'] != RECOVERY
            or state['scientific_study'] != STUDY
            or state['previous_upper_usd'] != config['previous_upper_usd']
            or state['deadline_unix'] > state['started_unix'] +
                (CAP-state['previous_upper_usd'])/RATE*3600):
        raise ValueError('Persisted recovery identity/budget changed')


def execute_queue(jobs, state, *, save, spawn, attach, done, clock=time.time):
    """Persist intent/call ID and reattach after coordinator interruption.

    A crash in the spawn-to-record window fails closed. It must never silently
    submit another billable job merely because the call ID is unavailable.
    """
    for job in jobs:
        key = f'{job["seed"]}-{job["arm"]}-' + ('evaluate' if job['evaluate'] else 'freeze')
        active = state['active']
        if active is not None and active['key'] != key:
            # Completed prefix entries are encountered again during recovery.
            if key in state['completed']:
                if not done(job):
                    raise ValueError('Previously completed checkpoint no longer validates')
                continue
            raise ValueError('Active call does not match the first incomplete job')
        if active is None:
            verified = done(job)
            if key in state['completed'] and not verified:
                raise ValueError('Previously completed checkpoint no longer validates')
            if verified:
                state['completed'].setdefault(key, {'state': 'already_verified_cloud'})
                save(state)
                continue
            if clock() >= state['deadline_unix']-180:
                state['state'] = 'stopped_budget'; save(state)
                raise BudgetExpired('Original compute deadline reached; preserve all checkpoints')
            active = {'key': key, 'job': job, 'submitted_unix': clock(), 'call_id': None}
            state.update(active=active, state='submitting'); save(state)
            call = spawn(job, state['deadline_unix'])
            active['call_id'] = call.object_id
            state['state'] = 'running'; save(state)
        else:
            if not active['call_id']:
                raise RuntimeError('Unresolved submission intent; inspect call graph before any new GPU launch')
            call = attach(active['call_id'])
        try:
            while True:
                if clock() >= state['deadline_unix']-60:
                    call.cancel(terminate_containers=True)
                    state['state'] = 'stopped_budget'; save(state)
                    raise BudgetExpired('Compute deadline reached; active worker cancelled')
                try:
                    result = call.get(timeout=10)
                    break
                except TimeoutError:
                    continue
        except KeyboardInterrupt:
            state['state'] = 'coordinator_interrupted'; save(state)
            # Leave the persisted child alive; the next invocation attaches to it.
            raise
        if not done(job):
            raise ValueError('Worker returned without the required verified checkpoint')
        state['completed'][key] = result
        state.update(active=None, state='between_jobs'); save(state)


def verify_sources(root, config):
    root = Path(root)
    if (config.get('schema') != 'laya-modal-operational-recovery-v1'
            or config.get('recovery_id') != RECOVERY
            or config.get('scientific_study_id') != STUDY
            or config.get('modal_volume') != VOLUME
            or config.get('maximum_compute_usd') != CAP
            or config.get('conservative_usd_per_hour') != RATE):
        raise ValueError('Operational recovery identity or original compute cap changed')
    parent_path = root/'configs'/f'{STUDY}.json'
    if digest(parent_path) != config['scientific_contract_sha256']:
        raise ValueError('Scientific contract changed')
    parent = read(parent_path)
    for rel, sha in {**parent['source_sha256'], **config['recovery_source_sha256']}.items():
        if digest(root/rel) != sha:
            raise ValueError('Recovery bound source changed: '+rel)
    return parent

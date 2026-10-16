"""One bounded GPU pilot with persistent files and mandatory local weight delivery."""
import json
from pathlib import Path
import sys
import time

import modal

ROOT = Path(__file__).resolve().parents[1] if modal.is_local() else Path('/workspace/End-IMDB')
sys.path.insert(0, str(ROOT/'scripts'))
from laya_long_context_pilot import STUDY
from modal_laya_recovery import image, volume

app = modal.App(STUDY)


@app.function(image=image, gpu='L40S', cpu=4, memory=32768,
              volumes={'/workspace': volume}, timeout=900, startup_timeout=120,
              retries=0, max_containers=1, scaledown_window=2,
              include_source=False, serialized=True)
def run_pilot():
    import subprocess
    from common import file_digest, write_json
    root = Path('/workspace/End-IMDB')
    volume.reload()
    report = root/'reports'/STUDY
    report.mkdir(parents=True, exist_ok=True)
    python = '/.uv/.venv/bin/python'
    try:
        with (report/'console.log').open('x') as log:
            process = subprocess.run([python, '-u', str(root/'scripts/laya_long_context_pilot.py')],
                                     cwd=root, stdout=log, stderr=subprocess.STDOUT, timeout=780)
        if process.returncode:
            print((report/'console.log').read_text()[-10000:], flush=True)
            raise RuntimeError(f'Pilot exited {process.returncode}')
        summary = json.loads((report/'summary.json').read_text())
        files = []
        for base in (report, root/'checkpoints'/STUDY):
            if base.exists():
                for p in sorted(base.rglob('*')):
                    if p.is_file():
                        files.append({'path': str(p.relative_to(root)), 'bytes': p.stat().st_size,
                                      'sha256': file_digest(p)})
        delivery = {'state': 'committed_cloud_pending_local', 'study_id': STUDY,
                    'local_delivery_verified': False, 'files': files}
        write_json(report/'delivery.json', delivery)
        volume.commit()
        return {'summary': summary, 'delivery': delivery}
    finally:
        volume.commit()


@app.local_entrypoint()
def main():
    import shutil
    import subprocess
    from laya_checkpoint_delivery import atomic_json
    from laya_long_context_pilot import verify_contract
    from laya_modal_delivery import copy_from_volume
    contract, _ = verify_contract(ROOT)
    if shutil.disk_usage(ROOT).free < 15*1024**3:
        raise RuntimeError('Need 15 GiB free for verified pilot delivery')
    ledger_path = ROOT/'delivery/modal-budget.json'
    ledger = json.loads(ledger_path.read_text())
    if (ledger['upper_bound_used_usd'] != 25 or ledger['maximum_compute_usd'] != 25
            or any(j['state'] == 'reserved' or j['run_id'] == STUDY for j in ledger['jobs'])):
        raise ValueError('Budget changed, unresolved reservation, or duplicate submission')
    with volume.batch_upload(force=True) as batch:
        # Existing frozen model/data assets already reside on this volume.
        for rel in contract['upload_files']:
            batch.put_file(ROOT/rel, '/End-IMDB/'+rel)
        batch.put_file(ROOT/'configs'/f'{STUDY}.json', '/End-IMDB/configs/'+STUDY+'.json')
    record = {'run_id': STUDY, 'state': 'reserved', 'mode': 'long_context_pilot_only',
              'reserved_usd': 1., 'previous_budget_usd': 25., 'submitted_unix': time.time(),
              'maximum_seconds': 900, 'maximum_startup_seconds': 120}
    ledger['jobs'].append(record)
    ledger['upper_bound_used_usd'] = ledger['maximum_compute_usd'] = 26.
    atomic_json(ledger_path, ledger)
    call = run_pilot.spawn()
    record['function_call_id'] = call.object_id
    atomic_json(ledger_path, ledger)
    atomic_json(ROOT/'reports'/STUDY/'submission.json', record)
    print('SUBMITTED', call.object_id, flush=True)
    try:
        returned = call.get()
    except BaseException:
        # Keep the full reservation; never infer a stopped GPU from a lost client.
        raise
    record.update(state='returned_confirmed', charged_upper_usd=1.)
    atomic_json(ledger_path, ledger)
    delivery = returned['delivery']
    for file in delivery['files']:
        copy_from_volume(volume, file['path'], ROOT/file['path'], file)
    bundle = ROOT/'checkpoints'/STUDY/'pilot-2048'
    if bundle.exists():
        subprocess.run([str(ROOT/'.venv/bin/python'), str(ROOT/'scripts/laya_long_context_pilot.py'),
                        '--verify', str(bundle), '--device', 'cpu'], check=True, timeout=300)
        delivery.update(state='verified_local', local_delivery_verified=True)
    else:
        delivery.update(state='no_checkpoint_admission_failed', local_delivery_verified=False)
    atomic_json(ROOT/'reports'/STUDY/'local-delivery.json', delivery)
    print(json.dumps({'state': returned['summary']['state'], 'delivery': delivery['state'],
                      'cases': returned['summary']['cases']}), flush=True)

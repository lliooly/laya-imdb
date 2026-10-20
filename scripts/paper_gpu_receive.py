"""Receive a fixed completed study into a new isolated directory, with SHA checks.

Uses the existing authenticated OpenSSH control connection; contains no password.
No remote deletion, deployment or credential installation is performed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
STUDY = 'paper-gpu-supplement-20261019-v1'
REMOTE = 'PxqE7xkQ-默认项目-默认的单位/root/sj-tmp/imdb-paper-20261019'
ACCESS = json.loads((ROOT/'artifacts'/STUDY/'remote-access.json').read_text())
TARGET = ACCESS['target']
CONTROL = ACCESS['control_path']
SEEDS = {'baseline': (43, 143, 243), 'combined': (42, 142, 242), 'deberta': (42, 142, 242)}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    partial = path.with_suffix('.partial')
    partial.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(partial, path)


class Receiver:
    def __init__(self, out):
        self.out = out
        self.index = 0
        (out/'transport').mkdir()

    def transfer(self, commands, timeout=1800):
        self.index += 1
        base = self.out/'transport'/f'{self.index:05d}'
        batch = base.with_suffix('.sftp')
        log = base.with_suffix('.log')
        batch.write_text('\n'.join(commands+['bye'])+'\n')
        with log.open('w') as f:
            proc = subprocess.Popen(['sftp', '-b', str(batch), '-R', '128', '-B', '131072',
                '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-o', 'ConnectionAttempts=1',
                '-o', 'ControlPath='+CONTROL, '-P', '2020', TARGET],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                start_new_session=True)
            started = time.monotonic()
            finished_commands = False
            try:
                while proc.poll() is None:
                    content = log.read_text(errors='replace')
                    # This gateway sometimes keeps the SSH channel open after
                    # sftp has completed every command. The following command
                    # echo is reached only after previous transfers complete.
                    if 'sftp> bye' in content:
                        finished_commands = True
                        break
                    if time.monotonic()-started > timeout:
                        raise TimeoutError('SFTP transfer timeout; retain remote originals')
                    time.sleep(.5)
                finished_commands = finished_commands or 'sftp> bye' in log.read_text(errors='replace')
                if not finished_commands and proc.poll() != 0:
                    raise RuntimeError('SFTP did not complete; inspect '+str(log))
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGTERM)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()

    def get(self, relative, local):
        local.parent.mkdir(parents=True, exist_ok=True)
        if local.exists():
            raise FileExistsError('Never overwrite an existing received file: '+str(local))
        q = lambda v: json.dumps(str(v), ensure_ascii=False)
        self.transfer(['get '+q(REMOTE+'/'+relative)+' '+q(local)])

    def checkpoint(self, kind, seed):
        run = f'{STUDY}-{kind}-s{seed}'
        folder = self.out/'checkpoints'/run/'epoch-002'
        folder.mkdir(parents=True, exist_ok=False)
        relative = 'checkpoints/'+run+'/epoch-002/'
        self.get(relative+'bundle.json', folder/'bundle.json')
        manifest = json.loads((folder/'bundle.json').read_text())
        if (manifest['format'] != 'paper-supplement-bundle-v1' or manifest['run_id'] != run
                or manifest['kind'] != kind or manifest['seed'] != seed or manifest['epoch'] != 2
                or manifest['pilot']):
            raise ValueError('Unexpected checkpoint identity')
        files = manifest['files']
        if len({r['path'] for r in files}) != len(files) or sum(r['bytes'] for r in files) > 3*1024**3:
            raise ValueError('Invalid checkpoint file inventory')
        commands = []
        for r in files:
            path = PurePosixPath(r['path'])
            if (path.is_absolute() or '..' in path.parts or any(ord(x) < 32 for x in str(path))
                    or str(path) in ('', '.') or not 0 <= r['bytes'] <= 2*1024**3):
                raise ValueError('Unsafe checkpoint file path or size')
            local = folder/str(path)
            local.parent.mkdir(parents=True, exist_ok=True)
            q = lambda v: json.dumps(str(v), ensure_ascii=False)
            commands.append('get '+q(REMOTE+'/'+relative+str(path))+' '+q(local))
        self.transfer(commands, timeout=10800)
        for r in files:
            local = folder/r['path']
            if local.stat().st_size != r['bytes'] or sha(local) != r['sha256']:
                raise ValueError('Received checkpoint differs: '+r['path'])
        receipt = dict(state='verified', run_id=run, file_count=len(files),
                       bundle_sha256=sha(folder/'bundle.json'), files=files,
                       bytes=sum(r['bytes'] for r in files), remote_deletion_performed=False)
        write(folder.parent/'local-delivery.json', receipt)
        return receipt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    if not out.is_relative_to((ROOT/'artifacts'/STUDY).resolve()):
        raise ValueError('Receipts must remain under this study isolated artifact directory')
    out.mkdir(parents=True, exist_ok=False)
    receiver = Receiver(out)
    started = time.time()
    try:
        while time.time()-started < 72*3600:
            snapshot = out/'status'/f'{receiver.index+1:05d}.json'
            try:
                receiver.get('reports/'+STUDY+'/schedule-status-v2.json', snapshot)
                state = json.loads(snapshot.read_text())
            except (RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
                write(out/'receiver-status.json', dict(state='waiting_for_ssh_connection', error=str(exc)))
                time.sleep(60)
                continue
            write(out/'receiver-status.json', dict(state='waiting_for_training', remote=state))
            if state['state'] == 'failed':
                raise RuntimeError('Remote fixed schedule failed; preserve all files and investigate')
            if state['state'] == 'completed':
                break
            time.sleep(60)
        else:
            raise TimeoutError('Receiver reached its 72-hour bound')
        receipts = []
        for repeat in range(3):
            for kind in ('deberta', 'baseline', 'combined'):
                receipts.append(receiver.checkpoint(kind, SEEDS[kind][repeat]))
                write(out/'receiver-status.json', dict(state='receiving_checkpoints',
                    verified_checkpoint_count=len(receipts)))
        # Benchmarking runs in a separate bounded process after the training
        # schedule. Wait for its receipt before taking the final report snapshot.
        while time.time()-started < 72*3600:
            snapshot = out/'status'/f'postprocessing-{receiver.index+1:05d}.json'
            try:
                receiver.get('reports/'+STUDY+'/postprocessing-status.json', snapshot)
                post = json.loads(snapshot.read_text())
            except (RuntimeError, TimeoutError, json.JSONDecodeError) as exc:
                write(out/'receiver-status.json', dict(state='waiting_for_benchmark_receipt', error=str(exc),
                    verified_checkpoint_count=len(receipts)))
                time.sleep(60)
                continue
            if post['state'] == 'failed':
                raise RuntimeError('Remote benchmark failed; retain all received and remote files')
            if post['state'] == 'completed':
                break
            write(out/'receiver-status.json', dict(state='waiting_for_benchmark', remote=post,
                verified_checkpoint_count=len(receipts)))
            time.sleep(60)
        else:
            raise TimeoutError('Receiver reached its 72-hour bound while waiting for benchmark')
        (out/'reports').mkdir()
        q = lambda v: json.dumps(str(v), ensure_ascii=False)
        commands = ['get -r '+q(REMOTE+'/reports/'+STUDY)+' '+q(out/'reports')]
        for repeat in range(3):
            for kind in SEEDS:
                run = f'{STUDY}-{kind}-s{SEEDS[kind][repeat]}'
                commands.append('get -r '+q(REMOTE+'/reports/'+run)+' '+q(out/'reports'))
        receiver.transfer(commands)
        summary = json.loads((out/'reports'/STUDY/'summary.json').read_text())
        if summary['state'] != 'completed' or len(summary['results']) != 3 or summary['official_test_accessed']:
            raise ValueError('Unexpected final summary')
        if not (out/'reports'/STUDY/'benchmark.json').exists():
            raise RuntimeError('Nine-run results delivered but GPU cost benchmark is still missing')
        write(out/'local-delivery.json', dict(state='verified', checkpoints=receipts,
            final_summary_sha256=sha(out/'reports'/STUDY/'summary.json'), official_test_accessed=False,
            remote_deletion_performed=False))
        write(out/'receiver-status.json', dict(state='completed', checkpoint_count=len(receipts)))
    except BaseException as exc:
        write(out/'receiver-status.json', dict(state='failed', error=repr(exc), remote_files_preserved=True))
        raise


if __name__ == '__main__':
    main()

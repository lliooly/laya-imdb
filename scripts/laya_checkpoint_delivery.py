"""Immutable bundles, verified resumable copies and durable epoch acknowledgments.

The producer never treats remote files or a transfer exit code as local delivery.
SSH authentication is handled by OpenSSH; no credentials are stored here.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import select
import shlex
import shutil
import stat
import subprocess
import sys
import time

SCHEMA = 'laya-local-delivery-v1'
CHUNK = 4 * 1024 * 1024


class DeliveryBlocked(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(path, count=None):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while count is None or count > 0:
            data = f.read(CHUNK if count is None else min(CHUNK, count))
            if not data:
                if count:
                    raise ValueError('Prefix exceeds file size')
                break
            h.update(data)
            if count is not None:
                count -= len(data)
    return h.hexdigest()


def safe_relative(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('Invalid relative path')
    p = PurePosixPath(value)
    if p.is_absolute() or any(x in ('', '.', '..') for x in value.split('/')):
        raise ValueError('Unsafe delivery path')
    return p


def within(root, relative):
    root = Path(root).resolve()
    rel = safe_relative(relative)
    p = root
    for part in rel.parts:
        p = p / part
        if p.is_symlink():
            raise ValueError('Symlinks are not delivery files')
    if not p.resolve().is_relative_to(root):
        raise ValueError('Delivery path escaped root')
    return p


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with partial.open('wb') as f:
        f.write(canonical(value) + b'\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(partial, path)
    fsync_dir(path.parent)


def request_sha(request):
    return hashlib.sha256(canonical(request)).hexdigest()


def validate_request(request):
    if request['schema'] != SCHEMA or request['stage'] not in ('epoch-001', 'epoch-002', 'frozen', 'evaluated'):
        raise ValueError('Unknown delivery request')
    run = request['identity']['run_id']
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,160}', run):
        raise ValueError('Invalid run ID')
    if request['bundle'] != f'checkpoints/{run}/{request["stage"]}':
        raise ValueError('Bundle does not match run/stage')
    safe_relative(request['bundle'])
    if not re.fullmatch(r'[0-9a-f]{32}', request['nonce']):
        raise ValueError('Invalid request nonce')
    files = request['files']
    if not files or len({f['path'] for f in files}) != len(files):
        raise ValueError('Empty or duplicate manifest')
    for f in files:
        safe_relative(f['path'])
        if type(f['bytes']) is not int or f['bytes'] < 0 or not re.fullmatch(r'[0-9a-f]{64}', f['sha256']):
            raise ValueError('Invalid file size/hash')
    names = {f['path'] for f in files}
    if request['stage'] in ('frozen', 'evaluated'):
        required = ({'frozen.json', 'calibration-reference.json', 'calibration-reloaded.json'}
                    if request['stage'] == 'frozen' else {'validation.json', 'frozen.json'})
        if request.get('checkpoint_bundle') != f'checkpoints/{run}/epoch-002':
            raise ValueError('Freeze must refer to this run second epoch')
        if not re.fullmatch(r'[0-9a-f]{64}', request.get('epoch_request_sha256', '')):
            raise ValueError('Missing prior delivery identity')
    else:
        required = {'model.safetensors', 'encoder/config.json', 'head_control.json',
                    'manifest.json', 'epoch.json', 'actual_inputs.jsonl', 'updates.jsonl',
                    'assets/source-lock.json', 'assets/laya_model_lock.json',
                    'assets/study-contract.json',
                    'assets/uv.lock', 'assets/cuda-uv.lock',
                    'assets/tokenizer/tokenizer.json', 'assets/tokenizer/tokenizer_config.json'}
        if request['stage'] == 'epoch-001':
            required.add('resume.pt')
        if not any(n.startswith('assets/source/') for n in names):
            raise ValueError('Missing runnable source snapshot')
    if not required <= names:
        raise ValueError('Missing required delivery files: ' + str(sorted(required - names)))


def seal_bundle(root, bundle, identity, stage, **extra):
    """Publish after all files are closed. Sealed bundles must never be mutated."""
    folder = within(root, bundle)
    records = []
    for p in sorted(folder.rglob('*')):
        if p.is_symlink():
            raise ValueError('Bundle contains a symlink')
        if p.is_dir():
            continue
        if not stat.S_ISREG(p.stat().st_mode):
            raise ValueError('Only regular files can be archived')
        with p.open('rb') as f:
            os.fsync(f.fileno())
        records.append({'path': p.relative_to(folder).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)})
    request = {'schema': SCHEMA, 'identity': identity, 'stage': stage, 'bundle': bundle,
               'nonce': secrets.token_hex(16), 'files': records, **extra}
    validate_request(request)
    request_path = within(root, f'delivery/requests/{identity["run_id"]}.{stage}.json')
    if request_path.exists():
        old = json.loads(request_path.read_text())
        request['nonce'] = old['nonce']
        if request != old:
            raise ValueError('A sealed bundle changed')
        return old
    atomic_json(request_path, request)
    atomic_json(within(root, 'delivery/pending.json'), request)
    return request


class LocalTransport:
    """Real file transport used for offline integration and interruption tests."""
    def __init__(self, root):
        self.root = Path(root).resolve()

    @contextmanager
    def open_read(self, relative, offset=0):
        p = within(self.root, relative)
        if not stat.S_ISREG(p.stat().st_mode):
            raise ValueError('Source is not a regular file')
        with p.open('rb') as f:
            f.seek(offset)
            yield f

    def prefix_sha(self, relative, count):
        return digest(within(self.root, relative), count)

    def read_json(self, relative):
        with self.open_read(relative) as f:
            return json.load(f)

    def write_json(self, relative, value):
        atomic_json(within(self.root, relative), value)


class SSHTransport:
    """One streaming SSH process per file; resume never trusts a partial length."""
    def __init__(self, host, port, root, identity_file=None, timeout=10800):
        if host.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_.@:-]+', host):
            raise ValueError('Invalid SSH host')
        if not 1 <= port <= 65535 or not str(root).startswith('/'):
            raise ValueError('Require valid port and absolute remote root')
        self.root, self.timeout = str(root), timeout
        self.ssh = ['ssh', '-p', str(port), '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                    '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=2']
        if identity_file:
            self.ssh += ['-i', str(identity_file)]
        self.ssh.append(host)

    def command(self, relative, action, count=0):
        safe_relative(relative)
        # All variable arguments are shell-quoted; the remote program checks containment.
        source = '''import sys,pathlib,hashlib,json,os,stat,shutil
root=pathlib.Path(sys.argv[1]).resolve(); rel=pathlib.PurePosixPath(sys.argv[2])
assert not rel.is_absolute() and all(x not in ('..','.') for x in rel.parts)
p=root
for x in rel.parts:
 p=p/x
 assert not p.is_symlink()
assert p.resolve().is_relative_to(root)
action=sys.argv[3]; count=int(sys.argv[4])
if action=='write':
 data=json.load(sys.stdin);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.partial')
 with tmp.open('w') as f:
  json.dump(data,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p);fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd)
else:
 if not p.exists():sys.exit(44)
 assert stat.S_ISREG(p.stat().st_mode)
 with p.open('rb') as f:
  if action=='read':
   f.seek(count);shutil.copyfileobj(f,sys.stdout.buffer,4*1024*1024)
  else:
   h=hashlib.sha256()
   while count:
    b=f.read(min(count,4*1024*1024));assert b;h.update(b);count-=len(b)
   print(h.hexdigest())
'''
        return self.ssh + [shlex.join(['python3', '-u', '-c', source, self.root, relative, action, str(count)])]

    @contextmanager
    def open_read(self, relative, offset=0):
        process = subprocess.Popen(self.command(relative, 'read', offset), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE)
        deadline = time.monotonic() + self.timeout
        class Stream:
            def read(self, count):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not select.select([process.stdout], [], [], remaining)[0]:
                    raise TimeoutError('SSH file transfer deadline exceeded')
                return os.read(process.stdout.fileno(), count)
        try:
            yield Stream()
            process.stdout.close()
            if process.wait(timeout=30):
                raise OSError('SSH stream exited unsuccessfully')
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()

    def prefix_sha(self, relative, count):
        value = subprocess.check_output(self.command(relative, 'prefix', count), timeout=self.timeout).decode().strip()
        if not re.fullmatch(r'[0-9a-f]{64}', value):
            raise OSError('SSH prefix response is not a SHA')
        return value

    def read_json(self, relative):
        try:
            data = subprocess.check_output(self.command(relative, 'read'), timeout=self.timeout)
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 44:
                raise FileNotFoundError(relative) from exc
            raise
        return json.loads(data)

    def write_json(self, relative, value):
        subprocess.run(self.command(relative, 'write'), input=canonical(value), timeout=self.timeout, check=True)


def copy_verified(transport, remote_relative, destination, expected):
    destination = Path(destination)
    if destination.exists():
        if destination.stat().st_size != expected['bytes'] or digest(destination) != expected['sha256']:
            raise ValueError('Existing immutable destination is corrupt')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + '.partial')
    if partial.is_symlink():
        raise ValueError('Partial destination is a symlink')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected['bytes'] or (offset and digest(partial) != transport.prefix_sha(remote_relative, offset)):
        with partial.open('wb') as f:
            f.flush(); os.fsync(f.fileno())
        offset = 0
    with transport.open_read(remote_relative, offset) as src, partial.open('ab') as out:
        remaining = expected['bytes'] - offset
        while remaining:
            data = src.read(min(CHUNK, remaining))
            if not data:
                raise OSError('Interrupted or truncated checkpoint stream')
            out.write(data)
            remaining -= len(data)
        if src.read(1):
            raise ValueError('Source file grew after sealing')
        out.flush(); os.fsync(out.fileno())
    if partial.stat().st_size != expected['bytes'] or digest(partial) != expected['sha256']:
        raise ValueError('Transferred checkpoint SHA mismatch')
    os.replace(partial, destination)
    fsync_dir(destination.parent)


def validate_receipt(request, receipt):
    if (receipt.get('schema') != SCHEMA or receipt.get('state') != 'verified_local'
            or receipt.get('request_sha256') != request_sha(request)
            or receipt.get('identity') != request['identity'] or receipt.get('stage') != request['stage']
            or receipt.get('files') != request['files']
            or receipt.get('verification', {}).get('strict_load') is not True
            or not receipt.get('verification', {}).get('state_sha256')):
        raise ValueError('Missing, stale or invalid local verification receipt')


def numerically_equal(a, b):
    """Allow only last-bit CPU platform differences in recomputed scalar metrics."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(numerically_equal(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(numerically_equal(x,y) for x,y in zip(a,b))
    if isinstance(a, float) or isinstance(b, float):
        return isinstance(a,(int,float)) and isinstance(b,(int,float)) and math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    return type(a) is type(b) and a == b


def verify_files(root, request):
    validate_request(request)
    for record in request['files']:
        p = within(root, request['bundle'] + '/' + record['path'])
        if not stat.S_ISREG(p.stat().st_mode) or p.stat().st_size != record['bytes'] or digest(p) != record['sha256']:
            raise ValueError('Delivered file no longer matches sealed manifest: ' + record['path'])


def verify_checkpoint(root, request):
    """Executed in an independent CPU process before any success receipt exists."""
    import torch
    from laya_head_control import load_fp32, state_digest, fit_temperature, logit_metrics
    from laya_robust_training import read_resume
    torch.set_num_threads(4)
    verify_files(root, request)
    folder = within(root, request['bundle'])
    checkpoint = folder
    if request['stage'] in ('frozen', 'evaluated'):
        prior_path = within(root, 'delivery/receipts/' + request['epoch_request_sha256'] + '.json')
        prior = json.loads(prior_path.read_text())
        old_request = json.loads(within(root, 'delivery/received-requests/' + request['epoch_request_sha256'] + '.json').read_text())
        validate_receipt(old_request, prior)
        if old_request['bundle'] != request['checkpoint_bundle'] or old_request['identity'] != request['identity']:
            raise ValueError('Frozen delivery refers to another checkpoint')
        verify_files(root, old_request)
        checkpoint = within(root, request['checkpoint_bundle'])
    model, metadata = load_fp32(checkpoint)
    if not all(not v.is_floating_point() or (v.dtype == torch.float32 and bool(torch.isfinite(v).all())) for v in model.state_dict().values()):
        raise FloatingPointError('Delivered weights are not finite FP32')
    sha = state_digest(model.state_dict())
    manifest = json.loads((checkpoint/'manifest.json').read_text())
    if manifest['identity'] != request['identity']:
        raise ValueError('Delivered manifest belongs to another experiment')
    if digest(checkpoint/'assets/study-contract.json') != request['identity']['contract_sha256']:
        raise ValueError('Archived contract differs from training identity')
    if request['stage'] == 'epoch-001':
        recovery = read_resume(checkpoint/'resume.pt', request['identity'], sha)
        groups = recovery['optimizer']['param_groups']
        parameters = [p for n,p in model.named_parameters() if n.startswith('encoder.')]
        parameters += [p for n,p in model.named_parameters() if not n.startswith('encoder.')]
        keys = [key for group in groups for key in group['params']]
        if len(groups) != 2 or len(keys) != len(parameters) or len(set(keys)) != len(keys):
            raise ValueError('Optimizer parameter groups differ from model')
        for key, param in zip(keys,parameters,strict=True):
            item = recovery['optimizer']['state'].get(key)
            if item is not None and (item['exp_avg'].shape != param.shape or item['exp_avg_sq'].shape != param.shape):
                raise ValueError('Optimizer recovery tensor shape differs')
    if request['stage'] == 'frozen':
        frozen = json.loads((folder/'frozen.json').read_text())
        reference = json.loads((folder/'calibration-reference.json').read_text())
        current = json.loads((folder/'calibration-reloaded.json').read_text())
        if (frozen['identity'] != request['identity'] or frozen['checkpoint_state_sha256'] != sha
                or reference['rows'] != current['rows'] or reference['logits'] != current['logits']
                or len(current['rows']) != 400 or len({r['sample_id'] for r in current['rows']}) != 400
                or [r['sample_id'] for r in current['rows']] != manifest['calibration_ids']
                or not frozen['independent_reload_logits_bitwise_equal']):
            raise ValueError('Missing exact independent calibration reload')
        fit = fit_temperature([z[1]-z[0] for z in current['logits']], [r['true_label'] for r in current['rows']])
        if not numerically_equal(fit, frozen['temperature_fit']):
            raise ValueError('Frozen temperature does not match calibration predictions')
    if request['stage'] == 'evaluated':
        prediction = json.loads((folder/'validation.json').read_text())
        frozen = json.loads((folder/'frozen.json').read_text())
        rows = prediction['rows']
        if (prediction['identity'] != request['identity'] or frozen['identity'] != request['identity']
                or len(rows) != 2500 or len({r['sample_id'] for r in rows}) != 2500
                or frozen['checkpoint_state_sha256'] != sha
                or prediction['temperature'] != frozen['temperature_fit']['temperature']):
            raise ValueError('Invalid delivered development identity/count/temperature')
        for key, temp in [('raw_metrics', 1.), ('calibrated_metrics', prediction['temperature'])]:
            actual = logit_metrics([r['true_label'] for r in rows], prediction['logits'], [r['truncated'] for r in rows], temp)
            if not numerically_equal(actual, prediction[key]):
                raise ValueError('Delivered development metrics do not match logits')
    return {'strict_load': True, 'device': 'cpu', 'independent_process': True,
            'state_sha256': sha, 'parameter_count': metadata['parameter_count']}


def receive(transport, local_root, request, attempts=3, minimum_free_bytes=0):
    validate_request(request)
    if not 1 <= attempts <= 3:
        raise ValueError('At most three delivery attempts are allowed')
    sha = request_sha(request)
    receipt_path = within(local_root, f'delivery/receipts/{sha}.json')
    request_path = within(local_root, f'delivery/received-requests/{sha}.json')
    last_error = None
    started = time.monotonic()
    for attempt in range(attempts):
        try:
            Path(local_root).mkdir(parents=True, exist_ok=True)
            free = shutil.disk_usage(local_root).free
            if free < minimum_free_bytes:
                raise OSError('Insufficient local free space for approved delivery reserve')
            for record in request['files']:
                relative = request['bundle'] + '/' + record['path']
                copy_verified(transport, relative, within(local_root, relative), record)
            atomic_json(request_path, request)
            # Success comes from loading real received bytes in a new process.
            checked = subprocess.check_output([sys.executable, str(Path(__file__).resolve()), '--verify',
                '--local-root', str(Path(local_root).resolve()), '--request-file', str(request_path)], timeout=1800)
            verification = json.loads(checked)
            receipt = {'schema': SCHEMA, 'state': 'verified_local', 'identity': request['identity'],
                       'stage': request['stage'], 'request_sha256': sha, 'files': request['files'],
                       'verification': verification, 'minimum_free_bytes': minimum_free_bytes,
                       'transport': 'ssh' if isinstance(transport, SSHTransport) else 'local-test',
                       'local_free_bytes_before_copy': free, 'receive_seconds': time.monotonic()-started,
                       'bundle_bytes': sum(f['bytes'] for f in request['files'])}
            validate_receipt(request, receipt)
            if receipt_path.exists():
                previous = json.loads(receipt_path.read_text())
                validate_receipt(request, previous)
                if previous['verification'] != verification:
                    raise ValueError('Independent reload differs from existing receipt')
                receipt = previous
            else:
                atomic_json(receipt_path, receipt)
            # A lost ACK can be resent; all local bytes and the receipt were fsynced first.
            transport.write_json(f'delivery/acks/{sha}.json', receipt)
            return receipt
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            last_error = exc
    failure = {'state': 'blocked_delivery', 'request_sha256': sha, 'attempts': attempts,
               'error': str(last_error), 'identity': request['identity'], 'stage': request['stage']}
    atomic_json(within(local_root, f'delivery/failures/{sha}.json'), failure)
    # Best effort notification; failure to notify still cannot create a success ACK.
    try:
        transport.write_json(f'delivery/failures/{sha}.json', failure)
    except (OSError, subprocess.SubprocessError):
        pass
    raise DeliveryBlocked(str(last_error)) from last_error


def wait_for_receipt(root, request, timeout=10800, poll=1.):
    validate_request(request)
    sha = request_sha(request)
    ack = within(root, f'delivery/acks/{sha}.json')
    failure = within(root, f'delivery/failures/{sha}.json')
    deadline = time.monotonic() + timeout
    while True:
        if ack.exists():
            try:
                value = json.loads(ack.read_text())
                validate_receipt(request, value)
                return value
            except (ValueError, KeyError) as exc:
                raise DeliveryBlocked('Invalid local ACK; training remains blocked') from exc
        if failure.exists() or time.monotonic() >= deadline:
            atomic_json(within(root, 'delivery/status.json'), {'state': 'blocked_delivery',
                'request_sha256': sha, 'identity': request['identity'], 'stage': request['stage']})
            raise DeliveryBlocked('Local delivery failed or timed out; no further training allowed')
        time.sleep(min(poll, max(0, deadline - time.monotonic())))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--local-root', type=Path, required=True)
    p.add_argument('--request-file', type=Path)
    p.add_argument('--source-root', type=Path, help='Local source for offline integration only')
    p.add_argument('--host')
    p.add_argument('--port', type=int, default=22)
    p.add_argument('--remote-root')
    p.add_argument('--identity-file', type=Path)
    p.add_argument('--watch-seconds', type=float, default=0)
    p.add_argument('--minimum-free-gib', type=float, default=100)
    args = p.parse_args()
    if args.verify:
        if not args.request_file:
            p.error('--verify requires --request-file')
        print(json.dumps(verify_checkpoint(args.local_root, json.loads(args.request_file.read_text()))))
        return
    if args.source_root:
        if args.host:
            p.error('Select one transport')
        transport = LocalTransport(args.source_root)
    else:
        if not args.host or not args.remote_root:
            p.error('SSH receiver requires --host and --remote-root')
        transport = SSHTransport(args.host, args.port, args.remote_root, args.identity_file)
    deadline = time.monotonic() + args.watch_seconds
    delivered = set()
    while True:
        try:
            request = (json.loads(args.request_file.read_text()) if args.request_file
                       else transport.read_json('delivery/pending.json'))
            sha = request_sha(request)
            if sha not in delivered:
                receive(transport, args.local_root, request, minimum_free_bytes=int(args.minimum_free_gib*1024**3))
                delivered.add(sha)
                print(f'{request["identity"]["run_id"]} {request["stage"]}: verified locally', flush=True)
        except FileNotFoundError:
            if args.watch_seconds <= 0:
                raise
        if time.monotonic() >= deadline:
            break
        time.sleep(1)


if __name__ == '__main__':
    main()

"""Copy committed Modal Volume files locally, independently load, then acknowledge."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from laya_checkpoint_delivery import (SCHEMA, DeliveryBlocked, atomic_json, digest,
    fsync_dir, request_sha, validate_receipt, validate_request, within)
from laya_modal_runtime import MINIMUM_FREE, VOLUME


def volume_path(relative):
    from laya_checkpoint_delivery import safe_relative
    return '/End-IMDB/' + str(safe_relative(relative))


def read_json(volume, relative):
    return json.loads(b''.join(volume.read_file(volume_path(relative))))


def copy_from_volume(volume, relative, destination, record):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != record['bytes'] or digest(destination) != record['sha256']:
            raise ValueError('Immutable local file differs: ' + relative)
        return
    partial = destination.with_name(destination.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > record['bytes']:
        offset = 0
    prefix = digest(partial, offset) if offset else hashlib.sha256(b'').hexdigest()
    # Volume API starts at byte zero. Verify the remote prefix before appending;
    # this still avoids rewriting an already verified partial on local disk.
    iterator = iter(volume.read_file(volume_path(relative)))
    h = hashlib.sha256()
    skipped = 0
    tail = b''
    while skipped < offset:
        block = next(iterator, b'')
        if not block:
            raise ValueError('Remote file shorter than existing partial')
        take = min(len(block), offset-skipped)
        h.update(block[:take]); skipped += take
        tail = block[take:]
    if h.hexdigest() != prefix:
        partial.unlink()
        return copy_from_volume(volume, relative, destination, record)
    count = offset
    with partial.open('ab' if offset else 'wb') as out:
        def write(block):
            nonlocal count
            count += len(block)
            if count > record['bytes']:
                raise ValueError('Remote file exceeds manifest')
            out.write(block); h.update(block)
        write(tail)
        for block in iterator:
            write(block)
        out.flush(); os.fsync(out.fileno())
    if count != record['bytes'] or h.hexdigest() != record['sha256']:
        raise ValueError('Modal download size/SHA mismatch: ' + relative)
    os.replace(partial, destination)
    fsync_dir(destination.parent)


def receive(volume, acknowledgments, root, request, verifier_python, minimum_free_bytes=MINIMUM_FREE):
    validate_request(request)
    root = Path(root)
    sha = request_sha(request)
    received = within(root, f'delivery/received-requests/{sha}.json')
    receipt_path = within(root, f'delivery/receipts/{sha}.json')
    started = time.monotonic()
    last_error = None
    for attempt in range(3):
        try:
            free = shutil.disk_usage(root).free
            if free < minimum_free_bytes:
                raise OSError('Require 100 GiB local reserve before delivery')
            for record in request['files']:
                relative = request['bundle'] + '/' + record['path']
                copy_from_volume(volume, relative, within(root, relative), record)
            atomic_json(received, request)
            checked = subprocess.check_output([str(verifier_python),
                str(Path(__file__).with_name('laya_checkpoint_delivery.py')), '--verify',
                '--local-root', str(root), '--request-file', str(received)], timeout=1800)
            verification = json.loads(checked)
            receipt = {'schema': SCHEMA, 'state': 'verified_local', 'identity': request['identity'],
                'stage': request['stage'], 'request_sha256': sha, 'files': request['files'],
                'verification': verification, 'minimum_free_bytes': minimum_free_bytes,
                'transport': 'modal-volume', 'modal_volume': VOLUME,
                'local_free_bytes_before_copy': free, 'receive_seconds': time.monotonic()-started,
                'bundle_bytes': sum(f['bytes'] for f in request['files'])}
            validate_receipt(request, receipt)
            if receipt_path.exists():
                previous = json.loads(receipt_path.read_text())
                validate_receipt(request, previous)
                if previous['verification'] != verification:
                    raise ValueError('Independent local reload changed')
                receipt = previous
            else:
                atomic_json(receipt_path, receipt)
            # This Dict is just an ACK channel. All weight bytes and the durable
            # local receipt already exist before anything is sent to the cloud.
            acknowledgments[sha] = receipt
            return receipt
        except Exception as exc:
            last_error = exc
    failure = {'state': 'blocked_delivery', 'request_sha256': sha,
               'identity': request['identity'], 'stage': request['stage'],
               'attempts': 3, 'error': str(last_error)}
    atomic_json(within(root, f'delivery/failures/{sha}.json'), failure)
    acknowledgments[sha] = failure
    raise DeliveryBlocked(str(last_error)) from last_error

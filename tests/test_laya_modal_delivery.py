"""Local byte/strict-load tests of the Modal adapter; not cloud admission."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from laya_checkpoint_delivery import digest, request_sha, DeliveryBlocked
from laya_modal_delivery import copy_from_volume, receive


class FileVolume:
    def __init__(self, root):
        self.root = Path(root)
    def read_file(self, relative):
        assert relative.startswith('/End-IMDB/')
        with (self.root/relative[len('/End-IMDB/'):]).open('rb') as source:
            while True:
                block = source.read(137)
                if not block:
                    break
                yield block


class ModalDeliveryTests(unittest.TestCase):
    def test_unfinished_modal_poll_uses_builtin_timeout(self):
        from laya_modal_runtime import poll_call
        class Pending:
            def get(self, timeout):
                raise TimeoutError('No output yet')
        class Completed:
            def get(self, timeout):
                return {'state':'frozen_delivered'}
        self.assertEqual(poll_call(Pending()), (False, None))
        self.assertEqual(poll_call(Completed()), (True, {'state':'frozen_delivered'}))

    def test_byte_stream_prefix_verification_and_restart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); remote = root/'remote'; remote.mkdir()
            data = bytes(range(256))*41
            (remote/'weights').write_bytes(data)
            record = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
            volume = FileVolume(remote); target = root/'weights'
            partial = root/'weights.partial'
            partial.write_bytes(data[:537])
            copy_from_volume(volume, 'weights', target, record)
            self.assertEqual(target.read_bytes(), data)
            target.unlink(); partial.write_bytes(b'wrong previous content')
            copy_from_volume(volume, 'weights', target, record)
            self.assertEqual(target.read_bytes(), data)
            with self.assertRaises(ValueError):
                copy_from_volume(volume, '../outside', target, record | {'sha256':'0'*64})

    def test_real_checkpoint_independent_load_precedes_ack(self):
        from test_laya_checkpoint_delivery import DeliveryTests
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); remote = root/'remote'; local = root/'local'; local.mkdir()
            req, _, _ = DeliveryTests().bundle(remote)
            acks = {}
            receipt = receive(FileVolume(remote), acks, local, req, sys.executable, 0)
            self.assertEqual(receipt['transport'], 'modal-volume')
            self.assertTrue(receipt['verification']['strict_load'])
            self.assertTrue(receipt['verification']['independent_process'])
            self.assertEqual(acks[request_sha(req)], receipt)
            self.assertTrue((local/'delivery/receipts'/f'{request_sha(req)}.json').exists())
            self.assertEqual(receive(FileVolume(remote), acks, local, req, sys.executable, 0), receipt)
            from laya_modal_runtime import production_receipt
            with patch('laya_modal_runtime.ROOT', local):
                from laya_checkpoint_delivery import atomic_json
                atomic_json(local/'delivery/acks'/f'{request_sha(req)}.json', receipt)
                with self.assertRaisesRegex(ValueError, '100 GiB'):
                    production_receipt(req, 0)

    def test_missing_weight_or_disk_failure_never_sends_success(self):
        from test_laya_checkpoint_delivery import DeliveryTests
        for fault in ('missing', 'disk'):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); remote = root/'remote'; local = root/'local'; local.mkdir()
                req, folder, _ = DeliveryTests().bundle(remote)
                if fault == 'missing':
                    (folder/'model.safetensors').unlink()
                acks = {}
                with self.assertRaises(DeliveryBlocked):
                    receive(FileVolume(remote), acks, local, req, sys.executable,
                            2**70 if fault == 'disk' else 0)
                self.assertEqual(acks[request_sha(req)]['state'], 'blocked_delivery')
                self.assertFalse((local/'delivery/receipts'/f'{request_sha(req)}.json').exists())


if __name__ == '__main__':
    unittest.main()

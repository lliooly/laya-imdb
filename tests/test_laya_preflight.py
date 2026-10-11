"""CPU-only checks prevent a missing CUDA device from passing training preflight."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_laya import require_device


def runtime(cuda_available=False, mps_available=False, count=1):
    def device(name):
        kind, _, index = name.partition(':')
        return SimpleNamespace(type=kind, index=int(index) if index else None)
    return SimpleNamespace(device=device,
        cuda=SimpleNamespace(is_available=lambda: cuda_available,
                             device_count=lambda: count, current_device=lambda: 0),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps_available)))


class LayaPreflightTests(unittest.TestCase):
    def test_cuda_unavailable_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'CUDA unavailable'):
            require_device(runtime(), 'cuda')

    def test_cuda_out_of_range_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'index 1'):
            require_device(runtime(cuda_available=True), 'cuda:1')

    def test_available_cuda_is_accepted(self):
        require_device(runtime(cuda_available=True), 'cuda')
        require_device(runtime(cuda_available=True), 'cuda:0')

    def test_mps_unavailable_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'MPS unavailable'):
            require_device(runtime(), 'mps')

    def test_cpu_and_available_mps_are_accepted(self):
        require_device(runtime(), 'cpu')
        require_device(runtime(mps_available=True), 'mps')


if __name__ == '__main__':
    unittest.main()

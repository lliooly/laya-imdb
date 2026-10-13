"""Guard strict admission and reject changing workspace after CUDA initialization."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from laya_runtime import prepare_runtime, apply_runtime


class RuntimeTests(unittest.TestCase):
    def test_legacy_profile_is_untouched(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(prepare_runtime({'device': 'cpu'}))
            self.assertNotIn('CUBLAS_WORKSPACE_CONFIG', os.environ)
        self.assertIsNone(apply_runtime(None, None))

    def test_invalid_protocols_are_rejected(self):
        for spec in ({'runtime_profile': 'typo'},
                     {'runtime_profile': 'strict', 'device': 'mps', 'training': {'amp': False}},
                     {'runtime_profile': 'strict', 'device': 'cuda', 'training': {'amp': True}}):
            with self.assertRaises(ValueError):
                prepare_runtime(spec)

    def test_initialized_cuda_rejects_late_workspace_change(self):
        fake = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: True))
        spec = {'runtime_profile': 'strict', 'device': 'cuda', 'training': {'amp': False}}
        with patch.dict(sys.modules, {'torch': fake}), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'after CUDA initialization'):
                prepare_runtime(spec)

    def test_strict_flags_do_not_consume_rng(self):
        import torch
        state = torch.random.get_rng_state().clone()
        enabled = torch.are_deterministic_algorithms_enabled()
        warn = torch.is_deterministic_algorithms_warn_only_enabled()
        deterministic, benchmark = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        try:
            with patch.dict(os.environ, {'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}):
                result = apply_runtime(torch, 'strict')
                self.assertTrue(result['deterministic_algorithms'])
                self.assertFalse(result['warn_only'])
                self.assertFalse(result['cudnn_benchmark'])
                self.assertTrue(result['cudnn_deterministic'])
                self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn)
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = deterministic, benchmark


if __name__ == '__main__':
    unittest.main()

"""Regression for integer sampling beyond the exact range of float32 indices."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_cuda_training import sample_indices


class SamplingTests(unittest.TestCase):
    def test_large_tensor_endpoint_stays_in_bounds(self):
        import torch
        numel = 50331648
        indices = sample_indices(torch, numel, 'cpu')
        self.assertEqual(indices[0].item(), 0)
        self.assertEqual(indices[-1].item(), numel - 1)
        self.assertTrue(bool(((indices >= 0) & (indices < numel)).all()))
        self.assertEqual(len(indices.unique()), 16)

    def test_small_tensor_has_no_duplicate_or_missing_indices(self):
        import torch
        for numel in (1, 2, 7, 16):
            self.assertEqual(sample_indices(torch, numel, 'cpu').tolist(), list(range(numel)))


if __name__ == '__main__':
    unittest.main()

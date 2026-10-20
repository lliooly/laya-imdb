"""Guard explicit initialization seeds and immutable checkpoint manifests."""
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import torch
import paper_gpu_supplement as study
from deberta_training import groups_for


class SupplementTests(unittest.TestCase):
    def test_seed_groups_are_disjoint_and_orders_change(self):
        groups = [{study.SEEDS[k][i] for k in study.SEEDS} for i in range(3)]
        for i in range(3):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        self.assertNotEqual(groups_for(22100, 0, 42), groups_for(22100, 0, 142))

    def test_loader_seed_is_explicit_reproducible_and_not_reset_to_42(self):
        class Loader:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                model = torch.nn.Linear(5, 2)
                model.config = SimpleNamespace(position_biased_input=False)
                return model, dict(missing_keys=['classifier.weight'], unexpected_keys=[],
                                   mismatched_keys=[], error_msgs=[])
        with patch.dict(sys.modules, {'transformers': SimpleNamespace(AutoModelForSequenceClassification=Loader)}):
            a = study.initial_deberta(42)
            b = study.initial_deberta(142)
            again = study.initial_deberta(42)
        self.assertFalse(torch.equal(a.weight, b.weight))
        self.assertTrue(torch.equal(a.weight, again.weight))

    def test_bundle_tampering_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder/'weights').write_bytes(b'original')
            study.seal(folder, kind='deberta')
            study.check_bundle(folder)
            (folder/'weights').write_bytes(b'changed!')
            with self.assertRaises(ValueError):
                study.check_bundle(folder)
            with self.assertRaises(ValueError):
                study.paths('deberta', 43)


if __name__ == '__main__':
    unittest.main()

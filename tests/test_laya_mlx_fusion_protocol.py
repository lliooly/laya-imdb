"""Reject accidental precision/protocol drift before any GPU execution."""
import copy
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT
from validate_laya_mlx_fusion import validate_config
from validate_laya_fusion_inference import compare_fused

class MLXProtocolTests(unittest.TestCase):
    def test_protocol_cannot_silently_change_device_batch_or_tolerance(self):
        spec = json.loads((ROOT/'configs/laya-fusion-inference-mlx-fp16-20261014-v1.json').read_text())
        validate_config(spec)
        for key, value in [('device', 'mps'), ('batch_size', 1),
                           ('tolerance', {'logits_absolute': .1, 'probabilities_absolute': .02})]:
            changed = copy.deepcopy(spec)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_config(changed)
        # Small FP16 differences must still fail when they flip the decision.
        check = compare_fused(['near-boundary'], [[.499, .501]], [[.501, .499]], spec['tolerance'])
        self.assertFalse(check['passed'])
        self.assertEqual(check['label_mismatches'], ['near-boundary'])
        self.assertEqual(check['out_of_tolerance_ids'], [])

if __name__ == '__main__':
    unittest.main()

"""Guard the explicit precision mode before loading any weights or touching Metal."""
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT
from laya_mlx_precise_inference import PreciseMLXFusionPredictor
from validate_laya_mlx_precise import validate_config

class PreciseModeTests(unittest.TestCase):
    def test_missing_or_enabled_tf32_fails_before_loading(self):
        for value in (None,'1','false',''):
            with patch.dict(os.environ,{},clear=True):
                if value is not None:os.environ['MLX_ENABLE_TF32']=value
                with self.assertRaisesRegex(RuntimeError,'fresh process'):
                    PreciseMLXFusionPredictor(ROOT,'missing',{}, {})
    def test_unchanged_acceptance_gate_and_explicit_backend(self):
        spec=json.loads((ROOT/'configs/laya-fusion-inference-mlx-fp32-precise-20261014-v1.json').read_text())
        validate_config(spec)
        for change in ({'device':'mlx'},{'tolerance':{'logits_absolute':1.,'probabilities_absolute':.1}}):
            with self.assertRaises(ValueError):validate_config(dict(spec,**change))

if __name__=='__main__':unittest.main()

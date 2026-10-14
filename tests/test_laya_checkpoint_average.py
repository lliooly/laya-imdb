import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import ROOT, file_digest
from average_laya_checkpoints import average_checkpoints
from comparison_metrics import sigmoid
from run_laya_checkpoint_average import (validate_spec, fit_temperature_only,
    checked_records, export_head_tail, save_calibrated_checkpoint)


class CheckpointAverageStudyTests(unittest.TestCase):
    def test_protocol_rejects_weight_search_bias_and_changed_source(self):
        spec = json.loads((ROOT/'configs/laya-checkpoint-average-prefix-s42-strict-20261014-v1.json').read_text())
        validate_spec(spec)
        for key, value in (('weights', [.6, .4]), ('source_run', 'other'),
                           ('inference_rule', 'head'), ('temperature_bounds', [.05, 20.]),
                           ('manifest_sha256', 'bad'), ('run_id', '../old')):
            with self.assertRaises(ValueError):
                validate_spec(dict(spec, **{key: value}))
        with self.assertRaises(ValueError):
            validate_spec(dict(spec, bias=True))

    def test_temperature_only_is_stationary_and_preserves_class_decisions(self):
        margin = np.array([-4.]*100+[4.]*100+[0.]*20)
        labels = np.array([0]*75+[1]*25+[1]*75+[0]*25+[0, 1]*10)
        fit = fit_temperature_only(margin, labels)
        self.assertEqual(fit['bias'], 0.)
        self.assertLess(fit['after']['nll'], fit['before']['nll'])
        self.assertAlmostEqual(float(np.mean((sigmoid(margin/fit['temperature'])-labels)*margin)), 0., places=10)
        np.testing.assert_array_equal(margin > 0, margin/fit['temperature'] > 0)
        self.assertEqual(fit['before']['accuracy'], fit['after']['accuracy'])
        for m, y in (([np.nan], [0]), ([1], [2]), ([], [])):
            with self.assertRaises(ValueError):
                fit_temperature_only(m, y)

    def test_reference_rejects_misordered_ids_labels_truncation_and_nonfinite(self):
        rows = [{'id': 'train/a', 'label': 1}]
        record = {'sample_id': 'train/a', 'true_label': 1, 'input_sha256': 'a'*64,
                  'truncated': False, 'review_tokens': 10, 'review_token_budget': 466, 'logits': [0., 1.]}
        doc = {'split': 'calibration', 'rule': 'head_tail', 'test_accessed': False, 'rows': [record]}
        self.assertEqual(checked_records(doc, rows, 'calibration')[1].tolist(), [1.])
        for key, value in (('sample_id', 'train/b'), ('true_label', 0), ('truncated', True),
                           ('logits', [0., np.inf]), ('input_sha256', 'invalid')):
            bad = copy.deepcopy(doc);bad['rows'][0][key] = value
            with self.assertRaises(ValueError):
                checked_records(bad, rows, 'calibration')

    def test_real_average_calibrated_save_reload_and_forward(self):
        import torch
        import laya.train as upstream
        from laya.common import DecisionModel
        from safetensors.torch import load_file
        from transformers import AutoTokenizer, BertConfig, BertModel
        torch.set_num_threads(1)
        torch.manual_seed(8)
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            root = Path(directory)
            tok = AutoTokenizer.from_pretrained(str(ROOT/'.cache/laya/english/tokenizer'), local_files_only=True)
            model = DecisionModel(BertModel(BertConfig(vocab_size=len(tok), hidden_size=64,
                num_hidden_layers=1, num_attention_heads=1, intermediate_size=128,
                max_position_embeddings=512)), head_layers=1, n_act=2).eval()
            cfg = {'encoder': 'synthetic-only', 'head_layers': 1, 'max_len': 512,
                   'head_max_len': 192, 'temperature': [1., 1., 1.], 'act_costs': {'escalate': .5}}
            one, two, raw, calibrated = [root/name for name in ('one', 'two', 'raw', 'calibrated')]
            upstream.save_checkpoint(model, tok, cfg, str(one))
            for p in model.scorer.parameters():
                p.add_(.001)
            upstream.save_checkpoint(model, tok, cfg, str(two))
            sources = [file_digest(p/'model.safetensors') for p in (one, two)]
            average_checkpoints(one, two, raw, *sources)
            save_calibrated_checkpoint(torch, raw, calibrated, {'temperature': 2.75}, 'f'*64)
            a, b = load_file(raw/'model.safetensors'), load_file(calibrated/'model.safetensors')
            for name in a:
                if name != 'temperature':
                    self.assertTrue(torch.equal(a[name], b[name]), name)
            self.assertEqual(b['temperature'][0].item(), 2.75)
            reloaded, tokenizer, saved_cfg = upstream.load_checkpoint(str(calibrated))
            reloaded.float().eval()
            self.assertEqual(saved_cfg['temperature'][0], 2.75)
            rows = [{'id': 'train/short', 'label': 1, 'text': 'A wonderful movie.'},
                    {'id': 'train/long', 'label': 0, 'text': 'The story is slow and dull. '*85}]
            records, margins = export_head_tail(torch, upstream, reloaded, tokenizer, rows, 'cpu', 2,
                                                root, 'calibration', 2.75)
            self.assertEqual([r['truncated'] for r in records], [False, True])
            np.testing.assert_allclose([r['probabilities'][1] for r in records], sigmoid(margins/2.75))
            self.assertEqual(sources, [file_digest(p/'model.safetensors') for p in (one, two)])
            with self.assertRaises(FileExistsError):
                save_calibrated_checkpoint(torch, raw, calibrated, {'temperature': 2.75}, 'f'*64)


if __name__ == '__main__':
    unittest.main()

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT, file_digest
from export_laya_posthoc import prepare_rules
from laya_fusion_inference import FrozenFusionPredictor, MODEL_NAMES
from laya_probability_fusion import apply_gated_mixture
from validate_laya_fusion_inference import compare_fused, compare_trace


class ActualInferenceTests(unittest.TestCase):
    def test_real_model_reload_label_independence_window_reuse_and_fusion(self):
        import torch
        import laya.train as upstream
        from laya.common import DecisionModel
        from transformers import AutoTokenizer, BertConfig, BertModel
        torch.set_num_threads(1)
        torch.manual_seed(28)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer = AutoTokenizer.from_pretrained(str(ROOT/'.cache/laya/english/tokenizer'), local_files_only=True)
            model = DecisionModel(BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=64,
                num_hidden_layers=1, num_attention_heads=1, intermediate_size=128,
                max_position_embeddings=512)), head_layers=1, n_act=2).eval()
            cfg = {'encoder': 'synthetic-only', 'head_layers': 1, 'max_len': 512,
                   'head_max_len': 192, 'temperature': [1.5, 1., 1.], 'act_costs': {'escalate': .5}}
            assets = {}
            for i, name in enumerate(MODEL_NAMES):
                folder = root/'checkpoints'/name
                if i:
                    with torch.no_grad():
                        for p in model.scorer.parameters():
                            p.add_(.01)
                upstream.save_checkpoint(model, tokenizer, cfg, str(folder))
                paths = [p for p in folder.rglob('*') if p.is_file() and p.suffix in ('.json', '.safetensors')]
                assets[name] = {'directory': name, 'sha256': {str(p.relative_to(folder)): file_digest(p) for p in paths}}
            fit = {'primary_models': list(MODEL_NAMES), 'mixture': {'coefficient': .702},
                   'validation_used_for_fitting': False}
            fit = json.loads(json.dumps(fit))
            engine = FrozenFusionPredictor(root, 'checkpoints', assets, fit, 'cpu', batch_size=2)
            for name in MODEL_NAMES:
                engine.load(name)
            texts = ['An excellent and funny movie.', 'The film is dull and slow. '*100]
            result = engine.predict(texts)
            self.assertEqual([s['truncated'] for s in result['stats']], [False, True])
            self.assertEqual(result['forward_rows'], 6)
            for name in MODEL_NAMES:
                h, t = result['models'][name]['head'], result['models'][name]['head_tail']
                for key in ('input_sha256', 'logits', 'probabilities'):
                    self.assertEqual(h[0][key], t[0][key])
                self.assertNotEqual(h[1]['input_sha256'], t[1]['input_sha256'])
            expected = apply_gated_mixture(*[np.asarray([r['probabilities'] for r in result['models'][n][v]])
                for n in MODEL_NAMES for v in ('head', 'head_tail')], np.asarray([False, True]), .702)
            np.testing.assert_array_equal(result['primary'], expected)
            # Alter all target labels while holding text and options fixed: actual
            # model logits must remain identical, not merely mocked input metadata.
            variants = []
            for label in (0, 1):
                rows = [{'id': 'a', 'label': label, 'text': texts[0]}]
                head, _, _, _, _ = prepare_rules(engine.tokenizers[MODEL_NAMES[0]], rows)
                item = upstream.encode_item(engine.tokenizers[MODEL_NAMES[0]], head[0], 512, 192, [0, 1], parallel=False)
                batch = upstream.collate_items([[item]], tokenizer.pad_token_id)
                with torch.inference_mode():
                    variants.append(upstream._forward(engine.models[MODEL_NAMES[0]], batch, torch.device('cpu'), False, False))
            self.assertTrue(torch.equal(variants[0], variants[1]))
            baseline = engine.predict(texts, mode='baseline_head_tail')
            self.assertEqual(baseline['forward_rows'], 2)
            np.testing.assert_allclose(baseline['primary'],
                [r['probabilities'] for r in result['models'][MODEL_NAMES[0]]['head_tail']], atol=1e-6)
            engine.verify_assets()
            changed = copy.deepcopy(assets)
            changed[MODEL_NAMES[0]]['sha256']['model.safetensors'] = 'a'*64
            with self.assertRaises(ValueError):
                FrozenFusionPredictor(root, 'checkpoints', changed, fit, 'cpu')

    def test_parity_failure_preserves_predeclared_tolerances_and_tie_rule(self):
        tolerance = {'logits_absolute': .001, 'probabilities_absolute': .0001}
        row = {'sample_id': 'a', 'input_sha256': 'a'*64, 'review_tokens': 5,
               'review_token_budget': 466, 'truncated': False,
               'logits': [1., 1.], 'probabilities': [.5, .5]}
        self.assertTrue(compare_trace([row], [row], tolerance)['passed'])
        different = dict(row, logits=[1., 1.00001], probabilities=[.499999, .500001])
        self.assertFalse(compare_trace([different], [row], tolerance)['passed'])
        self.assertFalse(compare_fused(['a'], [[.499999, .500001]], [[.5, .5]], tolerance)['passed'])
        different = dict(row, logits=[1.01, 1.01])
        self.assertFalse(compare_trace([different], [row], tolerance)['passed'])
        with self.assertRaises(ValueError):
            compare_trace([dict(row, input_sha256='b'*64)], [row], tolerance)


if __name__ == '__main__':
    unittest.main()

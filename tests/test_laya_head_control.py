import copy
import gzip
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from transformers import BertConfig, BertModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import ROOT
from laya_head_control import HeadControl, state_digest, save_fp32, load_fp32, fit_temperature, audit_training, logit_metrics


def encoder(vocab=40, positions=32):
    return BertModel(BertConfig(vocab_size=vocab, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=1, intermediate_size=32, max_position_embeddings=positions,
        hidden_dropout_prob=0., attention_probs_dropout_prob=0.))


class HeadControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def batch(self):
        ids = torch.tensor([[2, 4, 8, 9, 8, 6], [2, 8, 10, 11, 8, 6]])
        pos = torch.tensor([[2, 4], [1, 4]])
        return ids, torch.ones_like(ids), pos, torch.ones_like(pos, dtype=torch.bool), torch.zeros(2, dtype=torch.long)

    def test_original_logits_and_gradients_match_upstream(self):
        from laya.common import DecisionModel
        torch.manual_seed(17)
        enc = encoder()
        actual = HeadControl(copy.deepcopy(enc), 'original', 42).train()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(1000003+42)
            expected = DecisionModel(copy.deepcopy(enc), head_layers=2, n_act=2, dropout=.1).train()
        torch.manual_seed(3)
        a = actual(*self.batch())[0]
        torch.manual_seed(3)
        b = expected(*self.batch())[0]
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        a.square().sum().backward();b.square().sum().backward()
        ref = dict(expected.named_parameters())
        for n, p in actual.named_parameters():
            if p.grad is not None:
                torch.testing.assert_close(p.grad, ref[n].grad, rtol=0, atol=0)

    def test_initialization_restores_rng_and_preserves_encoder(self):
        enc = encoder();before = state_digest(enc.state_dict());rng = torch.get_rng_state().clone()
        a = HeadControl(copy.deepcopy(enc), 'original', 42)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        b = HeadControl(copy.deepcopy(enc), 'original', 42)
        self.assertEqual(state_digest(a.state_dict()), state_digest(b.state_dict()))
        self.assertEqual(before, state_digest(a.encoder.state_dict()))
        c = HeadControl(copy.deepcopy(enc), 'original', 43)
        self.assertNotEqual(state_digest(a.scorer.state_dict()), state_digest(c.scorer.state_dict()))

    def test_linear_uses_two_raw_markers_and_has_gradients(self):
        model = HeadControl(encoder(), 'linear', 42).eval()
        ids, mask, pos, valid, qtype = self.batch()
        h = model.encoder(input_ids=ids, attention_mask=mask).last_hidden_state
        expected = model.scorer(h[torch.arange(2)[:, None], pos]).squeeze(-1)
        actual = model(ids, mask, pos, valid, qtype)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.nn.functional.cross_entropy(actual, torch.tensor([0, 1])).backward()
        self.assertGreater(float(model.scorer.weight.grad.abs().sum()), 0)
        self.assertGreater(float(model.encoder.embeddings.word_embeddings.weight.grad.abs().sum()), 0)
        self.assertIsNone(model.scorer.bias)

    def test_fp32_reload_exact_and_overwrite_rejected(self):
        for kind in ('original', 'linear'):
            model = HeadControl(encoder(), kind, 42).eval()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)/'saved'
                meta = save_fp32(model, path)
                loaded, after = load_fp32(path)
                self.assertEqual(meta, after)
                self.assertEqual(state_digest(model.state_dict()), state_digest(loaded.state_dict()))
                torch.testing.assert_close(model(*self.batch())[0], loaded.eval()(*self.batch())[0], rtol=0, atol=0)
                with self.assertRaises(FileExistsError):
                    save_fp32(model, path)
                cfg = path/'encoder/config.json'
                cfg.write_text(cfg.read_text()+'\n')
                with self.assertRaises(ValueError):
                    load_fp32(path)

    def test_temperature_convex_bounds_and_invalid_data(self):
        for margins, labels in (([2, -2, 1, -1], [1, 0, 0, 1]), ([2, -2], [1, 0]), ([0, 0], [0, 1])):
            fit = fit_temperature(margins, labels)
            self.assertLessEqual(fit['calibrated_nll'], fit['raw_nll']+1e-14)
            self.assertTrue(.5 <= fit['temperature'] <= 5)
        self.assertEqual(fit_temperature([0, 0], [0, 1])['temperature'], 1.)
        for margins, labels in (([], []), ([float('nan')], [0]), ([1], [2]), ([1, 2], [1])):
            with self.assertRaises(ValueError):
                fit_temperature(margins, labels)

    def test_nll_uses_logits_without_probability_clipping(self):
        raw = logit_metrics([1, 0], [[100., -100.], [-100., 100.]], [True, False])
        calibrated = logit_metrics([1, 0], [[100., -100.], [-100., 100.]], [True, False], 2.)
        self.assertEqual(raw['all']['nll'], 200.)
        self.assertEqual(calibrated['all']['nll'], 100.)
        self.assertEqual(raw['all']['correct'], 0)
        self.assertEqual(raw['truncated']['count'], 1)

    def test_real_input_pair_schedule_and_updates(self):
        import laya.train as upstream
        from cuda_numerics import guard_training
        from export_laya_posthoc import prepare_rules
        from run_laya_head_control import TRAINING
        tok = AutoTokenizer.from_pretrained(str(ROOT/'.cache/laya/english/tokenizer'), local_files_only=True)
        rows = [{'id': str(i), 'text': ('A wonderful film. ' if i%2 else 'An awful film. ')*(i+1), 'label': i%2} for i in range(4)]
        _, items, _, _, _ = prepare_rules(tok, rows)
        _, other, _, _, _ = prepare_rules(tok, [dict(r, label=1-r['label']) for r in rows])
        for a, b in zip(items, other):
            pa, pb = (upstream.encode_item(tok, it, 512, 192, [0, 1], False) for it in (a, b))
            self.assertEqual(pa['ids'], pb['ids']);self.assertNotEqual(pa['target'], pb['target'])
        cfg = upstream.TrainConfig(**dict(TRAINING, seed=42, micro_batch=2, log_every=0))
        torch.manual_seed(11);enc = encoder(len(tok), 512)
        streams = []
        for kind in ('original', 'linear'):
            model = HeadControl(copy.deepcopy(enc), kind, 42)
            before = state_digest(model.encoder.state_dict())
            with tempfile.TemporaryDirectory() as tmp:
                report = Path(tmp)
                with guard_training(torch, upstream), audit_training(upstream, model, tok, items,
                        [r['id'] for r in rows], cfg, report, False):
                    upstream.train_model(model, tok, items, cfg, torch.device('cpu'), 512, 192, parallel=False)
                audit = json.loads((report/'training_audit.json').read_text())
                self.assertEqual(audit['updates'], 4);self.assertEqual(audit['gradient_exposures'], 8)
                streams.append(gzip.open(report/'actual_inputs.jsonl.gz', 'rt').read())
                self.assertNotEqual(before, state_digest(model.encoder.state_dict()))
        self.assertEqual(streams[0], streams[1])

    def test_invalid_markers_and_nonfinite_checkpoint_fail(self):
        model = HeadControl(encoder(), 'linear', 42)
        args = list(self.batch());args[2] = torch.tensor([[-1, 4], [1, 4]])
        with self.assertRaises(ValueError):model(*args)
        with torch.no_grad():model.scorer.weight[0, 0] = float('nan')
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FloatingPointError):save_fp32(model, Path(tmp)/'bad')


if __name__ == '__main__':
    unittest.main()

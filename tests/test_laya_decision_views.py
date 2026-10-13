"""Algorithm checks use synthetic inputs; no real validation or model inference."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from common import ROOT, file_digest
from laya_decision_views import (canonical_logits, candidate_features, compare_views,
    fit_candidate_coefficient, gated_equal_fusion, probabilities, sampled_rule,
    validate_window_training, window_tokens)
from laya_window_training import window_training
from laya_optimization import optimize_training
from run_laya_decision_study import validate_study, export_views, evaluate_views, sliding_windows
from average_laya_checkpoints import average_checkpoints, average_tensor
from train_laya import QUESTION
from prepare_laya_window_evaluation import prepare_pair


class DecisionViewTests(unittest.TestCase):
    def test_exact_token_boundary_and_true_tail(self):
        tokens = list(range(467))
        self.assertEqual(window_tokens(tokens[:466], 'head_tail'), tokens[:466])
        self.assertEqual(window_tokens(tokens, 'head_tail'), tokens[:233]+tokens[-233:])
        self.assertEqual(len(window_tokens(tokens, 'head_tail')), 466)
        self.assertNotIn(233, window_tokens(tokens, 'head_tail'))
        self.assertEqual(window_tokens(tokens, 'head'), tokens[:466])

    def test_reverse_restores_label_meaning_and_negative_tie(self):
        self.assertEqual(canonical_logits([2, -1], [1, 0]).tolist(), [-1, 2])
        self.assertEqual(np.argmax(canonical_logits([0, 0], [1, 0])), 0)
        with self.assertRaises(ValueError):
            canonical_logits([0, np.nan], [1, 0])

    def test_fusion_preserves_short_samples_bitwise_and_averages_probabilities(self):
        head = probabilities([[1, -1], [-4, 4]])
        tail = head.copy()
        tail[1] = probabilities([[1, -1]])[0]
        fused = gated_equal_fusion(head, tail, np.array([False, True]))
        self.assertTrue(np.array_equal(fused[0], head[0]))
        np.testing.assert_array_equal(fused[1], (head[1]+tail[1])/2)
        # Probability averaging is not raw-logit averaging.
        self.assertFalse(np.allclose(fused[1], probabilities([[-1.5, 1.5]])[0]))
        bad = tail.copy()
        bad[0] = bad[0][::-1]
        with self.assertRaisesRegex(ValueError, 'bitwise'):
            gated_equal_fusion(head, bad, np.array([False, True]))

    def test_features_equivariant_and_fit_uses_only_calibration(self):
        h = np.array([[[1, 0, 0], [2, 1, 0]], [[0, 2, 1], [1, 0, 3]]], dtype=float)
        np.testing.assert_array_equal(candidate_features(h[:, ::-1]), candidate_features(h)[:, ::-1])
        z = np.array([[0, .1], [.1, 0.]])
        f = np.array([[1, 0], [0, 1.]])
        fit = fit_candidate_coefficient(z, f, [0, 1], 1.)
        self.assertEqual(fit['fit_split'], 'calibration_only')
        self.assertTrue(fit['head_training_eligible'])
        self.assertGreater(fit['coefficient'], 0)
        self.assertEqual(fit_candidate_coefficient(z, np.ones_like(z), [0, 1], 1.)['coefficient'], 0.)
        with self.assertRaises(ValueError):
            fit_candidate_coefficient(z, f, [0], 1.)

    def test_metrics_report_flip_scope_and_exact_mcnemar(self):
        a = probabilities([[1, 0], [1, 0], [1, 0], [1, 0]])
        b = probabilities([[1, 0], [0, 1], [0, 1], [0, 1]])
        report = compare_views(['a', 'b', 'c', 'd'], [0, 1, 1, 1], a, b, [False, True, True, True])
        self.assertEqual(report['flips_truncated'], 3)
        self.assertEqual(report['flips_untruncated'], 0)
        self.assertEqual(report['paired']['mcnemar_exact_two_sided_p'], .25)

    def test_schedule_does_not_consume_global_randomness(self):
        before = random.getstate()
        rules = [sampled_rule(42, e, f'train/{i}') for e in (0, 1) for i in range(200)]
        self.assertEqual(before, random.getstate())
        self.assertEqual(rules, [sampled_rule(42, e, f'train/{i}') for e in (0, 1) for i in range(200)])
        self.assertEqual(set(rules), {'head', 'head_tail'})

    def test_config_rejects_dev_selected_window_unknown_training_modes_and_bad_hashes(self):
        spec = json.loads((ROOT / 'configs/laya-decision-study-20261013-v1.json').read_text())
        validate_study(spec)
        for bad in (dict(spec, windows=[256]), dict(spec, runtime_profile=None), dict(spec, run_id='../old')):
            with self.assertRaises(ValueError):
                validate_study(bad)
        bad = copy.deepcopy(spec)
        bad['checkpoints'][0]['weight_sha256'] = 'invalid'
        with self.assertRaises(ValueError):
            validate_study(bad)
        with self.assertRaises(ValueError):
            validate_window_training({'mode': 'random', 'probability': .5})


class WindowMechanicalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer
        cls.tok = AutoTokenizer.from_pretrained(str(ROOT / '.cache/laya/english/tokenizer'), local_files_only=True)

    def run_toy(self, mode=None, fail=False, optimized=False):
        import torch
        import laya.train as upstream
        torch.set_num_threads(1)
        typed = [{'state': text, 'questions': {'sentiment': QUESTION},
                  'expected': {'sentiment': ('negative', 'positive')[i%2]}}
                 for i, text in enumerate(['A poor film.', 'An enjoyable film.',
                    'A long, slow story. '*170+'The ending is awful.',
                    'A long, slow story. '*170+'The ending is wonderful.'])]
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Embedding(len(self_tok), 4)
                self.scorer = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 1))
                self.dropout = torch.nn.Dropout(.2)
            def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
                h = self.encoder(input_ids)
                pool = (h*attention_mask[..., None]).sum(1)/attention_mask.sum(1)[:, None]
                h = self.dropout(h+pool[:, None])
                m = h.gather(1, marker_pos[..., None].expand(-1, -1, 4))
                return self.scorer(m).squeeze(-1), None
        self_tok = self.tok
        torch.manual_seed(91)
        model = Toy()
        if fail:
            model.scorer[1].weight.data.fill_(float('nan'))
        config = upstream.TrainConfig(epochs=2, micro_batch=2, grad_accum=1, amp=False,
                    gradient_checkpointing=False, log_every=0, seed=42, loss='soft-ce',
                    label_smoothing=.05 if optimized else 0.)
        originals = (upstream.train_model, upstream.items_from_rows, upstream.encode_item)
        ids = [f'train/{i}' for i in range(4)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with ExitStack() as stack:
                stack.enter_context(window_training(upstream, {'mode': mode} if mode else None, ids, path))
                stack.enter_context(optimize_training(torch, upstream, {'warmup_ratio': .05} if optimized else None, path))
                items, skipped = upstream.items_from_rows(self.tok, typed, 512, 192, 0.)
                self.assertFalse(skipped)
                self.assertEqual([item['target'] for item in items], [[1., 0.], [0., 1.], [1., 0.], [0., 1.]])
                if mode:
                    self.assertGreater(len(items[-1]['state_ids']), 512)
                before = copy.deepcopy(items)
                if fail:
                    from cuda_numerics import guard_training
                    with guard_training(torch, upstream), self.assertRaises(FloatingPointError):
                        upstream.train_model(model, self.tok, items, config, torch.device('cpu'), 512, 192)
                    history = []
                else:
                    history = upstream.train_model(model, self.tok, items, config, torch.device('cpu'), 512, 192)
                self.assertEqual(items, before)
                self.assertIs(upstream.encode_item, originals[2])
            self.assertEqual((upstream.train_model, upstream.items_from_rows, upstream.encode_item), originals)
            audit = json.loads((path / 'window_audit.json').read_text()) if mode else None
            journal = [json.loads(s) for s in (path / 'window_consumption.jsonl').read_text().splitlines()] if mode else []
        return ({k: v.detach().clone() for k, v in model.state_dict().items()},
                history, torch.get_rng_state(), audit, journal)

    def test_prefix_control_is_exact_and_random_keeps_exposure_and_short_inputs(self):
        import torch
        baseline = self.run_toy()
        prefix = self.run_toy('prefix')
        random_view = self.run_toy('random_head_tail')
        self.assertEqual(baseline[1], prefix[1])
        self.assertTrue(torch.equal(baseline[2], prefix[2]))
        self.assertTrue(torch.equal(prefix[2], random_view[2]))
        for key in baseline[0]:
            self.assertTrue(torch.equal(baseline[0][key], prefix[0][key]))
        self.assertEqual(random_view[3]['state'], 'completed')
        self.assertEqual([e['consumed'] for e in random_view[3]['epochs']], [4, 4])
        self.assertEqual([r['sample_id'] for r in prefix[4]], [r['sample_id'] for r in random_view[4]])
        for a, b in zip(prefix[4], random_view[4]):
            if not a['truncated'] or b['rule'] == 'head':
                self.assertEqual(a['input_sha256'], b['input_sha256'])

    def test_failure_restores_all_hooks_and_reports_failed(self):
        self.assertEqual(self.run_toy('random_head_tail', fail=True)[3]['state'], 'failed')

    def test_actual_optimization_wrapper_preserves_control_and_calibration_targets(self):
        import torch
        base = self.run_toy(optimized=True)
        prefix = self.run_toy('prefix', optimized=True)
        randomized = self.run_toy('random_head_tail', optimized=True)
        self.assertEqual(base[1], prefix[1])
        for key in base[0]:
            self.assertTrue(torch.equal(base[0][key], prefix[0][key]))
        self.assertTrue(torch.equal(prefix[2], randomized[2]))
        self.assertEqual(randomized[3]['state'], 'completed')

    def test_actual_export_reverse_and_short_reuse_on_cpu(self):
        import torch
        import laya.train as upstream
        # Tiny model with real upstream collate/forward and scorer hook.
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(len(self_tok), 3)
                self.scorer = torch.nn.Linear(3, 1)
            def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype, detach_encoder=False):
                h = self.embedding(input_ids)
                m = h.gather(1, marker_pos[..., None].expand(-1, -1, 3))
                return self.scorer(m).squeeze(-1), None
        self_tok = self.tok
        torch.manual_seed(5)
        model = Toy().eval()
        rows = [{'id': 'train/short', 'text': 'A wonderful movie.', 'label': 1},
                {'id': 'train/long', 'text': 'The story proceeds slowly. '*180+'An awful ending.', 'label': 0}]
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            outputs, timing, temp = export_views(torch, upstream, model, self.tok, {'temperature': [1., 1., 1.]},
                rows, 'cpu', 2, Path(directory), 'synthetic')
            self.assertEqual(outputs['head'][0]['input_sha256'], outputs['head_tail'][0]['input_sha256'])
            self.assertEqual(outputs['head'][0]['probabilities'], outputs['head_tail'][0]['probabilities'])
            self.assertEqual(timing['head_tail']['forward_rows'], 1)
            self.assertEqual(len(model.scorer._forward_pre_hooks), 0)
            report = evaluate_views(outputs, {'coefficient': 0., 'temperature': temp})
            self.assertEqual(report['single_and_fused']['head']['all']['count'], 2)

    def test_real_predict_long_with_tiny_random_checkpoint(self):
        import torch
        import laya.train as upstream
        from laya.common import DecisionModel
        from transformers import BertConfig, BertModel
        torch.set_num_threads(1)
        torch.manual_seed(7)
        encoder = BertModel(BertConfig(vocab_size=len(self.tok), hidden_size=64,
            num_hidden_layers=1, num_attention_heads=1, intermediate_size=128,
            max_position_embeddings=512))
        model = DecisionModel(encoder, head_layers=1, n_act=2).eval()
        cfg = {'encoder': 'synthetic-local-only', 'head_layers': 1, 'max_len': 512,
               'head_max_len': 192, 'temperature': [1., 1., 1.], 'act_costs': {'escalate': .5}}
        rows = [{'id': 'train/synthetic', 'label': 1,
                 'text': 'The story is slow and dull. '*85+'The final scene is wonderful.'}]
        with tempfile.TemporaryDirectory() as directory, torch.inference_mode():
            folder = Path(directory)
            cp = folder / 'checkpoint'
            upstream.save_checkpoint(model, self.tok, cfg, str(cp))
            outputs, _, _ = export_views(torch, upstream, model, self.tok, cfg, rows, 'cpu', 2, folder, 'validation')
            study = sliding_windows(torch, {'checkpoint': str(cp),
                'weight_sha256': file_digest(cp / 'model.safetensors')}, rows, outputs['head'],
                {'device': 'cpu', 'batch_size': 2}, folder)
            for window in (128, 256, 466):
                result = json.loads((folder / f'validation-long-{window}.json').read_text())['rows']
                self.assertEqual(len(result), 1)
                self.assertGreater(result[0]['window_count'], 1)
                self.assertAlmostEqual(sum(result[0]['probabilities']), 1.)
                self.assertEqual(study[str(window)]['metrics']['count'], 1)


class CheckpointAverageTests(unittest.TestCase):
    def test_weight_average_and_temperature_reset_and_mismatch_rejection(self):
        import torch
        self.assertEqual(average_tensor(torch, 'weight', torch.tensor([2.]), torch.tensor([4.])).tolist(), [3.])
        self.assertEqual(average_tensor(torch, 'temperature', torch.tensor([2.]), torch.tensor([4.])).tolist(), [1.])
        with self.assertRaises(ValueError):
            average_tensor(torch, 'counter', torch.tensor([1]), torch.tensor([2]))
        with self.assertRaises(FloatingPointError):
            average_tensor(torch, 'weight', torch.tensor([float('nan')]), torch.tensor([2.]))

    def test_real_safetensors_output_is_new_and_uncalibrated(self):
        import torch
        from safetensors.torch import save_file, load_file
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = [root / 'one', root / 'two']
            for folder, value in zip(inputs, (2., 4.)):
                folder.mkdir()
                for name in ('encoder', 'tokenizer'):
                    (folder / name).mkdir()
                    (folder / name / 'config.json').write_text('{}')
                (folder / 'rl_agent_config.json').write_text(json.dumps({'max_len': 512, 'temperature': [value]*3}))
                save_file({'weight': torch.tensor([value]), 'temperature': torch.tensor([value]*3)}, folder / 'model.safetensors')
            hashes = [file_digest(p / 'model.safetensors') for p in inputs]
            out = root / 'new'
            average_checkpoints(*inputs, out, *hashes)
            self.assertEqual(load_file(out / 'model.safetensors')['weight'].tolist(), [3.])
            self.assertEqual(json.loads((out / 'rl_agent_config.json').read_text())['temperature'], [1.]*3)
            self.assertTrue(json.loads((out / 'average_manifest.json').read_text())['calibration_required'])
            self.assertEqual(hashes, [file_digest(p / 'model.safetensors') for p in inputs])
            with self.assertRaises(FileExistsError):
                average_checkpoints(*inputs, out, *hashes)


class WindowPairBindingTests(unittest.TestCase):
    def fixture(self, root):
        gradient = [f'train/{i}' for i in range(22100)]
        calibration = [f'train/cal-{i}' for i in range(400)]
        for mode, run in (('prefix', 'prefix-run'), ('random_head_tail', 'random-run')):
            folder = root / 'reports' / run
            folder.mkdir(parents=True)
            spec = json.loads((ROOT / f'configs/laya-decision-window-{mode}-s42-strict-20261013-v1.json').read_text())
            spec['run_id'] = run
            (folder / 'manifest.json').write_text(json.dumps({'spec': spec,
                'gradient_train_ids': gradient, 'calibration_ids': calibration,
                'development_evaluation_scope': 'separate_final_saved_checkpoint_only'}))
            (folder / 'status.json').write_text('{"state":"completed"}')
            (folder / 'window_audit.json').write_text(json.dumps({'state': 'completed',
                'epochs': [{'consumed': 22100}, {'consumed': 22100}]}))
            cp = root / 'checkpoints' / run
            cp.mkdir(parents=True)
            (cp / 'model.safetensors').write_bytes(b'synthetic provenance fixture')

    def test_completed_pair_binds_hashes_and_rejects_extra_training_factor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            with patch('prepare_laya_window_evaluation.ROOT', root):
                spec = prepare_pair('prefix-run', 'random-run', 'new-study')
                self.assertEqual([c['name'] for c in spec['checkpoints']], ['prefix-trained', 'random-window-trained'])
                path = root / 'reports/random-run/manifest.json'
                doc = json.loads(path.read_text())
                doc['spec']['training']['encoder_lr'] *= 2
                path.write_text(json.dumps(doc))
                with self.assertRaisesRegex(ValueError, 'differ beyond'):
                    prepare_pair('prefix-run', 'random-run', 'new-study')

    def test_failed_or_partial_pair_cannot_be_evaluated_as_completed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            (root / 'reports/random-run/status.json').write_text('{"state":"failed"}')
            with patch('prepare_laya_window_evaluation.ROOT', root), self.assertRaises(ValueError):
                prepare_pair('prefix-run', 'random-run', 'new-study')


if __name__ == '__main__':
    unittest.main()

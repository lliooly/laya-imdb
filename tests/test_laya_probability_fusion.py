import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from common import digest, file_digest, write_json
from laya_decision_views import probabilities
import laya_probability_fusion as fusion
import run_laya_probability_fusion as runner


def binary(p):
    p = np.asarray(p)
    return np.column_stack((1-p, p))


class FusionMathTests(unittest.TestCase):
    def test_convex_solution_satisfies_stationarity_and_beats_grid(self):
        y = np.tile([0, 1, 0, 1, 1], 40)
        a = binary(np.tile([.15, .75, .35, .6, .3], 40))
        b = binary(np.tile([.4, .4, .7, .7, .8], 40))
        fit = fusion.fit_mixing_coefficient(a, b, y)
        self.assertGreater(fit['coefficient'], 0)
        self.assertLess(fit['coefficient'], 1)
        self.assertAlmostEqual(fit['derivative_at_solution'], 0, places=12)
        for weight in np.linspace(0, 1, 101):
            p = weight*a[:, 1]+(1-weight)*b[:, 1]
            nll = -np.mean(np.where(y == 1, np.log(p), np.log1p(-p)))
            self.assertLessEqual(fit['penalized_objective'], nll+.01*(weight-.5)**2+1e-14)

    def test_equal_models_boundary_optimum_and_saturated_probabilities(self):
        y = np.asarray([0, 1]*20)
        a = binary(np.where(y == 1, .8, .2))
        self.assertEqual(fusion.fit_mixing_coefficient(a, a, y)['coefficient'], .5)
        b = binary(1-a[:, 1])
        self.assertEqual(fusion.fit_mixing_coefficient(a, b, y)['coefficient'], 1.)
        fit = fusion.fit_mixing_coefficient(binary(y), binary(1-y), y)
        self.assertTrue(np.isfinite(fit['calibration_nll']))
        for value in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                fusion.mix_probabilities(a, b, value)
        with self.assertRaises(ValueError):
            fusion.fit_mixing_coefficient(a, b, np.full(len(y), 2))

    def test_frozen_mixture_reload_and_exact_short_view_guard(self):
        head_a = binary([.5, .3, .8])
        tail_a = binary([.5, .6, .8])
        head_b = binary([.5, .7, .6])
        tail_b = binary([.5, .4, .6])
        mask = np.asarray([False, True, False])
        expected = binary(.3*np.asarray([.5, .45, .8])+.7*np.asarray([.5, .55, .6]))
        fit = json.loads(json.dumps({'coefficient': .3}))
        before = head_a.copy()
        actual = fusion.apply_gated_mixture(head_a, tail_a, head_b, tail_b, mask, fit['coefficient'])
        np.testing.assert_allclose(actual, expected, atol=1e-15)
        np.testing.assert_array_equal(head_a, before)
        self.assertEqual(int(actual[0].argmax()), 0)  # Ties retain canonical negative class.
        tail_a[0] = [.3, .7]
        with self.assertRaises(ValueError):
            fusion.apply_gated_mixture(head_a, tail_a, head_b, tail_b, mask, .3)

    def test_rank_deficient_historical_ols_is_explicit_and_finite(self):
        y = np.asarray([0, 1]*20)
        p = binary(.2+.6*y)
        fit = fusion.fit_historical_ols([p]*8, y)
        self.assertEqual(fit['rank'], 1)
        self.assertTrue(fit['diagnostic_only'])
        actual, clipped = fusion.apply_historical_ols([p]*8, json.loads(json.dumps(fit)))
        x = np.column_stack([p[:, 1]]*8)
        expected_weights = np.linalg.lstsq(x, y, rcond=None)[0]
        np.testing.assert_allclose(fit['weights'], expected_weights)
        np.testing.assert_allclose(actual[:, 1], np.clip(x@expected_weights, 1e-12, 1-1e-12))
        self.assertEqual(clipped, 0)


class FusionProtocolTests(unittest.TestCase):
    def fixture(self, root):
        cal_ids = [f'train/{"pos" if i%2 else "neg"}/cal-{i}.txt' for i in range(400)]
        gradient = [f'train/{"pos" if i%2 else "neg"}/grad-{i}.txt' for i in range(22100)]
        dev_ids = [f'train/{"pos" if i%2 else "neg"}/dev-{i}.txt' for i in range(2500)]
        train_ids = cal_ids+gradient
        for path, ids in ((runner.TRAIN_PATH, train_ids), (runner.VALIDATION_PATH, dev_ids)):
            target = root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(''.join(json.dumps({'id': sid, 'label': int('/pos/' in sid),
                                                  'text': 'synthetic '+sid})+'\n' for sid in ids))
        source_hashes = {n: file_digest(root/p) for n, p in
                         (('train', runner.TRAIN_PATH), ('validation', runner.VALIDATION_PATH))}
        split = {'ids': {'train': train_ids, 'validation': dev_ids}, 'jsonl_sha256': source_hashes}
        write_json(root/runner.SPLIT_PATH, split)
        write_json(root/runner.HISTORY_PATH, {'synthetic': True})
        models = []
        paths = {runner.TRAIN_PATH, runner.VALIDATION_PATH, runner.SPLIT_PATH, runner.HISTORY_PATH}
        for j, name in enumerate(runner.MODELS):
            family, seed = name.split('-s')
            export = f'reports/laya-posthoc-{family}-s{seed}-fp32-v1'
            manifest_path = f'reports/{name}-fixture/manifest.json'
            manifest = {'spec': {'run_id': name+'-fixture', 'device': 'cuda',
                                'training': {'max_len': 512, 'head_max_len': 192, 'amp': False, 'seed': int(seed)}},
                        'train': {'ids': train_ids, 'source_sha256': source_hashes['train']},
                        'validation': {'ids': dev_ids, 'source_sha256': source_hashes['validation']},
                        'calibration_ids': cal_ids, 'gradient_train_ids': gradient,
                        'question': {'sentiment': 'synthetic'},
                        'model_lock': {'sha256': {'tokenizer/tokenizer.json': 'a'*64}}}
            write_json(root/manifest_path, manifest)
            metadata = {'state': 'completed', 'test_accessed': False, 'checkpoint_sha256': str(j)*64,
                        'manifest_sha256': file_digest(root/manifest_path),
                        'split_manifest_sha256': file_digest(root/runner.SPLIT_PATH),
                        'source_sha256': source_hashes, 'device': 'cuda', 'dtype': 'float32',
                        'amp': False, 'tf32': False, 'parallel_layout': False,
                        'review_token_budget': 466, 'head_token_length': 45,
                        'split_counts': {'calibration': 400, 'validation': 2500}, 'native_temperature': 1.5}
            write_json(root/export/'metadata.json', metadata)
            paths.update([manifest_path, export+'/metadata.json'])
            for split_name, ids in (('calibration', cal_ids), ('validation', dev_ids)):
                for rule in runner.RULES:
                    rows = []
                    for i, sid in enumerate(ids):
                        label = int('/pos/' in sid)
                        truncated = i%6 == 0
                        changed = truncated and rule == 'head_tail'
                        z = (1. if label else -1.)*(3 if (i+j)%27 else -1)
                        if changed:
                            z += .3
                        logits = np.asarray([[-z/2, z/2]])
                        rows.append({'sample_id': sid, 'true_label': label, 'logits': logits[0].tolist(),
                                     'probabilities': probabilities(logits, 1.5)[0].tolist(),
                                     'pred_label': int(logits.argmax()), 'input_sha256': digest(sid+str(changed)),
                                     'truncated': truncated, 'review_tokens': 600 if truncated else 120,
                                     'review_token_budget': 466})
                    path = export+f'/{split_name}-{rule}.json'
                    write_json(root/path, {'split': split_name, 'rule': rule, 'rows': rows, 'test_accessed': False})
                    paths.add(path)
            models.append({'name': name, 'export': export, 'manifest': manifest_path,
                           'weight_sha256': metadata['checkpoint_sha256']})
        spec = {'run_id': 'fusion-fixture', 'models': models, 'source_sha256':
                {p: file_digest(root/p) for p in paths}, 'primary_models': list(runner.PRIMARY),
                'mixing_l2': .01, 'primary_reference': 'baseline-s43/head_tail',
                'minimum_delta_pp': .1, 'maximum_mcnemar_p': .05}
        # An invalid official test sentinel must never be opened by this runner.
        (root/'data/processed/test.jsonl').write_text('DO NOT READ THIS OFFICIAL TEST SENTINEL')
        return spec

    def test_replay_freezes_before_development_and_preserves_all_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.fixture(root)
            observed = []
            original = runner.read_predictions

            def inspect_order(root, spec, metadata, split_name, ids, source):
                observed.append(split_name)
                frozen_path = root/'reports'/spec['run_id']/'frozen-fit.json'
                self.assertEqual(frozen_path.exists(), split_name == 'validation')
                if split_name == 'validation':
                    fit = json.loads(frozen_path.read_text())
                    self.assertFalse(fit['validation_used_for_fitting'])
                    self.assertEqual(fit['mixture']['calibration_count'], 400)
                return original(root, spec, metadata, split_name, ids, source)

            with patch.object(runner, 'read_predictions', side_effect=inspect_order):
                result = runner.run_study(spec, root)
            self.assertEqual(observed, ['calibration', 'validation'])
            self.assertFalse(result['test_accessed'])
            self.assertFalse(result['auto_search_or_training'])
            self.assertEqual(result['metrics']['two_model_fitted']['all']['count'], 2500)
            out = root/'reports'/spec['run_id']
            state = json.loads((out/'status.json').read_text())
            self.assertEqual(state['state'], 'completed')
            self.assertEqual(state['frozen_fit_sha256'], file_digest(out/'frozen-fit.json'))
            runner.verify_inputs(root, spec)
            with self.assertRaises(FileExistsError):
                runner.run_study(spec, root)

    def test_input_tampering_rejected_and_failure_recorded_before_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.fixture(root)
            source_path = root/spec['models'][0]['export']/'calibration-head.json'
            source_path.write_text(source_path.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                runner.run_study(spec, root)
            out = root/'reports'/spec['run_id']
            self.assertEqual(json.loads((out/'status.json').read_text())['state'], 'failed')
            self.assertFalse((out/'frozen-fit.json').exists())

    def test_semantically_invalid_predictions_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = self.fixture(root)
            split, cal_ids, meta = runner.admit_sources(root, spec)
            source = runner.read_source(root, spec, runner.TRAIN_PATH, split['ids']['train'])
            path = spec['models'][0]['export']+'/calibration-head.json'
            doc = json.loads((root/path).read_text())
            for key, value in (('true_label', 1), ('input_sha256', 'invalid'),
                               ('probabilities', [.1, .9]), ('review_tokens', 1),
                               ('sample_id', 'train/neg/outside.txt')):
                with self.subTest(key=key):
                    altered = copy.deepcopy(doc)
                    altered['rows'][0][key] = value
                    write_json(root/path, altered)
                    spec['source_sha256'][path] = file_digest(root/path)
                    with self.assertRaises(ValueError):
                        runner.read_predictions(root, spec, meta, 'calibration', cal_ids, source)

    def test_protocol_rejects_extra_search_dimensions_and_test_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = self.fixture(Path(directory))
            for mutate in (
                lambda s: s.update(mixing_l2=.02),
                lambda s: s.update(primary_models=['baseline-s42', 'combined-s43']),
                lambda s: s['source_sha256'].update({'data/processed/test.jsonl': 'a'*64}),
                lambda s: s.update(run_id='../old'),
            ):
                altered = copy.deepcopy(spec)
                mutate(altered)
                with self.assertRaises(ValueError):
                    runner.validate_spec(altered)


if __name__ == '__main__':
    unittest.main()

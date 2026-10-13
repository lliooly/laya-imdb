import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import laya_posthoc as posthoc
from common import file_digest


class PosthocTests(unittest.TestCase):
    def test_two_parameter_fit_satisfies_stationarity_and_beats_coarse_grid(self):
        margins = np.repeat([-1., 0., 1.], 100)
        labels = np.concatenate([np.asarray([1] * positive + [0] * (100 - positive))
                                 for positive in [19, 55, 86]])
        fit = posthoc.fit_temperature_bias(margins, labels)
        calibrated = posthoc.apply_calibration(margins, fit)
        residual = posthoc.sigmoid(calibrated) - labels
        self.assertGreater(fit['temperature'], .25)
        self.assertLess(fit['temperature'], 4)
        self.assertAlmostEqual(float(np.mean(residual * margins)), 0., places=10)
        self.assertAlmostEqual(float(np.mean(residual) + .02 * fit['bias']), 0., places=10)
        for inverse in np.linspace(.25, 4, 21):
            for bias in np.linspace(-3, 3, 21):
                z = margins * inverse + bias
                objective = np.logaddexp(0., np.where(labels == 1, -z, z)).mean() + .01 * bias * bias
                self.assertLessEqual(fit['penalized_objective'], objective + 1e-12)

    def test_bias_improves_known_imbalance_and_is_deterministic(self):
        margins = np.zeros(100)
        labels = np.asarray([0] * 20 + [1] * 80)
        fit = posthoc.fit_temperature_bias(margins, labels)
        self.assertEqual(fit, posthoc.fit_temperature_bias(margins, labels))
        self.assertGreater(fit['bias'], 0)
        self.assertLess(fit['calibration_nll'], np.log(2))
        pred = posthoc.apply_calibration(margins, fit) > 0
        self.assertGreater(np.mean(pred == labels), np.mean((margins > 0) == labels))
        self.assertLessEqual(fit['temperature'], 4)
        self.assertGreaterEqual(fit['temperature'], .25)
        self.assertLessEqual(abs(fit['bias']), 3)

    def test_positive_temperature_preserves_argmax_and_tie(self):
        margins = np.asarray([-100., -.01, 0., .01, 100.])
        for temperature in [.25, 1., 4.]:
            transformed = posthoc.apply_calibration(margins, {'temperature': temperature})
            np.testing.assert_array_equal(transformed > 0, margins > 0)
        for temperature in [0, -1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                posthoc.apply_calibration(margins, {'temperature': temperature})

    def test_invalid_fit_inputs_fail(self):
        for margins, labels in [([float('nan')], [1]), ([float('inf')], [1]),
                                ([1], [2]), ([1, 2], [0]), ([], [])]:
            with self.assertRaises(ValueError):
                posthoc.fit_temperature_bias(margins, labels)

    def _fixture(self, root):
        root = Path(root)
        export = root / 'export'
        export.mkdir()
        cal = [f'train/{"pos" if i % 2 else "neg"}/cal-{i}.txt' for i in range(400)]
        gradient = [f'train/neg/gradient-{i}.txt' for i in range(16)]
        val = [f'train/{"pos" if i % 2 else "neg"}/val-{i}.txt' for i in range(2500)]
        train = cal + gradient
        source_rows = {}
        for name, ids in [('train', train), ('validation', val)]:
            rows = [{'id': sid, 'label': int('/pos/' in sid), 'text': 'Frozen synthetic text ' + sid} for sid in ids]
            source_rows.update({r['id']: r for r in rows})
            (root / (name + '.jsonl')).write_text(''.join(json.dumps(r) + '\n' for r in rows))
        hashes = {name: file_digest(root / (name + '.jsonl')) for name in ('train', 'validation')}
        manifest = {'train': {'ids': train, 'source_sha256': hashes['train']},
                    'validation': {'ids': val, 'source_sha256': hashes['validation']},
                    'gradient_train_ids': gradient, 'calibration_ids': cal}
        self._write(root / 'manifest.json', manifest)
        self._write(root / 'split.json', {'ids': {'train': train, 'validation': val}, 'jsonl_sha256': hashes})
        metadata = {'schema_version': 1, 'state': 'completed', 'test_accessed': False,
                    'manifest_sha256': file_digest(root / 'manifest.json'),
                    'checkpoint_sha256': 'a' * 64, 'source_sha256': hashes, 'native_temperature': 1.5}
        self._write(export / 'metadata.json', metadata)
        for rule in posthoc.RULES:
            for split, ids in [('calibration', cal), ('validation', val)]:
                rows = []
                for i, sid in enumerate(ids):
                    label = source_rows[sid]['label']
                    margin = (.2 if label else -.2) - .3
                    if rule == 'head_tail':
                        margin += .1
                    rows.append({'sample_id': sid, 'true_label': label, 'input_sha256': 'b' * 64,
                                 'logits': [0., margin], 'pred_label': int(margin > 0)})
                self._write(export / f'{split}-{rule}.json', {'split': split, 'rule': rule, 'rows': rows})
        return (export, root / 'manifest.json', root / 'split.json', root / 'train.jsonl', root / 'validation.jsonl')

    @staticmethod
    def _write(path, value):
        Path(path).write_text(json.dumps(value) + '\n')

    def _mutate(self, path, mutation):
        doc = json.loads(Path(path).read_text())
        mutation(doc)
        self._write(path, doc)

    def test_full_audit_and_validation_cannot_influence_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(directory)
            first = posthoc.analyze_export(*args)
            self.assertFalse(first['test_accessed'])
            self.assertFalse(first['validation_used_for_fitting'])
            self.assertEqual(first['rules']['head']['fit']['calibration_count'], 400)
            self.assertEqual(first['rules']['head']['validation_metrics']['raw']['count'], 2500)
            self.assertGreater(first['rules']['head']['temperature_bias_vs_raw']['corrected'], 0)
            for rule in posthoc.RULES:
                def change_validation(doc):
                    for row in doc['rows']:
                        row['logits'] = [0., 99.]
                        row['pred_label'] = 1
                self._mutate(args[0] / f'validation-{rule}.json', change_validation)
            second = posthoc.analyze_export(*args)
            for rule in posthoc.RULES:
                self.assertEqual(first['rules'][rule]['fit'], second['rules'][rule]['fit'])
            self.assertNotEqual(first['rules']['head']['validation_metrics']['raw'],
                                second['rules']['head']['validation_metrics']['raw'])

    def test_source_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(directory)
            args[-1].write_text(args[-1].read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'source hash'):
                posthoc.analyze_export(*args)

    def test_prediction_bad_split_nonfinite_wrong_label_and_mixed_ids_rejected(self):
        mutations = [
            lambda d: d.update(split='validation'),
            lambda d: d['rows'][0].update(logits=[0., float('nan')]),
            lambda d: d['rows'][0].update(probabilities=[float('inf'), 0.]),
            lambda d: d['rows'][0].update(true_label=1),
            lambda d: d['rows'][0].update(sample_id='train/neg/gradient-0.txt'),
            lambda d: d['rows'][0].update(sample_id='train/neg/val-0.txt'),
            lambda d: d['rows'][0].update(sample_id='test/neg/0.txt'),
            lambda d: d['rows'][0].update(input_sha256='not-a-digest'),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                args = self._fixture(directory)
                self._mutate(args[0] / 'calibration-head.json', mutation)
                with self.assertRaises(ValueError):
                    posthoc.analyze_export(*args)

    def test_manifest_train_calibration_validation_intersections_rejected(self):
        for donor in ['gradient_train_ids', 'validation']:
            with self.subTest(donor=donor), tempfile.TemporaryDirectory() as directory:
                args = self._fixture(directory)
                doc = json.loads(args[1].read_text())
                bad = doc[donor][0] if donor == 'gradient_train_ids' else doc[donor]['ids'][0]
                doc['calibration_ids'][0] = bad
                self._write(args[1], doc)
                self._mutate(args[0] / 'metadata.json', lambda d: d.update(manifest_sha256=file_digest(args[1])))
                with self.assertRaisesRegex(ValueError, 'disjoint'):
                    posthoc.analyze_export(*args)

    def test_original_400_calibration_and_full_validation_required(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self._fixture(directory)
            self._mutate(args[1], lambda d: d['calibration_ids'].pop())
            self._mutate(args[0] / 'metadata.json', lambda d: d.update(manifest_sha256=file_digest(args[1])))
            with self.assertRaisesRegex(ValueError, '400 calibration'):
                posthoc.analyze_export(*args)


if __name__ == '__main__':
    unittest.main()

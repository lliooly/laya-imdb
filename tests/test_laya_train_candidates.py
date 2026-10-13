import copy
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_laya_train_candidates import select_gradient


class GradientIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = [{'id': f'train/neg/{i}_1.txt', 'label': 0} for i in range(22500)]
        ids = [r['id'] for r in cls.rows]
        dev = [f'train/pos/{i}_7.txt' for i in range(2500)]
        cls.manifest = {'gradient_train_ids': ids[400:], 'calibration_ids': ids[:400],
                        'train': {'ids': ids}, 'validation': {'ids': dev}}
        cls.split = {'ids': {'train': ids, 'validation': dev}}

    def test_preserves_manifest_gradient_order_only(self):
        m = copy.deepcopy(self.manifest)
        m['gradient_train_ids'].reverse()
        selected = select_gradient(m, self.split, self.rows)
        self.assertEqual([r['id'] for r in selected], m['gradient_train_ids'])
        self.assertFalse(set(m['calibration_ids']) & {r['id'] for r in selected})

    def test_partition_overlap_duplicate_and_missing_rejected(self):
        for field, replacement in [('gradient_train_ids', self.manifest['calibration_ids'][0]),
                                   ('validation', self.manifest['gradient_train_ids'][0])]:
            m = copy.deepcopy(self.manifest); s = copy.deepcopy(self.split)
            if field == 'validation':
                m[field]['ids'][0] = replacement; s['ids']['validation'][0] = replacement
            else: m[field][0] = replacement
            with self.assertRaises(ValueError): select_gradient(m, s, self.rows)
        m = copy.deepcopy(self.manifest); m['gradient_train_ids'][0] = m['gradient_train_ids'][1]
        with self.assertRaises(ValueError): select_gradient(m, self.split, self.rows)
        with self.assertRaises(ValueError): select_gradient(self.manifest, self.split, self.rows[:-1])

    def test_source_order_or_label_changes_rejected(self):
        rows = copy.deepcopy(self.rows); rows[0], rows[1] = rows[1], rows[0]
        with self.assertRaises(ValueError): select_gradient(self.manifest, self.split, rows)
        rows = copy.deepcopy(self.rows); rows[0]['label'] = True
        with self.assertRaises(ValueError): select_gradient(self.manifest, self.split, rows)


if __name__ == '__main__': unittest.main()

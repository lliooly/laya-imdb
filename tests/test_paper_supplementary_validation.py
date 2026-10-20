"""Protect cohort membership and grouped sampling in supplementary analysis."""
import unittest
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from paper_supplementary_validation import text_groups, representative_indices, cluster_interval


class TextGroups(unittest.TestCase):
    def test_representatives_do_not_depend_on_row_order(self):
        records = [dict(id='b', text='same', label=0), dict(id='a', text='same', label=0),
                   dict(id='c', text='other', label=1)]
        selected = representative_indices(records, text_groups(records))
        self.assertEqual([records[i]['id'] for i in selected], ['a', 'c'])
        records.reverse()
        selected = representative_indices(records, text_groups(records))
        self.assertEqual([records[i]['id'] for i in selected], ['a', 'c'])

    def test_conflicting_group_fails_instead_of_silently_picking_a_label(self):
        with self.assertRaises(ValueError):
            text_groups([dict(id='a', text='same', label=0), dict(id='b', text='same', label=1)])

    def test_cluster_draw_keeps_duplicate_rows_together(self):
        records = [dict(id='a', text='same', label=0), dict(id='b', text='same', label=0),
                   dict(id='c', text='other', label=1)]
        interval = cluster_interval(records, text_groups(records), np.zeros(3, bool), np.array([1, 1, 0]), 100, 1)
        self.assertAlmostEqual(interval['lower'], 2/3)
        self.assertAlmostEqual(interval['upper'], 2/3)
        self.assertEqual(interval['group_count'], 2)

    def test_identical_predictions_have_zero_interval(self):
        records = [dict(id=str(i), text=str(i), label=i % 2) for i in range(8)]
        correct = np.array([1, 0, 1, 1, 0, 0, 1, 0], bool)
        interval = cluster_interval(records, text_groups(records), correct, correct, 100, 4)
        self.assertEqual((interval['lower'], interval['upper']), (0., 0.))


if __name__ == '__main__':
    unittest.main()

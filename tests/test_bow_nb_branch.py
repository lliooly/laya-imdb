"""Scientific-boundary and sparse-transform checks on synthetic examples."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy.sparse import csr_matrix
from bow_nb_branch import (TOKEN_PATTERN, calibrated_probability, fit_sigmoid,
    fit_sparse_weight, make_vectorizer, mix, nb_log_count_ratio, validate_partitions)


class SparseBranchTests(unittest.TestCase):
    def test_nb_ratio_uses_class_document_counts_and_smoothing(self):
        x = csr_matrix([[1., 1., 0.], [1., 0., 0.], [0., 0., 1.], [0., 1., 1.]])
        expected = np.log(np.array([3., 2., 1.])/6)-np.log(np.array([1., 2., 3.])/6)
        np.testing.assert_allclose(nb_log_count_ratio(x, [1, 1, 0, 0]), expected)

    def test_unseen_documents_do_not_modify_vocabulary_or_idf(self):
        v = make_vectorizer('tfidf_lr', min_df=1, max_features=100)
        v.fit(['good movie 8/10', 'not good 1/10'])
        vocabulary, idf = v.vocabulary_.copy(), v.idf_.copy()
        v.transform(['unseenunique 999/1000'])
        self.assertEqual(v.vocabulary_, vocabulary)
        np.testing.assert_array_equal(v.idf_, idf)
        self.assertIn('8/10', vocabulary)
        self.assertIn('not', vocabulary)
        self.assertNotIn('unseenunique', vocabulary)

    def test_disjoint_partitions_fail_closed(self):
        with self.assertRaises(ValueError):
            validate_partitions(['a'], ['a'], ['c'])
        with self.assertRaises(ValueError):
            validate_partitions(['a', 'a'], ['b'], ['c'])

    def test_bad_sparse_branch_gets_exactly_zero_weight(self):
        neural = np.array([.01, .99, .05, .95])
        sparse = 1-neural
        fitted = fit_sparse_weight(neural, sparse, [0, 1, 0, 1])
        self.assertEqual(fitted['sparse_weight'], 0.)
        np.testing.assert_array_equal(mix(neural, sparse, 0), neural)

    def test_useful_branch_respects_upper_bound(self):
        fit = fit_sparse_weight([.6, .4], [.01, .99], [0, 1])
        self.assertEqual(fit['sparse_weight'], .25)
        with self.assertRaises(ValueError):
            mix([.1], [.9], .3)

    def test_calibration_is_monotonic_and_finite(self):
        z = np.array([-3., -1., 1., 3.])
        fit = fit_sigmoid(z, [0, 0, 1, 1])
        p = calibrated_probability(z, fit)
        self.assertTrue(np.all(np.diff(p) > 0))
        self.assertTrue(np.isfinite(p).all())
        self.assertGreaterEqual(fit['slope'], .05)
        self.assertLessEqual(fit['slope'], 20.)


if __name__ == '__main__':
    unittest.main()

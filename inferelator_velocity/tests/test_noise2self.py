import numpy as np
import scipy.sparse as sps
import sklearn.metrics
import numpy.testing as npt

import unittest

from inferelator_velocity.utils.graph import local_optimal_knn
from inferelator_velocity.utils.noise2self import (_dist_to_row_stochastic, _search_k,
                                                   knn_noise2self)

M, N = 100, 10

RNG = np.random.default_rng(100)

BASE = RNG.negative_binomial(
    np.linspace(5, 50, N).astype(int),
    0.25,
    (M, N)

)

NOISE = RNG.negative_binomial(
    20,
    0.75,
    (M, N)
)

EXPR = BASE + NOISE
DIST = sklearn.metrics.pairwise_distances(EXPR, metric='cosine')

def _knn(k):
    return local_optimal_knn(sps.csr_matrix(DIST), np.array([k] * 100), keep='smallest')


class TestRowStochastic(unittest.TestCase):

    def test_full_k(self):
        graph = sps.csr_matrix(DIST)

        row_stochastic = _dist_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.ones_like(row_sums), row_sums)
        self.assertEqual(len(row_sums), M)

    def test_small_k(self):
        graph = _knn(3)

        npt.assert_array_equal(graph.getnnz(axis=1), np.full(M, 3))

        row_stochastic = _dist_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.ones_like(row_sums), row_sums)
        self.assertEqual(len(row_sums), M)

    def test_zero_k(self):

        row_stochastic = _dist_to_row_stochastic(sps.csr_matrix((M, M), dtype=float))
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.zeros_like(row_sums), row_sums)

    def test_ksearch_regression(self):

        mse = _search_k(EXPR, DIST, np.arange(1, 7))
        self.assertEqual(np.argmin(mse), 4)

        npt.assert_almost_equal(
            np.array([234.314, 166.83420601, 149.88290938, 143.72348837, 138.18590639, 139.83859323]),
            mse
        )

    def test_knn_select_stack_regression(self):

        opt_pc, opt_k, local_ks = knn_noise2self(
            EXPR,
            np.arange(1, 11),
            np.array([3, 5, 7]),
            verbose=True
        )

        self.assertEqual(opt_pc, 3)
        self.assertEqual(opt_k, 4)

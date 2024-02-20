import numpy as np
import scipy.sparse as sps
import sklearn.metrics
import numpy.testing as npt
import anndata as ad
import unittest

from inferelator_velocity.utils.graph import local_optimal_knn
from inferelator_velocity.utils.noise2self import (
    _dist_to_row_stochastic,
    _connect_to_row_stochastic,
    _search_k,
    knn_noise2self,
    standardize_data
)
from inferelator_velocity import (
    program_graphs,
    global_graph
)
from inferelator_velocity.utils.keys import (
    PROGRAM_KEY,
    PROG_NAMES_SUBKEY,
    NOISE2SELF_KEY,
    NOISE2SELF_DIST_KEY,
    OBSP_DIST_KEY,
    UNS_GRAPH_SUBKEY
)

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
PEAKS = RNG.choice([0, 1], (M, N), p=(0.9, 0.1))

DIST = sklearn.metrics.pairwise_distances(EXPR, metric='cosine')
PDIST = sklearn.metrics.pairwise_distances(PEAKS, metric='cosine')

ADATA = ad.AnnData(EXPR.astype(int))


def _knn(k):
    return local_optimal_knn(
        sps.csr_matrix(DIST),
        np.array([k] * 100),
        keep='smallest'
    )


class TestRowStochastic(unittest.TestCase):

    loss = 'mse'

    def test_full_k(self):
        graph = sps.csr_matrix(DIST)

        row_stochastic = _dist_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.ones_like(row_sums), row_sums)
        self.assertEqual(len(row_sums), M)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))

    def test_full_k_connect(self):
        graph = sps.csr_matrix(DIST)

        row_stochastic = _connect_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.ones_like(row_sums), row_sums)
        self.assertEqual(len(row_sums), M)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))

    def test_small_k(self):
        graph = _knn(3)

        npt.assert_array_equal(graph.getnnz(axis=1), np.full(M, 3))

        row_stochastic = _dist_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.ones_like(row_sums), row_sums)
        self.assertEqual(len(row_sums), M)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))

    def test_small_k_connect(self):
        graph = _knn(3)

        npt.assert_array_equal(graph.getnnz(axis=1), np.full(M, 3))

        row_stochastic = _connect_to_row_stochastic(graph)
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.ones_like(row_sums), row_sums)
        self.assertEqual(len(row_sums), M)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))

    def test_zero_k(self):

        row_stochastic = _dist_to_row_stochastic(sps.csr_matrix((M, M), dtype=float))
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.zeros_like(row_sums), row_sums)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))

    def test_zero_k_connect(self):

        row_stochastic = _connect_to_row_stochastic(sps.csr_matrix((M, M), dtype=float))
        row_sums = row_stochastic.sum(axis=1).A1

        npt.assert_almost_equal(np.zeros_like(row_sums), row_sums)

        self.assertTrue(sps.isspmatrix_csr(row_stochastic))


class TestKNNSearch(unittest.TestCase):

    data = EXPR.astype(float)
    dist = DIST.copy()
    normalize = 'log'
    loss = 'mse'
    correct_loss = np.array([
        234.314,
        166.83420601,
        149.88290938,
        143.72348837,
        138.18590639,
        139.83859323
    ])
    correct_mse_argmin = 4
    correct_opt_pc = 7
    correct_opt_k = 4

    def test_ksearch_regression(self):

        mse = _search_k(
            self.data,
            self.dist,
            np.arange(1, 7),
            loss=self.loss
        )

        self.assertEqual(np.argmin(mse), self.correct_mse_argmin)

        npt.assert_almost_equal(
            self.correct_loss,
            mse
        )

    def test_ksearch_regression_sparse(self):

        mse = _search_k(
            sps.csr_matrix(self.data),
            self.dist,
            np.arange(1, 7),
            X_compare=self.data,
            loss=self.loss
        )

        self.assertEqual(np.argmin(mse), self.correct_mse_argmin)

        npt.assert_almost_equal(
            self.correct_loss,
            mse
        )

    def test_knn_select_stack_regression(self):

        _, opt_pc, opt_k, local_ks = knn_noise2self(
            self.data,
            np.arange(1, 11),
            np.array([3, 5, 7]),
            verbose=True,
            loss=self.loss,
            standardization_method=self.normalize
        )

        self.assertEqual(opt_pc, self.correct_opt_pc)
        self.assertEqual(opt_k, self.correct_opt_k)

    def test_knn_select_stack_regression_sparse(self):

        obsp, opt_pc, opt_k, local_ks = knn_noise2self(
            sps.csr_matrix(self.data),
            np.arange(1, 11),
            np.array([3, 5, 7]),
            verbose=True,
            loss=self.loss,
            standardization_method=self.normalize
        )

        self.assertEqual(opt_pc, self.correct_opt_pc)
        self.assertEqual(opt_k, self.correct_opt_k)

    def test_knn_select_stack_regression_nopcsearch(self):

        _, opt_pc, opt_k, local_ks = knn_noise2self(
            self.data,
            np.arange(1, 11),
            5,
            verbose=True,
            loss=self.loss,
            standardization_method=self.normalize
        )

        self.assertEqual(opt_pc, 5)
        self.assertIsNone(opt_k)


class TestKNNSearchNoNorm(TestKNNSearch):

    normalize = None
    data = standardize_data(
        ad.AnnData(EXPR.astype(np.float32))
    ).X

    @unittest.skip
    def test_ksearch_regression(self):
        pass

    @unittest.skip
    def test_ksearch_regression_sparse(self):
        pass


class TestKNNSearchLogLoss(TestKNNSearch):

    normalize = None
    data = PEAKS.astype(float)
    dist = PDIST.copy()
    loss = 'log_loss'
    correct_loss = np.array([
        0.999322,
        0.6487723,
        0.3080371,
        0.2210291,
        0.2230496,
        0.1966515
    ])
    correct_mse_argmin = 5
    correct_opt_pc = 7
    correct_opt_k = 8


class TestProgramGraphs(unittest.TestCase):

    def test_global_graph(self):

        adata = ADATA.copy()

        self.assertFalse(NOISE2SELF_KEY in adata.uns)
        self.assertFalse(NOISE2SELF_DIST_KEY in adata.obsp)

        global_graph(
            adata,
            npcs=np.array([3, 5, 7]),
            neighbors=np.arange(1, 11)
        )

        self.assertTrue(NOISE2SELF_KEY in adata.uns)
        self.assertTrue(NOISE2SELF_DIST_KEY in adata.obsp)

        self.assertEqual(adata.uns[NOISE2SELF_KEY]['npcs'], 7)
        self.assertEqual(adata.uns[NOISE2SELF_KEY]['neighbors'], 4)

    def test_program_graph(self):

        adata = ADATA.copy()
        adata.var[PROGRAM_KEY] = '0'
        adata.uns[PROGRAM_KEY] = {
            PROG_NAMES_SUBKEY: ['0']
        }

        unskey = UNS_GRAPH_SUBKEY.format(prog='0')

        self.assertFalse(unskey in adata.uns[PROGRAM_KEY])
        self.assertFalse(OBSP_DIST_KEY.format(prog='0') in adata.obsp)

        program_graphs(
            adata,
            npcs=np.array([3, 5, 7]),
            neighbors=np.arange(1, 11)
        )

        self.assertTrue(unskey in adata.uns[PROGRAM_KEY])
        self.assertTrue(OBSP_DIST_KEY.format(prog='0') in adata.obsp)

        self.assertEqual(adata.uns[PROGRAM_KEY][unskey]['npcs'], 7)
        self.assertEqual(adata.uns[PROGRAM_KEY][unskey]['neighbors'], 4)

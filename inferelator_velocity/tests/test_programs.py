import unittest

import numpy as np
import numpy.testing as npt
import anndata as ad

from inferelator_velocity.programs import information_distance, _make_array_discrete, program_select
from inferelator_velocity.metrics.information import mutual_information, _shannon_entropy

N = 1000
BINS = 10

EXPRESSION = np.random.default_rng(222222).random((N, 6))
EXPRESSION[:, 0] = (100 * EXPRESSION[:, 0]).astype(int)
EXPRESSION[:, 1] = EXPRESSION[:, 0] * 1.75 - 0.5
EXPRESSION[:, 2] = EXPRESSION[:, 0] ** 2
EXPRESSION[:, 3] = 0
EXPRESSION[:, 4] = np.arange(N)
EXPRESSION[:, 5] = np.arange(N) * 2 + 10


EXPRESSION_ADATA = ad.AnnData(EXPRESSION.astype(int), dtype=int)

ADATA_UNS_PROGRAM_KEYS = [
    'metric',
    'leiden_correlation',
    'metric_genes',
    'information_distance',
    'cluster_program_map',
    'n_comps',
    'n_programs',
    'program_names',
    'molecular_cv_loss'
]

PROGRAMS = ['1', '1', '0', '-1', '0', '0']
PROGRAMS_EUCLID = ['0', '0', '0', '-1', '1', '1']


class TestProgramMetrics(unittest.TestCase):

    def test_binning(self):

        expr = _make_array_discrete(EXPRESSION, BINS)

        npt.assert_equal(expr[:, 3], np.zeros_like(expr[:, 3]))
        npt.assert_equal(expr[:, 4], np.repeat(np.arange(BINS), N / BINS))

        self.assertEqual(expr[:, 0].min(), 0)
        self.assertEqual(expr[:, 0].max(), 9)

    def test_entropy(self):

        expr = _make_array_discrete(EXPRESSION, BINS)
        entropy = _shannon_entropy(expr, 10, logtype=np.log2)

        print(entropy)
        self.assertTrue(np.all(entropy >= 0))
        npt.assert_almost_equal(entropy[4], np.log2(BINS))
        npt.assert_almost_equal(entropy[3], 0.)

        entropy = _shannon_entropy(expr, 10, logtype=np.log)

        self.assertTrue(np.all(entropy >= 0))
        npt.assert_almost_equal(entropy[3], 0.)
        npt.assert_almost_equal(entropy[4], np.log(BINS))

    def test_mutual_info(self):

        expr = _make_array_discrete(EXPRESSION, BINS)

        entropy = _shannon_entropy(expr, 10, logtype=np.log2)
        mi = mutual_information(expr, 10, logtype=np.log2)

        self.assertTrue(np.all(mi >= 0))
        npt.assert_array_equal(mi[:, 3], np.zeros_like(mi[:, 3]))
        npt.assert_array_equal(mi[3, :], np.zeros_like(mi[3, :]))
        npt.assert_array_almost_equal(np.diagonal(mi), entropy)

    def test_info_distance(self):

        expr = _make_array_discrete(EXPRESSION, BINS)

        entropy = _shannon_entropy(expr, 10, logtype=np.log2)
        mi = mutual_information(expr, 10, logtype=np.log2)

        with np.errstate(divide='ignore', invalid='ignore'):
            calc_dist = 1 - mi / (entropy[:, None] + entropy[None, :] - mi)
            calc_dist[np.isnan(calc_dist)] = 0.

        i_dist, mi_from_dist = information_distance(expr, BINS, logtype=np.log2, return_information=True)

        self.assertTrue(np.all(i_dist >= 0))
        npt.assert_almost_equal(mi, mi_from_dist)
        npt.assert_almost_equal(i_dist, calc_dist)
        npt.assert_array_almost_equal(np.diagonal(i_dist), np.zeros_like(np.diagonal(i_dist)))

class TestProgram(unittest.TestCase):

    def test_find_program(self):

        adata = EXPRESSION_ADATA.copy()

        program_select(adata, verbose=True, filter_to_hvg=False)

        for k in ADATA_UNS_PROGRAM_KEYS:
            self.assertIn(k, adata.uns['programs'].keys())

        self.assertListEqual(
            PROGRAMS,
            adata.var['program'].tolist()
        )

    def test_find_prog_euclid(self):

        adata = EXPRESSION_ADATA.copy()

        program_select(adata, verbose=True, filter_to_hvg=False, metric='euclidean')

        self.assertListEqual(
            PROGRAMS_EUCLID,
            adata.var['program'].tolist()
        )

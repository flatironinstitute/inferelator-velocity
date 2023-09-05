import unittest

import numpy as np
import numpy.testing as npt
import anndata as ad

from inferelator_velocity.programs import (
    information_distance,
    _make_array_discrete,
    program_select
)
from inferelator_velocity.metrics.information import (
    mutual_information,
    _shannon_entropy
)
from inferelator_velocity.program_genes import assign_genes_to_programs

from inferelator_velocity.plotting.programs import programs_summary

N = 1000
BINS = 10

EXPRESSION = np.zeros((N, 6), dtype=int)
EXPRESSION[:, 0] = (100 * np.random.default_rng(222222).random(N)).astype(int)
EXPRESSION[:, 1] = EXPRESSION[:, 0] * 1.75 - 0.5
EXPRESSION[:, 2] = EXPRESSION[:, 0] ** 2
EXPRESSION[:, 3] = 0
EXPRESSION[:, 4] = np.arange(N)
EXPRESSION[:, 5] = np.arange(N) * 2 + 10


EXPRESSION_ADATA = ad.AnnData(EXPRESSION.astype(int))

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

PROGRAMS = ['0', '0', '0', '-1', '1', '1']
PROGRAMS_ASSIGNED = ['0', '0', '0', '0', '1', '1']

TIMES_0 = EXPRESSION[:, 0] / 100
TIMES_1 = np.arange(N).astype(float)


class TestProgramMetrics(unittest.TestCase):

    n_jobs = 1

    def test_binning(self):

        expr = _make_array_discrete(EXPRESSION, BINS)

        npt.assert_equal(expr[:, 3], np.zeros_like(expr[:, 3]))
        npt.assert_equal(expr[:, 4], np.repeat(np.arange(BINS), N / BINS))

        self.assertEqual(expr[:, 0].min(), 0)
        self.assertEqual(expr[:, 0].max(), 9)

    def test_entropy(self):

        expr = _make_array_discrete(EXPRESSION, BINS)
        entropy = _shannon_entropy(
            expr,
            10,
            logtype=np.log2,
            n_jobs=self.n_jobs
        )

        self.assertTrue(np.all(entropy >= 0))
        npt.assert_almost_equal(entropy[4], np.log2(BINS))
        npt.assert_almost_equal(entropy[3], 0.)

        entropy = _shannon_entropy(
            expr,
            10,
            logtype=np.log,
            n_jobs=self.n_jobs
        )

        self.assertTrue(np.all(entropy >= 0))
        npt.assert_almost_equal(entropy[3], 0.)
        npt.assert_almost_equal(entropy[4], np.log(BINS))

    def test_mutual_info(self):

        expr = _make_array_discrete(EXPRESSION, BINS)

        entropy = _shannon_entropy(
            expr,
            10,
            logtype=np.log2,
            n_jobs=self.n_jobs
        )
        mi = mutual_information(
            expr,
            10,
            logtype=np.log2,
            n_jobs=self.n_jobs
        )

        self.assertTrue(np.all(mi >= 0))
        npt.assert_array_equal(mi[:, 3], np.zeros_like(mi[:, 3]))
        npt.assert_array_equal(mi[3, :], np.zeros_like(mi[3, :]))
        npt.assert_array_almost_equal(np.diagonal(mi), entropy)

    def test_info_distance(self):

        expr = _make_array_discrete(EXPRESSION, BINS)

        entropy = _shannon_entropy(
            expr,
            10,
            logtype=np.log2,
            n_jobs=self.n_jobs
        )
        mi = mutual_information(
            expr,
            10,
            logtype=np.log2,
            n_jobs=self.n_jobs
        )

        with np.errstate(divide='ignore', invalid='ignore'):
            calc_dist = 1 - mi / (entropy[:, None] + entropy[None, :] - mi)
            calc_dist[np.isnan(calc_dist)] = 0.

        i_dist, mi_from_dist = information_distance(
            expr,
            BINS,
            logtype=np.log2,
            return_information=True,
            n_jobs=self.n_jobs
        )

        self.assertTrue(np.all(i_dist >= 0))
        npt.assert_almost_equal(mi, mi_from_dist)
        npt.assert_almost_equal(i_dist, calc_dist)
        npt.assert_array_almost_equal(
            np.diagonal(i_dist),
            np.zeros_like(np.diagonal(i_dist))
        )


class TestProgramMetricsParallel(TestProgramMetrics):

    n_jobs = 2


class TestProgram(unittest.TestCase):

    def test_find_program(self):

        adata = EXPRESSION_ADATA.copy()

        program_select(
            adata,
            verbose=True,
            filter_to_hvg=False,
            normalize=False,
            metric='information'
        )

        for k in ADATA_UNS_PROGRAM_KEYS:
            self.assertIn(k, adata.uns['programs'].keys())

        self.assertListEqual(
            PROGRAMS,
            adata.var['programs'].tolist()
        )

    def test_find_prog_euclid(self):

        adata = EXPRESSION_ADATA.copy()

        program_select(
            adata,
            verbose=True,
            filter_to_hvg=False,
            normalize=False,
            metric='euclidean'
        )

        self.assertListEqual(
            PROGRAMS,
            adata.var['programs'].tolist()
        )


class TestAssignGenesBasedOnTime(unittest.TestCase):

    def test_assign_programs(self):

        adata = EXPRESSION_ADATA.copy()

        program_select(adata, filter_to_hvg=False)
        adata.obs['program_0_time'] = TIMES_0
        adata.obs['program_1_time'] = TIMES_1

        new_program_labels = assign_genes_to_programs(
            adata,
            normalize=False
        )

        self.assertListEqual(new_program_labels.tolist(), PROGRAMS_ASSIGNED)

        new_program_labels, mi_mi = assign_genes_to_programs(
            adata,
            use_existing_programs=False,
            verbose=True,
            return_mi=True,
            normalize=False
        )

        self.assertListEqual(new_program_labels.tolist(), PROGRAMS_ASSIGNED)

    def test_assign_programs_nan(self):

        adata = EXPRESSION_ADATA.copy()
        t0 = TIMES_0.copy().astype(float)
        t0[1] = np.nan
        t1 = TIMES_1.copy().astype(float)
        t1[0] = np.nan

        program_select(adata, filter_to_hvg=False)
        adata.obs['program_0_time'] = t0
        adata.obs['program_1_time'] = t1

        new_program_labels = assign_genes_to_programs(
            adata,
            normalize=False,
            verbose=True
        )

        self.assertListEqual(new_program_labels.tolist(), PROGRAMS_ASSIGNED)

        new_program_labels, mi_mi = assign_genes_to_programs(
            adata,
            use_existing_programs=False,
            verbose=True,
            return_mi=True,
            normalize=False
        )

        self.assertListEqual(new_program_labels.tolist(), PROGRAMS_ASSIGNED)


class TestProgramPlot(unittest.TestCase):

    def test_plot_program(self):

        adata = EXPRESSION_ADATA.copy()

        program_select(
            adata,
            verbose=True,
            filter_to_hvg=False,
            normalize=False
        )

        f, a = programs_summary(adata)

        self.assertEqual(
            len(a),
            5
        )

import numpy as np
import scipy.sparse as sps
import sklearn.metrics
import anndata as ad
import unittest

from inferelator_velocity.utils.graph import local_optimal_knn

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

import unittest

import scipy.sparse as sps
import numpy as np
import numpy.testing as npt

from inferelator_velocity.utils.graph import (
    get_shortest_paths,
    get_total_path,
    local_optimal_knn
)
from ._stubs import (
    KNN_N as N,
    DIST,
    CONN,
    KNN
)

COL = {
    'a': ('b', 0, 0.5),
    'b': ('c', 0.5, 1)
}

COC = {
    'a': ('b', 0, 0.5),
    'b': ('c', 0.5, 1),
    'c': ('a', 1, 2)
}

C = ['a', 'b', 'c']


class TestOptKNN(unittest.TestCase):

    def setUp(self) -> None:
        self.dist = DIST.copy()
        self.conn = CONN.copy()

        self.sdist = sps.csr_matrix(DIST)
        self.sconn = sps.csr_matrix(CONN)

    def test_largest(self):

        correct = DIST.copy()
        correct[correct < 5] = 0.
        scorrect = sps.csr_matrix(correct)

        c = local_optimal_knn(self.dist, [5] * N, keep='largest')
        npt.assert_array_almost_equal(c, correct)
        npt.assert_array_almost_equal(self.dist, correct)
        self.assertEqual(id(c), id(self.dist))

        cs = local_optimal_knn(self.sdist, [5] * N, keep='largest')

        npt.assert_array_almost_equal(cs.data, scorrect.data)
        npt.assert_array_almost_equal(cs.indices, scorrect.indices)
        npt.assert_array_almost_equal(cs.indptr, scorrect.indptr)
        self.assertEqual(id(cs), id(self.sdist))
        self.assertTrue(sps.isspmatrix_csr(cs))

    def test_smallest(self):

        correct = DIST.copy()
        correct[correct > 5] = 0.
        scorrect = sps.csr_matrix(correct)

        c = local_optimal_knn(self.dist, [5] * N, keep='smallest')
        npt.assert_array_almost_equal(c, correct)
        npt.assert_array_almost_equal(self.dist, correct)
        self.assertEqual(id(c), id(self.dist))

        cs = local_optimal_knn(self.sdist, [5] * N, keep='smallest')

        npt.assert_array_almost_equal(cs.data, scorrect.data)
        npt.assert_array_almost_equal(cs.indices, scorrect.indices)
        npt.assert_array_almost_equal(cs.indptr, scorrect.indptr)
        self.assertEqual(id(cs), id(self.sdist))

        self.assertTrue(sps.isspmatrix_csr(cs))

    def test_k_0(self):

        correct = np.zeros_like(DIST)
        scorrect = sps.csr_matrix(correct)

        c = local_optimal_knn(self.dist, [0] * N)

        npt.assert_array_almost_equal(c, correct)
        npt.assert_array_almost_equal(self.dist, correct)
        self.assertEqual(id(c), id(self.dist))

        cs = local_optimal_knn(self.sdist, [0] * N)

        npt.assert_array_almost_equal(cs.data, scorrect.data)
        npt.assert_array_almost_equal(cs.indices, scorrect.indices)
        npt.assert_array_almost_equal(cs.indptr, scorrect.indptr)
        self.assertEqual(id(cs), id(self.sdist))
        self.assertTrue(sps.isspmatrix_csr(cs))

    def test_k_too_big(self):

        correct = DIST.copy()
        scorrect = sps.csr_matrix(correct)

        c = local_optimal_knn(self.dist, [1000] * N)

        npt.assert_array_almost_equal(c, correct)
        npt.assert_array_almost_equal(self.dist, correct)
        self.assertEqual(id(c), id(self.dist))

        cs = local_optimal_knn(self.sdist, [1000] * N)

        npt.assert_array_almost_equal(cs.data, scorrect.data)
        npt.assert_array_almost_equal(cs.indices, scorrect.indices)
        npt.assert_array_almost_equal(cs.indptr, scorrect.indptr)
        self.assertEqual(id(cs), id(self.sdist))
        self.assertTrue(sps.isspmatrix_csr(cs))


class TestGetShortestPath(unittest.TestCase):

    def test_shortest_path_easy(self):

        p1 = np.arange(6).tolist()
        p2 = np.arange(5, N).tolist()
        p3 = [0, 9]

        paths = get_shortest_paths(KNN, [0, 5, N-1])

        self.assertEqual(paths.shape, (3, 3))

        self.assertListEqual(paths[0, 0], [0])
        self.assertListEqual(paths[0, 1], p1)
        self.assertListEqual(paths[0, 2], p3)

        self.assertListEqual(paths[1, 0], p1[::-1])
        self.assertListEqual(paths[1, 1], [5])
        self.assertListEqual(paths[1, 2], p2)

        self.assertListEqual(paths[2, 0], p3[::-1])
        self.assertListEqual(paths[2, 1], p2[::-1])
        self.assertListEqual(paths[2, 2], [9])

    def test_total_path(self):

        paths = get_shortest_paths(KNN, [0, 5, N-1])
        tp, tpc = get_total_path(paths, COL, C)

        self.assertListEqual(tp, np.arange(N).tolist())

        for i, c in zip([0, 5, N - 1], C):
            self.assertEqual(tp[tpc[c]], i)

        tp, tpc = get_total_path(paths, COC, C)

        self.assertListEqual(tp, np.arange(N).tolist() + [0])

        for i, c in zip([0, 5, N - 1], C):
            self.assertEqual(tp[tpc[c]], i)

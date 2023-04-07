import unittest
import numpy as np
import numpy.testing as npt
from scipy.stats import spearmanr

from inferelator_velocity.metrics.circcorrcoef import (
    _rank_vector,
    _radian_rank_vector,
    _rank_circular_array,
    circular_rank_correlation
)


class TestCircCorrCoef(unittest.TestCase):

    v1 = np.arange(100)
    v2 = np.concatenate((
        np.arange(50),
        np.arange(50)[::-1]
    ))
    v3 = np.concatenate((
        np.arange(50, 100),
        np.arange(0, 50)
    ))
    v4 = np.arange(100)[::-1]

    n_jobs = 1

    def setUp(self) -> None:

        self.arr = np.hstack((
            self.v1.reshape(-1, 1),
            self.v2.reshape(-1, 1),
            self.v3.reshape(-1, 1),
            self.v4.reshape(-1, 1)
        )).astype(float)

        return super().setUp()

    def test_vector_rank(self):

        npt.assert_array_equal(
            _rank_vector(self.v1),
            self.v1 + 1
        )

        npt.assert_array_equal(
            _rank_vector(self.v2),
            np.concatenate((
                np.arange(1, 101, 2),
                np.arange(1, 101, 2)[::-1]
            )).astype(float) + 0.5
        )

    def test_vector_radian(self):

        npt.assert_array_almost_equal(
            _radian_rank_vector(self.v1),
            ((self.v1 + 1) / 100) * 2 * np.pi
        )

    def test_array_radian(self):

        npt.assert_array_almost_equal(
            _rank_circular_array(self.arr, n_jobs=self.n_jobs),
            np.hstack((
                _radian_rank_vector(self.v1).reshape(-1, 1),
                _radian_rank_vector(self.v2).reshape(-1, 1),
                _radian_rank_vector(self.v3).reshape(-1, 1),
                _radian_rank_vector(self.v4).reshape(-1, 1)
            ))
        )

    def test_array_circcorr(self):

        npt.assert_array_almost_equal(
            circular_rank_correlation(self.arr[:, [0, 3]], n_jobs=self.n_jobs),
            np.array([
                [1, -1],
                [-1, 1]
            ])
        )

        npt.assert_array_almost_equal(
            circular_rank_correlation(self.arr, n_jobs=self.n_jobs),
            np.array([
                [1, 0, -1, -1],
                [0, 1, 0, 0],
                [-1, 0, 1, 1],
                [-1, 0, 1, 1]
            ])
        )

        # Double check that spearman's not doing the same
        with self.assertRaises(AssertionError):
            npt.assert_array_almost_equal(
                spearmanr(self.arr)[0],
                np.array([
                    [1, 0, -1, -1],
                    [0, 1, 0, 0],
                    [-1, 0, 1, 1],
                    [-1, 0, 1, 1]
                ])
            )


class TestCircCorrCoefParallel(TestCircCorrCoef):

    n_jobs = 2

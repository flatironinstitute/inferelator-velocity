import unittest

import numpy as np
import numpy.testing as npt
import scipy.sparse as sps

from inferelator_velocity.utils.math import (
    scalar_projection,
    mean_squared_error,
    variance,
    coefficient_of_variation
)


N = 1000

DATA_SEED = np.random.default_rng(1010101).random(N)
DATA_SEED[0] = 0
DATA_SEED[-1] = 1

DATA_SPACE = [0.5, -1, 10, 100]

DATA = np.vstack((DATA_SEED * DATA_SPACE[0],
                  DATA_SEED * DATA_SPACE[1],
                  DATA_SEED * DATA_SPACE[2],
                  DATA_SEED * DATA_SPACE[3])).T


class TestScalarProjections(unittest.TestCase):

    def test_scalar_no_weight(self):

        sl = np.sqrt(np.sum(np.square(DATA), axis=1))

        sp = scalar_projection(
            DATA,
            center_point=0,
            off_point=N - 1,
            normalize=False
        )

        npt.assert_array_almost_equal(sp, sl)

    def test_scalar_no_weight_normalized(self):

        sl = np.sqrt(np.sum(np.square(DATA), axis=1))
        sl = sl / sl.max()

        sp = scalar_projection(
            DATA,
            center_point=0,
            off_point=N - 1,
            normalize=True
        )

        npt.assert_array_almost_equal(sp, sl)

    def test_scalar_weighted(self):

        sl = np.sqrt(np.sum(np.square(DATA[:, 0:2]), axis=1))
        sp = scalar_projection(
            DATA,
            center_point=0,
            off_point=N - 1,
            normalize=False,
            weights=np.array([1, 1, 0, 0])
        )

        npt.assert_array_almost_equal(sp, sl)

        sl = np.sqrt(np.square(DATA[:, 0]))
        sp = scalar_projection(
            DATA,
            center_point=0,
            off_point=N - 1,
            normalize=False,
            weights=np.array([1, 0, 0, 0])
        )

        npt.assert_array_almost_equal(sp, sl)


class TestMSE(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        rng = np.random.default_rng(12345)

        cls.X = rng.random((100, 20))
        cls.Y = rng.random((100, 20))
        cls.Z = np.mean((cls.X - cls.Y) ** 2)
        cls.Z_row = np.mean((cls.X - cls.Y) ** 2, axis=1)
        cls.Z_noY = np.mean(cls.X ** 2)
        cls.Z_noY_row = np.mean(cls.X ** 2, axis=1)

        return super().setUpClass()

    def test_dense_dense(self):

        npt.assert_array_almost_equal(
            mean_squared_error(self.X, self.Y),
            self.Z
        )

        npt.assert_array_almost_equal(
            mean_squared_error(self.X, self.Y, by_row=True),
            self.Z_row
        )

        npt.assert_array_almost_equal(
            mean_squared_error(self.X, None),
            self.Z_noY
        )

        npt.assert_array_almost_equal(
            mean_squared_error(self.X, None, by_row=True),
            self.Z_noY_row
        )

    def test_sparse_dense(self):

        X = sps.csr_matrix(self.X)

        npt.assert_array_almost_equal(
            mean_squared_error(X, self.Y),
            self.Z
        )

        npt.assert_array_almost_equal(
            mean_squared_error(X, self.Y, by_row=True),
            self.Z_row
        )

        npt.assert_array_almost_equal(
            mean_squared_error(X, None),
            self.Z_noY
        )

        npt.assert_array_almost_equal(
            mean_squared_error(X, None, by_row=True),
            self.Z_noY_row
        )

    def test_sparse_sparse(self):

        X = sps.csr_matrix(self.X)
        Y = sps.csr_matrix(self.Y)

        npt.assert_array_almost_equal(
            mean_squared_error(X, Y),
            self.Z
        )

        npt.assert_array_almost_equal(
            mean_squared_error(X, Y, by_row=True),
            self.Z_row
        )


class TestVariance(unittest.TestCase):

    def setUp(self) -> None:

        self.arr = np.random.rand(50, 5)
        self.arr[self.arr < 0.5] = 0

        self.sarr = sps.csr_matrix(self.arr)
        return super().setUp()

    def test_flattened(self):

        npt.assert_almost_equal(
            variance(self.arr),
            variance(self.sarr)
        )

    def test_axis_0(self):

        npt.assert_almost_equal(
            variance(self.arr, axis=0),
            variance(self.sarr, axis=0)
        )

    def test_axis_1(self):

        npt.assert_almost_equal(
            variance(self.arr, axis=1),
            variance(self.sarr, axis=1)
        )


class TestCV(unittest.TestCase):

    def setUp(self) -> None:

        self.arr = np.random.rand(50, 5)
        self.arr[self.arr < 0.5] = 0

        self.sarr = sps.csr_matrix(self.arr)
        return super().setUp()

    def test_flattened(self):

        npt.assert_almost_equal(
            coefficient_of_variation(self.arr),
            coefficient_of_variation(self.sarr)
        )

    def test_axis_0(self):

        npt.assert_almost_equal(
            coefficient_of_variation(self.arr, axis=0),
            coefficient_of_variation(self.sarr, axis=0)
        )

    def test_axis_1(self):

        npt.assert_almost_equal(
            coefficient_of_variation(self.arr, axis=1),
            coefficient_of_variation(self.sarr, axis=1)
        )

import unittest
import contextlib

import numpy as np
import numpy.testing as npt
import scipy.sparse as sps

from inferelator_velocity.utils.math import (
    scalar_projection,
    variance,
    coefficient_of_variation,
    pairwise_metric
)

from inferelator_velocity.utils.mcv import mcv_mse
try:
    from inferelator_velocity.utils.mcv import (
        _mse_rowwise
    )
    NUMBA=True

except ImportError:
    NUMBA=False


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


@unittest.skipIf(not NUMBA, "NUMBA not installed")
class TestMSENumba(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:

        rng = np.random.default_rng(12345)

        cls.X = rng.random((100, 20))
        cls.PC = rng.random((100, 10))
        cls.ROTATION = rng.random((10, 20))
        cls.Y = cls.PC @ cls.ROTATION
        cls.Z = np.mean((cls.X - cls.Y) ** 2)
        cls.Z_row = np.mean((cls.X - cls.Y) ** 2, axis=1)
        cls.Z_noY = np.mean(cls.X ** 2)
        cls.Z_noY_row = np.mean(cls.X ** 2, axis=1)

    def test_mse_rowwise(self):

        npt.assert_array_almost_equal(
            mcv_mse(self.X, self.PC, self.ROTATION),
            self.Z
        )

        npt.assert_array_almost_equal(
            mcv_mse(sps.csr_array(self.X), self.PC, self.ROTATION, by_row=True),
            self.Z_row
        )

        npt.assert_array_almost_equal(
            mcv_mse(sps.csr_array(self.X), self.PC, self.ROTATION),
            self.Z
        )


class TestMSE(unittest.TestCase):

    metric = 'mse'
    none_context = contextlib.nullcontext()

    @classmethod
    def setUpClass(cls) -> None:

        rng = np.random.default_rng(12345)

        cls.X = rng.random((100, 20))
        cls.Y = rng.random((100, 20))
        cls.Z = np.mean((cls.X - cls.Y) ** 2)
        cls.Z_row = np.mean((cls.X - cls.Y) ** 2, axis=1)
        cls.Z_noY = np.mean(cls.X ** 2)
        cls.Z_noY_row = np.mean(cls.X ** 2, axis=1)

    def test_dense_dense(self):

        npt.assert_array_almost_equal(
            pairwise_metric(self.X, self.Y, metric=self.metric),
            self.Z
        )

        npt.assert_array_almost_equal(
            pairwise_metric(self.X, self.Y, by_row=True, metric=self.metric),
            self.Z_row
        )

        with self.none_context:
            npt.assert_array_almost_equal(
                pairwise_metric(self.X, None, metric=self.metric),
                self.Z_noY
            )

            npt.assert_array_almost_equal(
                pairwise_metric(self.X, None, by_row=True, metric=self.metric),
                self.Z_noY_row
            )

    def test_sparse_dense(self):

        X = sps.csr_matrix(self.X)

        npt.assert_array_almost_equal(
            pairwise_metric(X, self.Y, metric=self.metric),
            self.Z
        )

        npt.assert_array_almost_equal(
            pairwise_metric(X, self.Y, by_row=True, metric=self.metric),
            self.Z_row
        )

        with self.none_context:
            npt.assert_array_almost_equal(
                pairwise_metric(X, None, metric=self.metric),
                self.Z_noY
            )

            npt.assert_array_almost_equal(
                pairwise_metric(X, None, by_row=True, metric=self.metric),
                self.Z_noY_row
            )

    def test_sparse_sparse(self):

        X = sps.csr_matrix(self.X)
        Y = sps.csr_matrix(self.Y)

        npt.assert_array_almost_equal(
            pairwise_metric(X, Y, metric=self.metric),
            self.Z
        )

        npt.assert_array_almost_equal(
            pairwise_metric(X, Y, by_row=True, metric=self.metric),
            self.Z_row
        )


class TestMAE(TestMSE):

    metric = 'mae'

    @classmethod
    def setUpClass(cls) -> None:

        rng = np.random.default_rng(12345)

        cls.X = rng.random((100, 20))
        cls.Y = rng.random((100, 20))

        cls.Z = np.mean((cls.X - cls.Y))
        cls.Z_row = np.mean((cls.X - cls.Y), axis=1)
        cls.Z_noY = np.mean(cls.X)
        cls.Z_noY_row = np.mean(cls.X, axis=1)


class TestLogLoss(TestMSE):

    metric = 'log_loss'

    @classmethod
    def setUpClass(cls) -> None:

        rng = np.random.default_rng(12345)

        cls.X = rng.choice([0, 1], (100, 20))
        cls.Y = rng.random((100, 20))

        cls.Z = cls.X * np.log(cls.Y)
        cls.Z += (1 - cls.X) * np.log(1 - cls.Y)
        cls.Z *= -1

        cls.Z_row = np.mean(cls.Z, axis=1)
        cls.Z = np.mean(cls.Z_row)

        cls.Z_noY = None
        cls.Z_noY_row = None

    def setUp(self) -> None:
        self.none_context = self.assertRaises(ValueError)
        return super().setUp()


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

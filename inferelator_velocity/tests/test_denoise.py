import unittest
import numpy as np
import numpy.testing as npt
import scipy.sparse as sps

from ._stubs import (
    EXPR_KNN,
    EXPRESSION_ADATA
)

from inferelator_velocity.denoise_data import (
    _dist_to_row_stochastic,
    denoise
)

DENOISE_EXPR = np.dot(
    _dist_to_row_stochastic(EXPR_KNN).A,
    EXPRESSION_ADATA.X
)


class TestDenoise(unittest.TestCase):

    threshold = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.expect = DENOISE_EXPR.copy()

        if cls.threshold is not None:
            cls.expect[cls.expect < cls.threshold] = 0

        return super().setUpClass()

    def setUp(self) -> None:
        self.data = EXPRESSION_ADATA.copy()
        self.data.X = self.data.X.astype(np.float32)
        self.data.obsp['graph'] = EXPR_KNN

        return super().setUp()

    def test_errors(self):

        with self.assertRaises(RuntimeError):
            denoise(
                self.data,
                chunk_size=500,
                output_layer='abc',
                zero_threshold=self.threshold
            )

        self.data.X = self.data.X.astype(int)
        with self.assertRaises(RuntimeError):
            denoise(
                self.data,
                chunk_size=500,
                graph_key='graph',
                output_layer='abc',
                zero_threshold=self.threshold
            )

    def test_denoise_dense_to_dense(self):

        denoise(
            self.data,
            chunk_size=500,
            graph_key='graph',
            output_layer='abc',
            zero_threshold=self.threshold
        )

        npt.assert_almost_equal(
            self.expect,
            self.data.layers['abc'],
            decimal=2
        )

    def test_denoise_sparse_to_dense(self):

        self.data.X = sps.csr_matrix(self.data.X)

        denoise(
            self.data,
            chunk_size=500,
            graph_key='graph',
            output_layer='abc',
            dense=True,
            zero_threshold=self.threshold
        )

        npt.assert_almost_equal(
            self.expect,
            self.data.layers['abc'],
            decimal=2
        )

    def test_denoise_sparse_to_sparse(self):

        self.data.X = sps.csr_matrix(self.data.X)

        denoise(
            self.data,
            chunk_size=500,
            graph_key='graph',
            output_layer='abc',
            dense=False,
            zero_threshold=self.threshold
        )

        npt.assert_almost_equal(
            self.expect,
            self.data.layers['abc'].A,
            decimal=2
        )


class TestDenoiseThreshold(TestDenoise):

    threshold = 100

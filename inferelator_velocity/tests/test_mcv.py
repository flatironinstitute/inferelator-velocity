import unittest
import numpy as np
import scipy.sparse as sps

from inferelator_velocity.utils.mcv import mcv_pcs


rng = np.random.default_rng(100)
COUNTS = rng.negative_binomial(5, 0.5, (1000, 10))


class TestMCV(unittest.TestCase):

    def test_sparse_log(self):
        data = sps.csr_matrix(COUNTS)
        self.assertEqual(
            np.argmin(
                mcv_pcs(data, n=1, n_pcs=5)
            ),
            0
        )

    def test_sparse_log_scale(self):
        data = sps.csr_matrix(COUNTS)

        self.assertEqual(
            np.argmin(
                mcv_pcs(data, n=1, n_pcs=5, standardization_method='log_scale')
            ),
            0
        )

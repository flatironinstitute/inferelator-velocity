import unittest

import numpy as np
import numpy.testing as npt

from inferelator_velocity.utils.math import (get_centroids, scalar_projection)


N = 1000

DATA_SEED = np.random.default_rng(1010101).random(N)
DATA_SEED[0] = 0
DATA_SEED[-1] = 1

DATA_SPACE = [0.5, -1, 10, 100]

DATA = np.vstack((DATA_SEED * DATA_SPACE[0],
                  DATA_SEED * DATA_SPACE[1],
                  DATA_SEED * DATA_SPACE[2],
                  DATA_SEED * DATA_SPACE[3])).T


class TestVelocity(unittest.TestCase):

    def test_scalar_no_weight(self):

        sl = np.sqrt(np.sum(np.square(DATA), axis=1))

        sp = scalar_projection(DATA, center_point=0, off_point=N - 1, normalize=False)

        npt.assert_array_almost_equal(sp, sl)

    def test_scalar_no_weight_normalized(self):

        sl = np.sqrt(np.sum(np.square(DATA), axis=1))
        sl = sl / sl.max()

        sp = scalar_projection(DATA, center_point=0, off_point=N - 1, normalize=True)

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

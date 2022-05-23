import unittest

import numpy as np
import numpy.testing as npt

from inferelator_velocity.velocity import calc_velocity

N = 10

V_SLOPES = np.array([1, -1, 0, 1])
V_EXPRESSION = np.random.default_rng(222222).random((N, 4))
VELOCITY = np.multiply(V_EXPRESSION, V_SLOPES[None, :])
V_EXPRESSION[:, 3] = 0

T_SLOPES = np.array([0.5, -0.5, 0, 1])

TIME = np.arange(N)
T_EXPRESSION = np.multiply(TIME[:, None], T_SLOPES[None, :])
T_EXPRESSION = np.add(T_EXPRESSION, np.array([0, 10, 1, 5])[None, :])


class TestVelocity(unittest.TestCase):

    def test_calc_velocity(self):

        correct_velo = np.tile(T_SLOPES[:, None], N).T
        velo = calc_velocity(T_EXPRESSION, TIME, np.ones((N, N)), N,
                             wrap_time=None)

        npt.assert_array_almost_equal(correct_velo, velo)

        velo_wrap = calc_velocity(T_EXPRESSION, TIME, np.ones((N, N)), N,
                                  wrap_time=0)

        npt.assert_array_almost_equal(correct_velo, velo_wrap)

    def test_calc_velocity_nan(self):

        correct_velo = np.tile(T_SLOPES[:, None], N).T
        correct_velo[0, :] = np.nan

        t = TIME.copy().astype(float)
        t[0] = np.nan

        velo = calc_velocity(T_EXPRESSION, t, np.ones((N, N)), N,
                             wrap_time=None)

        npt.assert_array_almost_equal(correct_velo, velo)

        velo_wrap = calc_velocity(T_EXPRESSION, t, np.ones((N, N)), N,
                                  wrap_time=0)

        npt.assert_array_almost_equal(correct_velo, velo_wrap)

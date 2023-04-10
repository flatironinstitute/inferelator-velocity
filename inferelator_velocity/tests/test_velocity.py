import unittest

import numpy as np
import numpy.testing as npt
from scipy.sparse import csr_matrix

from inferelator_velocity.velocity import calc_velocity, _calc_local_velocity

N = 10

V_SLOPES = np.array([1, -1, 0, 1])
V_EXPRESSION = np.random.default_rng(222222).random((N, 4))
VELOCITY = np.multiply(V_EXPRESSION, V_SLOPES[None, :])
V_EXPRESSION[:, 3] = 0

T_SLOPES = np.array([0.5, -0.5, 0, 1])

TIME = np.arange(N)
T_EXPRESSION = np.multiply(TIME[:, None], T_SLOPES[None, :])
T_EXPRESSION = np.add(T_EXPRESSION, np.array([0, 10, 1, 5])[None, :])

KNN = np.diag(np.ones(N - 1), -1) + np.diag(np.ones(N - 1), 1)
KNN[0, N - 1] = 1
KNN[N - 1, 0] = 1


class TestVelocity(unittest.TestCase):

    def setUp(self) -> None:
        self.expr = T_EXPRESSION.copy()
        self.ones_graph = np.ones((N, N))
        self.knn = KNN.copy()

    def test_calc_velocity(self):

        correct_velo = np.tile(T_SLOPES[:, None], N).T
        velo = calc_velocity(
            self.expr,
            TIME,
            self.ones_graph,
            wrap_time=None
        )

        npt.assert_array_almost_equal(correct_velo, velo)

        velo_wrap = calc_velocity(
            self.expr,
            TIME,
            self.ones_graph,
            wrap_time=0
        )

        npt.assert_array_almost_equal(correct_velo, velo_wrap)

    def test_calc_velocity_nan(self):

        correct_velo = np.tile(T_SLOPES[:, None], N).T
        correct_velo[0, :] = np.nan

        t = TIME.copy().astype(float)
        t[0] = np.nan

        velo = calc_velocity(
            self.expr,
            t,
            self.ones_graph,
            wrap_time=None
        )

        npt.assert_array_almost_equal(correct_velo, velo)

        velo_wrap = calc_velocity(
            self.expr,
            t,
            self.ones_graph,
            wrap_time=0
        )

        npt.assert_array_almost_equal(correct_velo, velo_wrap)

    def test_calc_velocity_wraps(self):

        correct_velo = np.tile(T_SLOPES[:, None], N).T

        npt.assert_array_almost_equal(
            correct_velo,
            calc_velocity(
                self.expr,
                TIME,
                self.knn,
                wrap_time=None
            )
        )

        npt.assert_array_almost_equal(
            correct_velo,
            calc_velocity(
                self.expr,
                TIME,
                self.knn,
                wrap_time=200
            )
        )

        npt.assert_array_almost_equal(
            correct_velo,
            calc_velocity(
                self.expr,
                TIME,
                self.knn,
                wrap_time=0
            )
        )

        # Correct the first and last velocities
        # Which change cause of wrapping
        wrap_edge_correct = correct_velo.copy()
        wrap_edge_correct[0, :] *= -4
        wrap_edge_correct[-1, :] *= -4

        npt.assert_array_almost_equal(
            wrap_edge_correct,
            calc_velocity(self.expr, TIME, self.knn,
                          wrap_time=10)
        )

    def test_single_velocity(self):

        velo_0 = _calc_local_velocity(
            self.expr[0:5],
            TIME[0:5],
            2
        )

        npt.assert_array_almost_equal(velo_0.ravel(), T_SLOPES)

    def test_single_velocity_wrap(self):

        velo_0 = _calc_local_velocity(
            self.expr[0:6],
            np.hstack((TIME[7:], TIME[0:3])),
            2,
            wrap_time=N
        )

        npt.assert_array_almost_equal(velo_0.ravel(), T_SLOPES)

        velo_1 = _calc_local_velocity(
            self.expr[0:6],
            np.hstack((TIME[7:], TIME[0:3])),
            3,
            wrap_time=N
        )

        npt.assert_array_almost_equal(velo_1.ravel(), T_SLOPES)


class TestVelocitySparse(TestVelocity):

    def setUp(self) -> None:
        self.expr = csr_matrix(T_EXPRESSION)
        self.ones_graph = csr_matrix(np.ones((N, N)))
        self.knn = csr_matrix(KNN)

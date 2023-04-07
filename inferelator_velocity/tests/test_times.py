import unittest

import numpy as np

from inferelator_velocity.times import (
    program_times,
    calculate_times
)

N = 10

DIST = np.tile(np.arange(N), (N, 1)).astype(float)
CONN = 1 / (DIST + 1)

KNN = np.diag(np.ones(N - 1), -1) + np.diag(np.ones(N - 1), 1)
KNN[0, N - 1] = 1
KNN[N - 1, 0] = 1

COL = {
    'a': ('b', 0, 0.5),
    'b': ('c', 0.5, 1)
}

EXPR = np.dot(np.arange(N).reshape(-1, 1), np.arange(4).reshape(1, -1))
EXPR += (5 * np.random.default_rng(222222).random((N, 4))).astype(int)
LAB = np.array(['a'] * 3 + ['b'] * 3 + ['c'] * 4)


class TestTimeEsts(unittest.TestCase):

    def test_times(self):

        times = calculate_times(
            EXPR,
            LAB,
            COL,
            n_neighbors=4,
            verbose=True
        )

        self.assertListEqual(
            [0, 0.5, 1.],
            [times[v] for k, v in {'a': 2, 'b': 5, 'c': 9}.items()]
        )


class TestTimeFunctions(unittest.TestCase):

    def test_wrap_time(self):
        pass

import unittest

import numpy as np
import anndata as ad

from inferelator_velocity.times import (
    program_times,
    calculate_times
)

from inferelator_velocity.plotting.program_times import (
    program_time_summary
)

from inferelator_velocity.utils.keys import (
    PROG_NAMES_SUBKEY,
    PROGRAM_KEY
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


class TestProgramTimes(unittest.TestCase):

    def setUp(self) -> None:

        self.adata = ad.AnnData(
            EXPR
        )

        self.adata.obs['group'] = LAB
        self.adata.var[PROGRAM_KEY] = '0'
        self.adata.uns[PROGRAM_KEY] = {
            PROG_NAMES_SUBKEY: ['0']
        }

        return super().setUp()

    def test_program_times(self):

        program_times(
            self.adata,
            {'0': 'group'},
            {'0': COL}
        )

        times = self.adata.obs['program_0_time']

        self.assertListEqual(
            [0, 0.5, 1.],
            [times[v] for k, v in {'a': 2, 'b': 5, 'c': 9}.items()]
        )

    def test_program_times_exceptions(self):

        with self.assertRaises(ValueError):

            program_times(
                self.adata,
                {'0': 'group'},
                {'0': COL},
                program_var_key='abiubiosu'
            )

        with self.assertRaises(ValueError):

            program_times(
                self.adata,
                {'1': 'group'},
                {'0': COL}
            )

        with self.assertRaises(ValueError):

            program_times(
                self.adata,
                {'0': 'group'},
                {'1': COL}
            )

        del self.adata.uns[PROGRAM_KEY]

        with self.assertRaises(RuntimeError):

            program_times(
                self.adata,
                {'0': 'group'},
                {'0': COL}
            )

    def test_program_time_plots(self):

        with self.assertRaises(RuntimeError):

            f, a = program_time_summary(
                self.adata,
                '0'
            )

        program_times(
            self.adata,
            {'0': 'group'},
            {'0': COL}
        )

        f, a = program_time_summary(
            self.adata,
            '0'
        )

        self.assertEqual(
            len(a),
            4
        )


class TestTimeFunctions(unittest.TestCase):

    def test_wrap_time(self):
        pass

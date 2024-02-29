import numpy as np
import anndata as ad
import scipy.sparse as sps

KNN_N = 10

DIST = np.tile(np.arange(KNN_N), (KNN_N, 1)).astype(float)
CONN = 1 / (DIST + 1)

KNN = np.diag(np.ones(KNN_N - 1), -1) + np.diag(np.ones(KNN_N - 1), 1)
KNN[0, KNN_N - 1] = 1
KNN[KNN_N - 1, 0] = 1

rng = np.random.default_rng(100)
COUNTS = rng.negative_binomial(5, 0.5, (1000, 10))

N = 1000
BINS = 10

EXPRESSION = np.zeros((N, 6), dtype=int)
EXPRESSION[:, 0] = (100 * np.random.default_rng(222222).random(N)).astype(int)
EXPRESSION[:, 1] = EXPRESSION[:, 0] * 1.75 - 0.5
EXPRESSION[:, 2] = EXPRESSION[:, 0] ** 2
EXPRESSION[:, 3] = 0
EXPRESSION[:, 4] = np.arange(N)
EXPRESSION[:, 5] = np.arange(N) * 2 + 10

K = 15
EXPR_KNN = sps.csr_matrix(
    (
        rng.uniform(low=0.1, high=2., size=K * N),
        (
            np.repeat(np.arange(N), K),
            np.concatenate(
                tuple(
                    rng.choice(np.arange(1000), size=(K, ), replace=False)
                    for _ in range(N)
                )
            )
        )
    ),
    dtype=np.float32
)

EXPRESSION_ADATA = ad.AnnData(EXPRESSION.astype(int))

ADATA_UNS_PROGRAM_KEYS = [
    'metric',
    'leiden_correlation',
    'metric_genes',
    'information_distance',
    'cluster_program_map',
    'n_comps',
    'n_programs',
    'program_names',
    'molecular_cv_loss'
]

PROGRAMS = ['0', '0', '0', '-1', '1', '1']
PROGRAMS_ASSIGNED = ['0', '0', '0', '0', '1', '1']

TIMES_0 = EXPRESSION[:, 0] / 100
TIMES_1 = np.arange(N).astype(float)

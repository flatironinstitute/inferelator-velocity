import numpy as np

from sklearn.utils import gen_even_slices
from joblib import Parallel, delayed, effective_n_jobs


def circular_rank_correlation(
    X,
    n_jobs=-1
):
    """
    Calculate a rank circular correlation metric between features
    by ranking values and normalizing ranks to [0, 2*pi]

    rho = E[sin(x - x_bar) * sin(y - y_bar)] /
    sqrt(var(sin(x - x_bar)) * var(sin(y - y_bar)))

    Jammalamadaka and Sarma (1988)

    :param X: Samples x Features [m x n] array
    :type X: np.ndarray, sp.spmatrix
    :param n_jobs: Number of cores,
        defaults to -1
    :type n_jobs: int, optional
    :return: Correlation matrix [n x n]
    :rtype: np.ndarray
    """

    if X.ndim != 2:
        raise ValueError(
            "X must be 2-dimensional: "
            f"{X.ndim}-dimensional {X.shape} passed"
        )

    radian_array = _rank_circular_array(
        X,
        n_jobs=n_jobs
    )

    n = radian_array.shape[1]

    if n_jobs != 1:
        slices = list(
            gen_even_slices(
                n,
                effective_n_jobs(n_jobs)
            )
        )
        views = Parallel(n_jobs=n_jobs)(
            delayed(_circcorrcoef_array)(
                radian_array,
                subset=i
            )
            for i in slices
        )

        corr = np.empty(
            (n, n),
            dtype=float
        )

        for i, c in zip(slices, views):
            corr[:, i] = c

        return corr

    else:
        return _circcorrcoef_array(radian_array)


def _circcorrcoef_array(
    X,
    subset=slice(None)
):

    X_beta = X[:, subset]

    n = X.shape[1]
    m = X_beta.shape[1]

    corr = np.empty(
        (n, m),
        dtype=float
    )

    # Calculate feature-wise correlations
    for i in range(n):
        for j in range(m):
            _xi = X[:, i]
            _xj = X_beta[:, j]

            # Check for constant vectors
            # and define corr as NaN if found
            if _xi.min() == _xi.max():
                corr[i, j] = np.nan
            elif _xj.min() == _xj.max():
                corr[i, j] = np.nan
            else:
                corr[i, j] = _circcorrcoef(
                    X[:, i],
                    X_beta[:, j]
                )

    return corr


def _rank_circular_array(
    X,
    n_jobs=-1
):

    n = X.shape[1]

    def _array_apply(x_sub):
        return np.apply_along_axis(
            _radian_rank_vector,
            0,
            x_sub
        )

    if n_jobs != 1:

        slices = list(
            gen_even_slices(
                n,
                effective_n_jobs(n_jobs)
            )
        )

        views = Parallel(n_jobs=n_jobs)(
            delayed(_array_apply)(
                X[:, i]
            )
            for i in slices
        )

        rad_array = np.empty(
            X.shape,
            dtype=float
        )

        for i, r in zip(slices, views):
            rad_array[:, i] = r

        return rad_array

    else:
        return _array_apply(X)


def _rank_vector(x):

    n = len(x)

    # Argsort-based rank
    _idx = x.argsort()
    rank = np.empty(n, dtype=float)
    rank[_idx] = np.arange(1, n + 1)

    # Find duplicates
    repeats, counts = np.unique(
        x,
        return_counts=True
    )

    # Shortcut if there are no duplicates
    if len(repeats) == n:
        return rank

    # Take the mean rank wherever there are duplicates
    for v in repeats[counts > 1]:
        _ridx = x == v
        rank[_ridx] = np.mean(rank[_ridx])

    return rank


def _radian_rank_vector(x):

    x = _rank_vector(x).astype(float)
    x = x / x.max()

    return x * 2 * np.pi


def _circcorrcoef(x, y):

    _x = np.sin(x - np.mean(x))
    _y = np.sin(y - np.mean(y))

    _e = np.mean(_x * _y)
    _v = np.var(_x) * np.var(_y)

    return _e / np.sqrt(_v)

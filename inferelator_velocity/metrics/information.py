import itertools
import numpy as np

from sklearn.utils import gen_even_slices
from joblib import Parallel, delayed, effective_n_jobs
from inferelator.regression.mi import _make_table, _calc_mi


def information_distance(
    discrete_array,
    bins,
    n_jobs=-1,
    logtype=np.log,
    return_information=False
):
    """
    Calculate shannon information distance
    D(X, X) = 1 - MI(X, X) / H(X, X)
    Where MI(X, X) is mutual information between features of X
    H(X, X) is joint entropy between features of X

    :param discrete_array: Discrete integer array [Obs x Features]
        with values from 0 to `bins`
    :type discrete_array: np.ndarray [int]
    :param bins: Number of discrete bins in integer array
    :type bins: int
    :param n_jobs: Number of parallel jobs for joblib,
        -1 uses all cores
        None does not parallelize
    :type n_jobs: int, None
    :param logtype: Log function to use for information calculations,
        defaults to np.log
    :type logtype: func, optional
    :param return_information: Return mutual information in addition to
        distance, defaults to False
    :type return_information: bool, optional
    :return: Information distance D(X, X) array [Features x Features],
        and MI(X, X) array [Features x Features] if return_information is True
    :rtype: np.ndarray [float], np.ndarray [float] (optional)
    """

    # Calculate MI(X, X)
    mi_xx = mutual_information(
        discrete_array,
        bins,
        logtype=logtype,
        n_jobs=n_jobs
    )

    # Calculate H(X)
    h_x = _shannon_entropy(
        discrete_array,
        bins,
        logtype=logtype,
        n_jobs=n_jobs
    )

    # Calulate distance as 1 - MI(X, X) / H(X, X)
    # Where H(X, X) = H(X) + H(X) - MI(X, X)
    with np.errstate(divide='ignore', invalid='ignore'):
        h_xx = h_x[None, :] + h_x[:, None] - mi_xx
        d_xx = 1 - mi_xx / h_xx

    # Explicitly set distance where h_xx == 0
    # This is a rare edge case where there is no entropy in either gene
    # As MI(x, y) == H(x, y), we set the distance to 0 by convention
    d_xx[h_xx == 0] = 0.

    # Trim floats to 0 based on machine precision
    # Might need a looser tol; there's a lot of float ops here
    d_xx[np.abs(d_xx) <= (bins * np.spacing(bins))] = 0.

    # Return distance or distance & MI
    if return_information:
        return d_xx, mi_xx
    else:
        return d_xx


def mutual_information(
    x,
    bins,
    y=None,
    n_jobs=-1,
    logtype=np.log
):
    """
    Calculate mutual information between features of a discrete array

    :param x: Discrete integer array [Obs x Features]
        with values from 0 to `bins`
    :type x: np.ndarray [int]
    :param bins: Number of discrete bins in integer array
    :type bins: int
    :param y: Discrete integer array [Obs x Features]
        with values from 0 to `bins`
        Information will be calculated between y and x features
        if y is not None, if y is None between x and x features,
        defaults to None
    :param n_jobs: Number of parallel jobs for joblib,
        -1 uses all cores
        None does not parallelize
    :type n_jobs: int, None
    :param logtype: Log function to use for information calculations,
        defaults to np.log
    :type logtype: func, optional
    :return: Mutual information array [Features, Features]
    :rtype: np.ndarray [float]
    """

    m = x.shape[1]
    n = x.shape[1] if y is None else y.shape[1]

    if n_jobs != 1:
        slices = list(gen_even_slices(n, effective_n_jobs(n_jobs)))

        views = Parallel(n_jobs=n_jobs)(
            delayed(_mi_slice)(
                x,
                bins,
                y_slicer=i,
                y=y,
                logtype=logtype
            )
            for i in slices
        )

        mutual_info = np.empty((m, n), dtype=float)

        for i, r in zip(slices, views):
            mutual_info[:, i] = r

        return mutual_info

    else:
        return _mi_slice(
            x,
            bins,
            y=y,
            logtype=logtype
        )


def _shannon_entropy(
    discrete_array,
    bins,
    n_jobs=-1,
    logtype=np.log
):
    """
    Calculate shannon entropy for each feature in a discrete array

    :param discrete_array: Discrete integer array [Obs x Features]
        with values from 0 to `bins`
    :type discrete_array: np.ndarray [int]
    :param bins: Number of discrete bins in integer array
    :type bins: int
    :param n_jobs: Number of parallel jobs for joblib,
        -1 uses all cores
        None does not parallelize
    :type n_jobs: int, None
    :param logtype: Log function to use for information calculations,
        defaults to np.log
    :type logtype: func, optional
    :return: Shannon entropy array [Features, ]
    :rtype: np.ndarray [float]
    """

    m, n = discrete_array.shape

    if n_jobs != 1:
        slices = list(gen_even_slices(n, effective_n_jobs(n_jobs)))

        views = Parallel(n_jobs=n_jobs)(
            delayed(_entropy_slice)(
                discrete_array[:, i],
                bins,
                logtype=logtype
            )
            for i in slices
        )

        entropy = np.empty(n, dtype=float)

        for i, r in zip(slices, views):
            entropy[i] = r

        return entropy

    else:
        return _entropy_slice(
            discrete_array,
            bins,
            logtype=logtype
        )


def _entropy_slice(
    x,
    bins,
    logtype=np.log
):

    def _entropy(vec):
        px = np.bincount(vec, minlength=bins) / vec.size
        log_px = logtype(
            px,
            out=np.full_like(px, np.nan),
            where=px > 0
        )
        return -1 * np.nansum(px * log_px)

    return np.apply_along_axis(_entropy, 0, x)


def _mi_slice(
    x,
    bins,
    y_slicer=slice(None),
    y=None,
    logtype=np.log
):

    with np.errstate(divide='ignore', invalid='ignore'):

        if y is None:
            y = x[:, y_slicer]
        else:
            y = y[:, y_slicer]

        n1, n2 = x.shape[1], y.shape[1]

        mutual_info = np.empty((n1, n2), dtype=float)
        for i, j in itertools.product(range(n1), range(n2)):
            mutual_info[i, j] = _calc_mi(
                _make_table(
                    x[:, i],
                    y[:, j],
                    bins
                ),
                logtype=logtype
            )

        return mutual_info

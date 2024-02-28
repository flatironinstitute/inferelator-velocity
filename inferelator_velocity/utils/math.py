import numpy as np
import numba
import scipy.sparse as sps
from .misc import (
    make_vector_2D,
    is_csr
)

try:
    from sparse_dot_mkl import dot_product_mkl as dot

except ImportError as err:

    import warnings

    warnings.warn(
        "Unable to use MKL for sparse matrix math, "
        "defaulting to numpy/scipy matmul: "
        f"{str(err)}"
    )

    def dot(x, y, dense=False, cast=False, out=None):

        z = x @ y

        if dense and sps.issparse(z):
            z = z.A

        if out is not None:
            out[:] = z
            return out
        else:
            return z


def scalar_projection(
        data,
        center_point,
        off_point,
        normalize=True,
        weights=None,
        endpoint_distance=False
):
    """
    Scalar projection of data onto a line defined by two points

    :param data: Data
    :type data: np.ndarray
    :param center_point: Integer index of starting point for line
    :type center_point: int
    :param off_point: Integer index of ending point for line
    :type off_point: int
    :param normalize: Normalize distance between start and end of line to 1,
        defaults to True
    :type normalize: bool, optional
    :param weights: How much weight to put on each dimension, defaults to None
    :type weights: np.ndarray, optional
    :param endpoint_distance: Set distance for projection between the
        endpoints to zero
    :type endpoint_distance: bool
    :return: Scalar projection array
    :rtype: np.ndarray
    """

    data = data - data[center_point, :]

    if weights is not None:
        data = np.multiply(data, weights[None, :])

    vec = data[off_point, :]

    # Calculate scalar projection: (data (dot) vec) / ||vec||
    scalar_proj = np.dot(data, vec) / np.linalg.norm(vec)

    # Normalize projection such that vec is unit length
    if normalize:
        scalar_proj = scalar_proj / np.linalg.norm(vec)

    if endpoint_distance:
        _past_center = scalar_proj >= 0
        _past_offpoint = scalar_proj > scalar_proj[off_point]
        scalar_proj[_past_center & _past_offpoint] -= scalar_proj[off_point]
        scalar_proj[_past_center & ~_past_offpoint] = 0

    return scalar_proj


def get_centroids(comps, cluster_vector):
    return {
        k: _get_centroid(comps[cluster_vector == k, :])
        for k in np.unique(cluster_vector)
    }


def _get_centroid(comps):
    return np.sum(
        (comps - np.mean(comps, axis=0)[None, :]) ** 2,
        axis=1
    ).argmin()


def least_squares(x, y):
    """
    OLS wrapper that also calculates SE of slope estimate

    :param x: Design vector
    :type x: np.ndarray
    :param y: Response vector
    :type y: np.ndarray
    :return: OLS beta estimate, OLS beta SE estimate
    :rtype: float, float
    """

    if x.shape[0] == 0:
        return 0., 0.

    x = make_vector_2D(x)
    y = make_vector_2D(y)

    slope = np.linalg.lstsq(x, y, rcond=None)[0][0][0]

    return slope, _calc_se(x, y, slope)


def _calc_se(x, y, slope):

    if x.shape[0] == 0:
        return 0

    mse_x = np.sum(np.square(x - np.nanmean(x)))
    if mse_x == 0:
        return 0

    elif slope == 0:
        return np.mean(np.square(y - np.nanmean(y))) / mse_x

    else:
        mse_y = np.sum(np.square(y - np.dot(x, slope)))
        se_y = mse_y / (len(y) - 1)
        return se_y / mse_x


def _log_loss(x, y):

    if y is None:
        raise ValueError(
            "Cannot calculate log loss for only labels"
        )

    try:
        x = x.A
    except AttributeError:
        pass

    try:
        y = y.A
    except AttributeError:
        pass

    y = np.minimum(y, 1 - 1e-7)
    y = np.maximum(y, 1e-7)

    err = np.multiply(
        1 - x,
        np.log(1 - y)
    )
    err += np.multiply(
        x,
        np.log(y)
    )
    err = err.sum(axis=1)
    err *= -1
    return err


def _mse(x, y):

    if y is not None:
        ssr = x - y
    else:
        ssr = x.copy()

    if sps.issparse(ssr):
        ssr.data **= 2
    elif isinstance(ssr, np.matrix):
        ssr = ssr.A
        ssr **= 2
    else:
        ssr **= 2

    return ssr.sum(axis=1)


def _mae(x, y):

    if y is not None:
        ssr = x - y
    else:
        ssr = x

    return ssr.sum(axis=1)


def pairwise_metric(
    x,
    y,
    metric='mse',
    by_row=False,
    **kwargs
):
    """
    Pairwise metric between two arrays

    :param x: _description_
    :type x: _type_
    :param y: _description_
    :type y: _type_
    :param metric: _description_, defaults to 'mse'
    :type metric: str, optional
    :param by_row: _description_, defaults to False
    :type by_row: bool, optional
    :return: _description_
    :rtype: _type_
    """

    if metric == 'mse':
        metric = _mse
    elif metric == 'mae':
        metric = _mae
    elif metric == 'log_loss':
        metric = _log_loss

    loss = metric(
        x,
        y,
        **kwargs
    )

    try:
        loss = loss.A1
    except AttributeError:
        pass

    loss = loss / x.shape[1]

    if by_row:
        return loss

    else:
        return loss.sum() / loss.size


def variance(
    X,
    axis=None,
    ddof=0
):
    """
    Function to calculate variance for sparse or dense arrays

    :param X: Sparse or dense data array
    :type X: np.ndarray, sp.spmatrix
    :param axis: Across which axis (None flattens),
        defaults to None
    :type axis: int, optional
    :param ddof: Delta degrees of freedom,
        defaults to 0
    :type ddof: int, optional
    :return: Variance or vector of variances across an axis
    :rtype: numeric, np.ndarray
    """

    if sps.issparse(X):

        _n = np.prod(X.shape)

        if axis is None:
            _mean = X.mean()
            _nz = _n - X.size

            _var = np.sum(np.power(X.data - _mean, 2))
            _var += np.power(_mean, 2) * _nz

            return _var / (_n - ddof)

        else:
            _mean = X.mean(axis=axis).A1
            _nz = -1 * X.getnnz(axis=axis) + X.shape[axis]

            # Make a sparse mask of means over axis
            _mean_mask = sps.csr_matrix(
                ((np.ones(X.data.shape, dtype=float), X.indices, X.indptr))
            )

            if axis == 0:
                _mean_mask = _mean_mask.multiply(_mean[np.newaxis, :])
            else:
                _mean_mask = _mean_mask.multiply(_mean[:, np.newaxis])

            _var = (X - _mean_mask).power(2).sum(axis=axis).A1
            _var += np.power(_mean, 2) * _nz

        return _var / (X.shape[axis] - ddof)

    else:

        return np.var(X, axis=axis, ddof=ddof)


def coefficient_of_variation(
    X,
    axis=None,
    ddof=0
):
    """
    Calculate coefficient of variation

    :param X: Sparse or dense data array
    :type X: np.ndarray, sp.spmatrix
    :param axis: Across which axis (None flattens),
        defaults to None
    :type axis: int, optional
    :param ddof: Delta degrees of freedom,
        defaults to 0
    :type ddof: int, optional
    :return: CV or vector of CVs across an axis
    :rtype: numeric, np.ndarray
    """

    _var = variance(X, axis=axis, ddof=ddof)
    _mean = X.mean(axis=axis)

    try:
        _mean = _mean.A1
    except AttributeError:
        pass

    return np.divide(
        np.sqrt(_var),
        _mean,
        out=np.zeros_like(_var),
        where=_mean != 0
    )


@numba.njit(parallel=False)
def _csr_row_divide(
    a_data,
    a_indptr,
    row_vec
):

    n_row = row_vec.shape[0]

    for i in numba.prange(n_row):
        a_data[a_indptr[i]:a_indptr[i + 1]] /= row_vec[i]


@numba.njit(parallel=False)
def _mse_rowwise(
    a_data,
    a_indices,
    a_indptr,
    b_pcs,
    b_rotation
):

    n_row = b_pcs.shape[0]

    output = np.zeros(n_row, dtype=float)

    for i in numba.prange(n_row):

        _idx_a = a_indices[a_indptr[i]:a_indptr[i + 1]]
        _nnz_a = _idx_a.shape[0]

        row = b_pcs[i, :] @ b_rotation

        if _nnz_a == 0:
            continue

        else:

            row[_idx_a] -= a_data[a_indptr[i]:a_indptr[i + 1]]

        output[i] = np.mean(row ** 2)

    return output


@numba.njit(parallel=False)
def _sum_columns(data, indices, n_col):

    output = np.zeros(n_col, dtype=data.dtype)

    for i in numba.prange(data.shape[0]):
        output[indices[i]] += data[i]

    return output


@numba.njit(parallel=False)
def _sum_rows(data, indptr):

    output = np.zeros(indptr.shape[0] - 1, dtype=data.dtype)

    for i in numba.prange(output.shape[0]):
        output[i] = np.sum(data[indptr[i]:indptr[i + 1]])

    return output


def array_sum(array, axis=None):

    if not is_csr(array):
        _sums = array.sum(axis=axis)
        try:
            _sums = _sums.A1
        except AttributeError:
            pass
        return _sums

    if axis is None:
        return np.sum(array.data)

    elif axis == 0:
        return _sum_columns(
            array.data,
            array.indices,
            array.shape[1]
        )

    elif axis == 1:
        return _sum_rows(
            array.data,
            array.indptr
        )


def mcv_mse(x, pc, rotation, by_row=False, **metric_kwargs):

    if sps.issparse(x):

        y = _mse_rowwise(
            x.data,
            x.indices,
            x.indptr,
            np.ascontiguousarray(pc),
            np.ascontiguousarray(rotation, dtype=pc.dtype)
        )

        if by_row:
            return y

        else:
            return np.mean(y)

    else:

        return pairwise_metric(
            x,
            pc @ rotation,
            metric='mse',
            by_row=by_row,
            **metric_kwargs
        )

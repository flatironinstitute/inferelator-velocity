import numpy as _np
from scipy.sparse import issparse as _is_sparse
from tqdm import trange


def calc_velocity(
    expr,
    time_axis,
    neighbor_graph,
    wrap_time=None
):
    """
    Calculate local RNA velocity

    :param expr: Samples x Genes numpy with expression data
    :param time_axis: Samples, numpy array
    :param neighbor_graph: Samples x Samples numpy or scipy.sparse
        with nearest neighbor distances
    :return: Samples x Genes numpy with velocity data
    """

    return _np.vstack(
        [
            _calc_local_velocity(
                expr[n_idx, :].copy(),
                time_axis[n_idx].copy(),
                (n_idx == i).nonzero()[0][0],
                wrap_time=wrap_time
            )
            for i, n_idx in _find_local(expr, neighbor_graph)
        ]
    )


def _calc_local_velocity(
    expr,
    time_axis,
    center_index,
    wrap_time=None
):
    """
    Calculate a local rate of change

    :param expr: Samples x Genes numpy with expression data
    :param time_axis: Samples, numpy array
    :param center_index: The data point which we're calculating velocity for
    :return:
    """

    n, m = expr.shape

    if _np.isnan(time_axis[center_index]):
        return _np.full(m, _np.nan)

    time_axis = time_axis - time_axis[center_index]

    if wrap_time is not None:
        time_axis = _wrap_time(time_axis, wrap_time)

    # Calculate change in expression and time relative to the centerpoint
    y_diff = _np.subtract(expr, expr[center_index, :])

    _time_nan = _np.isnan(time_axis)

    # Remove any data where the time value is NaN
    if _np.sum(_time_nan) > 0:

        # If there's only 3 or fewer data points remaining, return NaN
        if _np.sum(~_time_nan) < 4:
            return _np.full(m, _np.nan)

        time_axis = time_axis[~_time_nan]
        y_diff = y_diff[~_time_nan, :]

    # Calculate (XT * X)^-1 * XT
    time_axis = time_axis.reshape(-1, 1)
    x_for_hat = _np.dot(
        _np.linalg.inv(
            _np.dot(
                time_axis.T,
                time_axis)
        ),
        time_axis.T
    )

    # Return the slope for each gene as velocity
    return _np.dot(x_for_hat, y_diff)


def _find_local(expr, neighbor_graph):
    """
    Find a return an expression matrix for a locally connected graph

    :param expr: Samples x Genes numpy or scipy with expression data
    :type expr: np.ndarray, sp.sparse.csr_matrix
    :param neighbor_graph: Samples x Samples connectivity matrix,
        where any non-zero value is connected.
    :return:
    """

    n, m = expr.shape
    neighbor_sparse = _is_sparse(neighbor_graph)

    for i in trange(n):

        n_slice = neighbor_graph[i, :]

        if neighbor_sparse:
            keepers = n_slice.indices
        else:
            keepers = _np.where(n_slice != 0)[0]

        if all(keepers != i):
            keepers = _np.insert(keepers, 0, i)

        yield i, keepers


def _wrap_time(times, wrap_time):
    """
    Wrap times by taking the minimum time difference
    with and without wrapping

    :param times: Centered time vector
    :type times: np.ndarray
    :param wrap_time: Time to wrap at
    :type wrap_time: numeric
    :return: Wrapped time vector
    :rtype: np.ndarray
    """

    times = _np.vstack((
        times + wrap_time,
        times,
        times - wrap_time
    ))

    times = times[
        _np.argmin(
            _np.abs(times),
            axis=0
        ),
        _np.arange(
            times.shape[1]
        )
    ]

    return times

import numpy as np
import scipy.sparse as sps
import numba


def mcv_mse_sparse(
    x,
    pc,
    rotation,
    by_row=False,
    axis=1,
    **metric_kwargs
):

    if axis == 1:
        func = _mse_rowwise
    elif axis == 0:
        func = _mse_columnwise
    else:
        raise ValueError

    y = func(
        x.data,
        x.indices,
        x.indptr,
        np.ascontiguousarray(pc),
        np.ascontiguousarray(rotation, dtype=pc.dtype),
        x.shape[1]
    )

    if by_row:
        return y

    else:
        return np.mean(y)


def sparse_sum(sparse_array, axis=None, squared=False):

    if not sps.issparse(sparse_array):
        raise ValueError("sparse_sum requires a sparse array")

    if axis is None:
        return np.sum(sparse_array.data)

    elif axis == 0:
        func = _sum_columns_squared if squared else _sum_columns
        return func(
            sparse_array.data,
            sparse_array.indices,
            sparse_array.shape[1]
        )

    elif axis == 1:
        func = _sum_rows_squared if squared else _sum_rows
        return func(
            sparse_array.data,
            sparse_array.indptr
        )


def sparse_normalize_total(sparse_array, target_sum=10_000):

    if not is_csr(sparse_array):
        raise ValueError("sparse_sum requires a sparse csr_array")

    if sparse_array.data.dtype == np.int32:
        dtype = np.float32
    elif sparse_array.data.dtype == np.int64:
        dtype = np.float64
    else:
        dtype = None

    if dtype is not None:
        float_view = sparse_array.data.view(dtype)
        float_view[:] = sparse_array.data
        sparse_array.data = float_view

    n_counts = sparse_sum(sparse_array, axis=1)

    if target_sum is None:
        target_sum = np.median(n_counts)

    _csr_row_divide(
        sparse_array.data,
        sparse_array.indptr,
        n_counts / target_sum
    )


@numba.njit(parallel=False)
def _mse_rowwise(
    a_data,
    a_indices,
    a_indptr,
    b_pcs,
    b_rotation,
    n_cols
):

    n_row = b_pcs.shape[0]

    output = np.zeros(n_row, dtype=float)

    for i in numba.prange(n_row):

        _idx_a = a_indices[a_indptr[i]:a_indptr[i + 1]]
        _nnz_a = _idx_a.shape[0]

        row = b_pcs[i, :] @ b_rotation

        if _nnz_a == 0:
            pass
        else:
            row[_idx_a] -= a_data[a_indptr[i]:a_indptr[i + 1]]

        output[i] = np.mean(row ** 2)

    return output


@numba.njit(parallel=False)
def _mse_columnwise(
    a_data,
    a_indices,
    a_indptr,
    b_pcs,
    b_rotation,
    n_cols
):

    n_row = b_pcs.shape[0]
    output = np.zeros(n_cols, dtype=float)

    for i in numba.prange(n_row):

        _idx_a = a_indices[a_indptr[i]:a_indptr[i + 1]]
        _nnz_a = _idx_a.shape[0]

        row = b_pcs[i, :] @ b_rotation

        if _nnz_a == 0:
            pass
        else:
            row[_idx_a] -= a_data[a_indptr[i]:a_indptr[i + 1]]

        output += row ** 2

    return output / n_row


@numba.njit(parallel=False)
def _sum_columns(data, indices, n_col):

    output = np.zeros(n_col, dtype=data.dtype)

    for i in numba.prange(data.shape[0]):
        output[indices[i]] += data[i]

    return output


@numba.njit(parallel=False)
def _sum_columns_squared(data, indices, n_col):

    output = np.zeros(n_col, dtype=data.dtype)

    for i in numba.prange(data.shape[0]):
        output[indices[i]] += data[i] ** 2

    return output


@numba.njit(parallel=False)
def _sum_rows(data, indptr):

    output = np.zeros(indptr.shape[0] - 1, dtype=data.dtype)

    for i in numba.prange(output.shape[0]):
        output[i] = np.sum(data[indptr[i]:indptr[i + 1]])

    return output


@numba.njit(parallel=False)
def _sum_rows_squared(data, indptr):

    output = np.zeros(indptr.shape[0] - 1, dtype=data.dtype)

    for i in numba.prange(output.shape[0]):
        output[i] = np.sum(data[indptr[i]:indptr[i + 1]] ** 2)

    return output


@numba.njit(parallel=False)
def _csr_row_divide(data, indptr, row_normalization_vec):

    for i in numba.prange(indptr.shape[0] - 1):
        data[indptr[i]:indptr[i + 1]] /= row_normalization_vec[i]


@numba.njit(parallel=False)
def _csr_column_divide(data, indices, column_normalization_vec):

    for i, idx in enumerate(indices):
        data[i] /= column_normalization_vec[idx]


def is_csr(x):
    return sps.isspmatrix_csr(x) or isinstance(x, sps.csr_array)


def is_csc(x):
    return sps.isspmatrix_csc(x) or isinstance(x, sps.csc_array)

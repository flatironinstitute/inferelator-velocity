import numpy as np
import anndata as ad
import scanpy as sc

import pandas.api.types as pat
import warnings


def order_dict_to_lists(order_dict):
    """
    Convert dict to two ordered lists
    Used to convert a dict {start_label: (stop_label, start_time, stop_time)}
    into ordered metadata
    """

    # Create a doubly-linked list
    _dll = {}

    for start, (end, _, _) in order_dict.items():

        if start in _dll:

            if _dll[start][1] is not None:
                raise ValueError(
                    f"Both {_dll[start][1]} and {end} follow {start}"
                )

            _dll[start] = (_dll[start][0], end)

        else:

            _dll[start] = (None, end)

        if end in _dll:

            if _dll[end][0] is not None:
                raise ValueError(
                    f"Both {_dll[end][0]} and {start} precede {end}"
                )

            _dll[end] = (start, _dll[end][1])

        else:

            _dll[end] = (start, None)

    _start = None

    for k in order_dict.keys():

        if _dll[k][0] is None and _start is not None:
            raise ValueError("Both {k} and {_start} lack predecessors")

        elif _dll[k][0] is None:
            _start = k

    if _start is None:
        _start = list(order_dict.keys())[0]

    _order = [_start]
    _time = [order_dict[_start][1], order_dict[_start][2]]
    _next = _dll[_start][1]

    while _next is not None and _next != _start:
        _order.append(_next)
        _next = _dll[_next][1]
        if _next in order_dict and _next != _start:
            _time.append(order_dict[_next][1])

    return _order, _time


def vprint(*args, verbose=False, **kwargs):
    if verbose:
        print(*args, **kwargs)


def check_matrix_sizes(a, b):

    if a is None or b is None:
        return True

    elif a.shape != b.shape:
        raise ValueError(
            f"Expression data {a.shape} ",
            f"and velocity data {b.shape} ",
            "are not the same size"
        )

    else:
        return True


def check_vector_sizes(a, b, axis=None):

    if a is None or b is None:
        return True

    elif axis is None and a.size != b.size:
        raise ValueError(
            "Vectors are not the same size: ",
            f"{a.size} & {b.size} do not align"
        )

    elif axis == 0 and a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Vector {b.shape} is not the same size as ",
            f"Matrix columns {a.shape}"
        )

    elif axis == 1 and a.shape[0] != b.shape[0]:
        raise ValueError(
            f"Vector {b.shape} is not the same size as ",
            f"Matrix rows {a.shape}"
        )

    else:
        return True


def make_vector_2D(vec):
    """
    Make sure a vector is a 2D column vector

    :param vec: Vector
    :type vec: np.ndarray
    :raises ValueError: Raises a ValueError if this is not a vector
    :return: Vector [N, 1]
    :rtype: np.ndarray
    """

    if vec.ndim > 2:
        raise ValueError(f"Vector shape {vec.shape} invalid")

    elif vec.ndim == 1:
        return vec.reshape(-1, 1)

    elif vec.ndim == 2 and vec.shape[1] != 1:
        raise ValueError(f"Vector shape {vec.shape} invalid")

    else:
        return vec


def get_bins(data, n_bins=None, centers=None, width=None):

    if n_bins is not None and centers is None and width is None:
        min_time, max_time = np.nanmin(data), np.nanmax(data)

        half_width = (max_time - min_time) / (2 * n_bins + 1)

        centers = np.linspace(
            min_time + half_width,
            max_time - half_width,
            num=n_bins
        )

    elif centers is not None and width is not None:
        half_width = width / 2

    else:
        raise ValueError("Pass number of bins, or pass both centers and width")

    return centers, half_width


def copy_count_layer(data, layer):

    lref = data.X if layer == 'X' else data.layers[layer]

    if not pat.is_integer_dtype(lref.dtype):
        warnings.warn(
            "Count data is expected, "
            f"but {lref.dtype} data has been passed. "
            "This data will be normalized and processed "
            "as count data. If it is not count data, "
            "these results will be nonsense."
        )

    d = ad.AnnData(lref.astype(float))
    d.layers['counts'] = lref.copy()
    d.var = data.var.copy()

    return d


def standardize_data(adata):

    sc.pp.normalize_per_cell(adata, min_counts=0)
    sc.pp.log1p(adata)

    return adata


def ragged_lists_to_array(
    lists,
    pad_value=-1
):
    """
    Pad ragged lists out to arrays

    :param lists: List of lists
    :type lists: list(list)
    :param pad_value: Value to pad with,
        defaults to -1
    :type pad_value: any, optional
    """

    _max_len = max(map(len, lists))

    return np.array(
        [[x[c] if c < len(x) else pad_value
         for c in range(_max_len)]
         for x in lists]
    )

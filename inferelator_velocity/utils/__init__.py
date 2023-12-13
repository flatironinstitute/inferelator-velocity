from .misc import (
    order_dict_to_lists,
    vprint,
    get_bins,
    copy_count_layer,
    standardize_data,
    ragged_lists_to_array,
    TruncRobustScaler
)
from .aggregation import aggregate_sliding_window_times
from .graph import local_optimal_knn
from .noise2self import knn_noise2self

from .math import (
    variance,
    coefficient_of_variation,
    dot
)

import numpy as _np


def is_iterable_arg(x):
    """
    Check to see if this is a list, tuple, or numpy array

    :param x: Object
    :type x: Any
    :return: Is an iterable collection of stuff
    :rtype: bool
    """

    iarg = type(x) == list
    iarg |= type(x) == tuple
    iarg |= isinstance(x, _np.ndarray)

    return iarg

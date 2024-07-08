import numpy as np
import scipy.sparse as sps
import warnings

from inferelator_velocity.utils.keys import (
    NOISE2SELF_DIST_KEY,
    NOISE2SELF_DENOISED_KEY
)
from scself import denoise_data
from scself.utils import array_sum


def denoise(
    data,
    layer='X',
    graph=None,
    graph_key=NOISE2SELF_DIST_KEY,
    output_layer=NOISE2SELF_DENOISED_KEY,
    chunk_size=None,
    zero_threshold=None,
    obs_count_key=None,
    connectivity=False,
    dense=None
):

    lref = data.X if layer == 'X' else data.layers[layer]

    if graph is None:
        if graph_key not in data.obsp.keys():
            raise RuntimeError(
                f"Graph {graph_key} not found in data.obsp; "
                f"run global_graph() first"
            )

        graph = data.obsp[graph_key]

    if isinstance(graph, (tuple, list)):
        _cast_data = False
        for i in range(len(graph)):
            if _check_dtype(graph[i].dtype, lref.dtype):
                graph[i] = graph[i].astype(np.float64)
                _cast_data = True

    elif (_cast_data := _check_dtype(graph.dtype, lref.dtype)):
        graph = graph.astype(np.float64)

    if _cast_data:
        lref = lref.astype(np.float64)

    _denoised_data = denoise_data(
        lref,
        graph,
        zero_threshold=zero_threshold,
        chunk_size=chunk_size,
        connectivity=connectivity,
        dense=dense
    )

    if dense is True and sps.issparse(_denoised_data):
        _denoised_data = _denoised_data.toarray()
    elif dense is False and not sps.issparse(_denoised_data):
        _denoised_data = sps.csr_matrix(_denoised_data)

    if output_layer == 'X':
        data.X = _denoised_data
        out_ref = data.X
    else:
        data.layers[output_layer] = _denoised_data
        out_ref = data.layers[output_layer]

    if obs_count_key is not None:
        data.obs[obs_count_key] = array_sum(
            out_ref,
            axis=1
        )

    return data


def _check_dtype(graph_dtype, data_dtype):
    if (
        (not np.issubdtype(graph_dtype, np.floating)) or
        (graph_dtype != data_dtype)
    ):
        warnings.warn(
            f"Graph dtype {graph_dtype} is not the "
            f"same as data dtype {data_dtype}; "
            "converting both to np.float64",
            RuntimeWarning
        )
        return True
    else:
        return False

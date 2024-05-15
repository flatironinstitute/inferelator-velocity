import numpy as np
import scipy.sparse as sps

from inferelator_velocity.utils.keys import (
    NOISE2SELF_DIST_KEY,
    NOISE2SELF_DENOISED_KEY
)
from scself._noise2self.common import row_normalize
from scself.utils import array_sum
from scself import dot


def denoise(
    data,
    layer='X',
    graph_key=NOISE2SELF_DIST_KEY,
    output_layer=NOISE2SELF_DENOISED_KEY,
    dense=True,
    chunk_size=10_000,
    zero_threshold=None,
    obs_count_key=None,
    connectivity=False
):

    lref = data.X if layer == 'X' else data.layers[layer]

    if graph_key not in data.obsp.keys():
        raise RuntimeError(
            f"Graph {graph_key} not found in data.obsp; "
            f"run global_graph() first"
        )

    if data.obsp[graph_key].dtype != lref.dtype:
        raise RuntimeError(
            f"Graph dtype {data.obsp[graph_key].dtype} is not the "
            f"same as data dtype {lref.dtype}; "
            "these must match and be float32 or float64"
        )

    _n_obs = lref.shape[0]

    if chunk_size is not None:
        _n_chunks = int(np.ceil(_n_obs / chunk_size))
    else:
        _n_chunks = 1

    if _n_chunks == 1:
        _denoised_data = _denoise_chunk(
            lref,
            row_normalize(
                data.obsp[graph_key],
                connectivity=connectivity
            ),
            dense=dense,
            zero_threshold=zero_threshold
        )

    elif dense or not sps.issparse(lref):
        _denoised_data = np.zeros(lref.shape, dtype=np.float32)

        for i in range(_n_chunks):
            _start, _stop = i * chunk_size, min((i + 1) * chunk_size, _n_obs)
            if _stop <= _start:
                break

            _denoise_chunk(
                lref,
                row_normalize(
                    data.obsp[graph_key][_start:_stop, :],
                    connectivity=connectivity
                ),
                dense=True,
                out=_denoised_data[_start:_stop, :],
                zero_threshold=zero_threshold
            )

    else:
        _denoised_data = []
        for i in range(_n_chunks):
            _start, _stop = i * chunk_size, min((i + 1) * chunk_size, _n_obs)
            if _stop <= _start:
                break

            _denoised_data.append(
                _denoise_chunk(
                        lref,
                        row_normalize(
                            data.obsp[graph_key][_start:_stop, :],
                            connectivity=connectivity
                        ),
                        zero_threshold=zero_threshold
                    )
            )

        _denoised_data = sps.vstack(_denoised_data)

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


def _denoise_chunk(
    x,
    graph,
    zero_threshold=None,
    out=None,
    dense=False
):

    if not sps.issparse(x):
        dense = True

    out = dot(
        graph,
        x,
        out=out,
        dense=dense
    )

    if zero_threshold is not None and dense:
        out[out < zero_threshold] = 0
    elif zero_threshold:
        out.data[out.data < zero_threshold] = 0

        try:
            out.eliminate_zeros()
        except AttributeError:
            pass

    return out

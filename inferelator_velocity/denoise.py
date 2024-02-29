import numpy as np
import scipy.sparse as sps

from inferelator_velocity.utils.math import array_sum
from inferelator_velocity.utils.noise2self import (
    _dist_to_row_stochastic,
    dot
)
from inferelator_velocity.utils.keys import (
    NOISE2SELF_DIST_KEY,
    NOISE2SELF_DENOISED_KEY
)


def denoise(
    data,
    layer='X',
    graph_key=NOISE2SELF_DIST_KEY,
    output_layer=NOISE2SELF_DENOISED_KEY,
    dense=True,
    chunk_size=10000,
    zero_threshold=None,
    obs_count_key=None
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
        data.layers[output_layer] = _denoise_chunk(
            lref,
            _dist_to_row_stochastic(data.obsp[graph_key]),
            dense=dense,
            zero_threshold=zero_threshold
        )

    elif dense or not sps.issparse(lref):
        data.layers[output_layer] = np.zeros(lref.shape, dtype=np.float32)

        for i in range(_n_chunks):
            _start, _stop = i * chunk_size, min((i + 1) * chunk_size, _n_obs)

            _denoise_chunk(
                lref,
                _dist_to_row_stochastic(
                    data.obsp[graph_key][_start:_stop, :]
                ),
                dense=True,
                out=data.layers[output_layer][_start:_stop, :],
                zero_threshold=zero_threshold
            )

    else:
        data.layers[output_layer] = sps.vstack(
            tuple(
                _denoise_chunk(
                    lref,
                    _dist_to_row_stochastic(
                        data.obsp[graph_key][
                            i * chunk_size:min((i + 1) * chunk_size, _n_obs),
                            :
                        ]
                    ),
                    zero_threshold=zero_threshold
                )
                for i in range(_n_chunks)
            )
        )

    if obs_count_key is not None:
        data.obs[obs_count_key] = array_sum(
            data.layers[output_layer],
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
        out.eliminate_zeros()

    return out

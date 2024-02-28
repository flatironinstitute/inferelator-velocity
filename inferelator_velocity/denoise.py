import numpy as np

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
    dense=True
):

    lref = data.X if layer == 'X' else data.layers[layer]

    if graph_key not in data.obsp.keys():
        raise RuntimeError(
            f"Graph {graph_key} not found in data.obsp; "
            f"run global_graph() first"
        )

    data.layers[output_layer] = np.zeros(lref.shape, dtype=np.float32)

    dot(
        _dist_to_row_stochastic(data.obsp[graph_key]),
        lref,
        dense=dense,
        out=data.layers[output_layer]
    )

    return data

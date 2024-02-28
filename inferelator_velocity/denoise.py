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
    output_layer=NOISE2SELF_DENOISED_KEY
):

    lref = data.X if layer == 'X' else data.layers[layer]

    if graph_key not in data.obsp.keys():
        raise RuntimeError(
            f"Graph {graph_key} not found in data.obsp; "
            f"run global_graph() first"
        )

    data.layers[output_layer] = dot(
        _dist_to_row_stochastic(data.obsp[graph_key]),
        lref
    )

import numpy as np

from .utils.noise2self import knn_noise2self
from .utils import (
    vprint,
    is_iterable_arg
)
from .utils.keys import (
    OBSP_DIST_KEY,
    UNS_GRAPH_SUBKEY,
    PROGRAM_KEY,
    NOISE2SELF_KEY,
    NOISE2SELF_DIST_KEY,
    PROG_NAMES_SUBKEY
)


def global_graph(
    data,
    layer="X",
    neighbors=None,
    npcs=None,
    use_sparse=True,
    verbose=False
):
    """
    Generate a k-NN graph for data using noise2self

    :param data: Data AnnData object
    :type data: ad.AnnData
    :param layer: Layer to use for kNN, defaults to "X"
    :type layer: str, optional
    :param neighbors: Search space for k neighbors,
        defaults to 15 - 105 by 10
    :type neighbors: np.ndarray, optional
    :param npcs: Search space for number of PCs,
        defaults to 5-105 by 10
    :type npcs: np.ndarray, optional
    :param use_sparse: Use sparse data structures (slower).
        Will densify a sparse expression matrix (faster, more memory) if False,
        defaults to True
    :type use_sparse: bool
    :param verbose: Print detailed status, defaults to False
    :type verbose: bool, optional
    :return: AnnData object with `noise2self` obps and uns key
    :rtype: ad.AnnData
    """

    vprint(
        f"Embedding graph from {data.shape} data",
        verbose=verbose
    )

    lref = data.X if layer == "X" else data.layers[layer]

    data.obsp[NOISE2SELF_DIST_KEY], npc, nn, nk = knn_noise2self(
        lref,
        neighbors=neighbors,
        npcs=npcs,
        verbose=verbose,
        use_sparse=use_sparse
    )

    vprint(
        f"Embedded optimal graph "
        f"containing {np.min(nk)} - {np.max(nk)} neighbors "
        f"from {npc} PCs",
        verbose=verbose
    )

    if NOISE2SELF_KEY not in data.uns:
        data.uns[NOISE2SELF_KEY] = {}

    data.uns[NOISE2SELF_KEY]['npcs'] = npc
    data.uns[NOISE2SELF_KEY]['neighbors'] = nn


def program_graphs(
    data,
    layer="X",
    program_var_key=PROGRAM_KEY,
    programs=None,
    neighbors=None,
    npcs=None,
    use_sparse=True,
    verbose=False
):
    """
    Embed neighbor graphs for each program

    :param data: AnnData object which `ifv.program_select()` has been called on
    :type data: ad.AnnData
    :param layer: Layer containing count data, defaults to "X"
    :type layer: str, optional
    :param program_var_key: Key to find program IDs in var data,
        defaults to 'programs'
    :type program_var_key: str, optional
    :param programs: Program IDs to calculate times for, defaults to None
    :type programs: tuple, optional
    :param neighbors: k values to search for global graph,
        defaults to None (5 to 105 by 10s)
    :type neighbors: np.ndarray, optional
    :param npcs: Number of PCs to use to embed graph,
        defaults to None (5 to 105 by 10s)
    :type npcs: np.ndarray, optional
    :param use_sparse: Use sparse data structures (slower).
        Will densify a sparse expression matrix (faster, more memory) if False,
        defaults to True
    :type use_sparse: bool
    :param verbose: Print detailed status, defaults to False
    :type verbose: bool, optional
    :return: AnnData object with `program_{id}_distances` obps key
    :rtype: ad.AnnData
    """

    if programs is None:
        programs = [
            p
            for p in data.uns[PROGRAM_KEY][PROG_NAMES_SUBKEY]
            if p != '-1'
        ]
    elif is_iterable_arg(programs):
        pass
    else:
        programs = [programs]

    for prog in programs:

        _obsp = OBSP_DIST_KEY.format(prog=prog)

        _var_idx = data.var[program_var_key] == prog

        vprint(
            f"Embedding graph for {prog} "
            f"containing {np.sum(_var_idx)} genes",
            verbose=verbose
        )

        lref = data.X if layer == "X" else data.layers[layer]

        data.obsp[_obsp], npc, nn, nk = knn_noise2self(
            lref[:, _var_idx],
            neighbors=neighbors,
            npcs=npcs,
            verbose=verbose,
            use_sparse=use_sparse
        )

        vprint(
            f"Embedded optimal graph for {prog} "
            f"containing {np.min(nk)} - {np.max(nk)} neighbors "
            f"from {npc} PCs",
            verbose=verbose
        )

        # Add basic summary stats to the .uns object
        if PROGRAM_KEY not in data.uns:
            data.uns[PROGRAM_KEY] = {}

        _uns_prog_key = UNS_GRAPH_SUBKEY.format(prog=prog)

        if _uns_prog_key not in data.uns[PROGRAM_KEY]:
            data.uns[PROGRAM_KEY][_uns_prog_key] = {}

        data.uns[PROGRAM_KEY][_uns_prog_key]['npcs'] = npc
        data.uns[PROGRAM_KEY][_uns_prog_key]['neighbors'] = nn

    return data

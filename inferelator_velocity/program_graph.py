import numpy as np

from .utils.noise2self import knn_noise2self
from .utils import vprint

OBSP_KEY = "program_{prog}_distances"
UNS_SUBKEY = "program_{prog}_npcs"

def program_graphs(data,
    layer="X",
    program_var_key='program',
    programs=('0', '1'),
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
    :param program_var_key: Key to find program IDs in var data, defaults to 'program'
    :type program_var_key: str, optional
    :param programs: Program IDs to calculate times for, defaults to ('0', '1')
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

    if type(programs) == list or type(programs) == tuple or isinstance(programs, np.ndarray):
        pass
    else:
        programs = [programs]

    for prog in programs:

        _obsp = OBSP_KEY.format(prog=prog)

        _var_idx = data.var[program_var_key] == prog

        vprint(f"Embedding graph for {prog} "
               f"containing {np.sum(_var_idx)} genes",
               verbose=verbose)

        lref = data.X if layer == "X" else data.layers[layer]

        data.obsp[_obsp], npc, nn, nk = knn_noise2self(
            lref[:, _var_idx],
            neighbors=neighbors,
            npcs=npcs,
            verbose=verbose,
        )

        vprint(f"Embedded optimal graph for {prog} "
               f"containing {np.min(nk)} - {np.max(nk)} neighbors "
               f"from {npc} PCs",
               verbose=verbose)

        data.uns['programs'][UNS_SUBKEY.format(prog=prog)] = npc

    return data

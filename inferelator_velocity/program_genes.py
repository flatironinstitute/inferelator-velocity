import numpy as np
import scanpy as sc

from .utils import copy_count_layer
from .utils.keys import OBS_TIME_KEY, N_BINS
from .metrics import mutual_information
from inferelator.regression.mi import _make_array_discrete


def assign_genes_to_programs(
    data,
    layer="X",
    programs=None
):
    """
    Find programs which have highest mutual information
    with every gene

    :param data: AnnData object which `ifv.program_select()` has been called on
    :type data: ad.AnnData
    :param layer: Layer containing count data, defaults to "X"
    :type layer: str, optional
    :param programs: Program IDs to calculate times for, defaults to None
    :type programs: tuple, optional
    :return: AnnData object with `program_{id}_distances` obps key
    :rtype: ad.AnnData
    """

    if programs is None:
        programs = [
            p
            for p in data.uns['programs']['program_names']
            if p != '-1'
        ]
    elif type(programs) == list or type(programs) == tuple or isinstance(programs, np.ndarray):
        pass
    else:
        programs = [programs]

    d = copy_count_layer(data, layer)
    sc.pp.normalize_per_cell(d)

    _times = np.zeros((d.shape[0], len(programs)))

    # Unpack times into an array
    for i, p in enumerate(programs):

        _tk = OBS_TIME_KEY.format(prog=p)

        if _tk in data.obs:
            _times[:, i] = data.obs[_tk].values
        else:
            raise ValueError(
                f"Unable to find times for program {p} "
                f"in .obs[{_tk}]"
            )

    # Calculate mutual information between times and genes

    mi = mutual_information(
        _make_array_discrete(d.X, N_BINS, axis=0),
        N_BINS,
        y=_make_array_discrete(_times, N_BINS, axis=0)
    )

    return programs[np.argmax(mi, axis=0)]

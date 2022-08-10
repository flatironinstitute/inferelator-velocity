import numpy as np
import scanpy as sc
import scipy.sparse as sps

from inferelator_velocity.utils.misc import vprint

from .utils import copy_count_layer
from .utils.keys import OBS_TIME_KEY, N_BINS
from .metrics import mutual_information
from inferelator.regression.mi import _make_array_discrete


def assign_genes_to_programs(
    data,
    layer="X",
    programs=None,
    return_mi=False,
    default_program=None,
    default_threshold=None,
    n_bins=N_BINS,
    use_sparse=True,
    verbose=False
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

    programs = np.asarray(programs)

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

    vprint(
        f"Extracted times {_times.shape} "
        f"for programs {', '.join(programs)}",
        verbose=verbose
    )

    # Calculate mutual information between times and genes

    vprint(
        f"Descretizing expression into {n_bins} bins",
        verbose=verbose
    )

    _discrete_X = _make_array_discrete(
            d.X if use_sparse or not sps.issparse(d.X) else d.X.A,
            n_bins,
            axis=0
        )

    vprint(
        f"Calculating mutual information for {_discrete_X.shape[1]}"
        f" genes x {len(programs)} programs",
        verbose=verbose
    )

    mi = mutual_information(
        _discrete_X,
        n_bins,
        y=_make_array_discrete(
            _times,
            n_bins,
            axis=0
        )
    )

    new_labels = programs[np.argmax(mi, axis=1)]

    if default_program is not None:
        vprint(
            "Setting genes with low mutual information "
            f"( < {default_threshold}) to program {default_program}",
            verbose=verbose
        )

        new_labels[np.max(mi, axis=1) < default_threshold] = default_program

    if return_mi:
        return new_labels, mi
    else:
        return new_labels

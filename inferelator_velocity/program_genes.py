import numpy as np
import scipy.sparse as sps

from inferelator_velocity.utils.misc import (
    vprint,
    standardize_data
)

from .utils import (
    copy_count_layer,
    is_iterable_arg
)

from .metrics import (
    mutual_information,
    make_array_discrete
)

from .utils.keys import (
    OBS_TIME_KEY,
    N_BINS,
    PROGRAM_KEY,
    get_program_ids
)


def assign_genes_to_programs(
    data,
    layer="X",
    programs=None,
    use_existing_programs=None,
    return_mi=False,
    default_program=None,
    default_threshold=None,
    n_bins=N_BINS,
    use_sparse=True,
    verbose=False,
    standardization_method='log'
):
    """
    Assign genes to programs based on the maximum mutual information
    between time and gene expression

    :param data: AnnData object which `ifv.program_select()` has been called on
    :type data: ad.AnnData
    :param layer: Layer containing count data, defaults to "X"
    :type layer: str, optional
    :param standardization_method: Normalize per cell and log, defaults to 'log'
    :type standardization_method: str, optional
    :param programs: Program IDs to calculate times for, defaults to None
    :type programs: tuple, optional
    :param use_existing_programs: Program IDs to take from original
        calculation, when None will defaults to taking all programs (only
        replacing -1), when False will replace all programs with new programs,
        defaults to None
    :type use_existing_programs: list, None
    :param return_mi: Return Mutual Information matrix
    :type return_mi: bool
    :param default_program: If set, use this program as a default for genes
        with low mutual information to any times
    :type default_program: str, None
    :param default_threshold: If set, use this as a threshold for mutual
        information to assign genes to default program
    :type default_threshold: numeric
    :param n_bins: Number of bins for descretization of continuous data
    :type n_bins: int,
    :param use_sparse: Use sparse data structures if provided, otherwise
        convert to dense
    :type use_sparse: bool,
    :param verbose: Verbose
    :type verbose: bool
    :return: New program labels
    :rtype: np.ndarray
    """

    if programs is None:
        programs = get_program_ids(data)
    elif is_iterable_arg(programs):
        pass
    else:
        programs = [programs]

    programs = np.asarray(programs)

    old_labels = data.var[PROGRAM_KEY]

    if use_existing_programs is None:
        _has_old = old_labels != "-1"

    elif use_existing_programs is not False:
        _has_old = old_labels.isin(use_existing_programs)

    else:
        _has_old = np.zeros_like(old_labels, dtype=bool)

    if np.all(_has_old):

        vprint(
            "All genes annotated with existing labels to keep: "
            f"{np.sum(_has_old)} existing program labels kept "
            f"for programs {', '.join(programs)}",
            verbose=verbose
        )

        if return_mi:
            return old_labels, None

        else:
            return old_labels

    d = standardize_data(
        copy_count_layer(data, layer),
        method=standardization_method
    )

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

    # Check for non-finite times
    # Remove any observations which have NaN or Inf times for any program
    _time_nan = np.sum(~np.isfinite(_times), axis=1).astype(bool)
    _n_time_nan = np.sum(_time_nan > 0)

    d = d.X if use_sparse or not sps.issparse(d.X) else d.X.A

    if _n_time_nan > 0:

        vprint(
            f"Ignoring {_n_time_nan} observations which have "
            f"non-finite time values",
            verbose=verbose
        )

        d = d[~_time_nan, :]
        _times = _times[~_time_nan]

    # Calculate mutual information between times and genes

    vprint(
        f"Descretizing expression into {n_bins} bins",
        verbose=verbose
    )

    _discrete_X = make_array_discrete(
            d,
            n_bins,
            axis=0
    )

    del d

    vprint(
        f"Calculating mutual information for {_discrete_X.shape[1]}"
        f" genes x {len(programs)} programs",
        verbose=verbose
    )

    mi = mutual_information(
        _discrete_X,
        n_bins,
        y=make_array_discrete(
            _times,
            n_bins,
            axis=0
        )
    )

    new_labels = programs[np.argmax(mi, axis=1)]

    if default_program is not None:
        vprint(
            "Setting genes with low mutual information "
            f"(<{default_threshold}) to program {default_program}",
            verbose=verbose
        )

        new_labels[np.max(mi, axis=1) < default_threshold] = default_program

    new_labels[_has_old] = old_labels[_has_old]
    _labels, _counts = np.unique(new_labels, return_counts=True)

    __old_labels, _old_counts = np.unique(
        new_labels[_has_old],
        return_counts=True
    )

    if np.sum(_has_old) > 0:
        vprint(
            f"{np.sum(_has_old)} existing program labels kept: ",
            ", ".join([
                f"Program {p}: {q} genes"
                for p, q in zip(__old_labels, _old_counts)
            ]),
            verbose=verbose
        )

    vprint(
        "Genes assigned to programs: ",
        ", ".join([
            f"Program {p}: {q} genes"
            for p, q in zip(_labels, _counts)
        ]),
        verbose=verbose
    )

    if return_mi:
        return new_labels, mi
    else:
        return new_labels

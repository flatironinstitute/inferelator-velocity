import matplotlib.pyplot as plt
import numpy as np

from ..utils.keys import (
    PROGRAM_KEY,
    OBSM_PCA_KEY,
    MCV_LOSS_SUBKEY
)


def mcv_plot(
    adata,
    program=None,
    ax=None,
    add_labels=True,
    text_size=10
):
    """
    Generate a summary figure for a specific program from an AnnData object

    :param adata: Anndata object which `ifv.program_times()` has been called on
    :type adata: ad.AnnData
    :param program: Program name
    :type program: str
    :param ax: Matplotlib axis to draw into
    :type ax: Axes, optional
    :param add_labels: Add labels to axes
    :type add_labels: bool
    :return: Returns figure & axes objects if `ax=None` was passed, otherwise
        returns axes
    :rtype: matplotlib.Figure (optional), dict[Axes]
    """

    # Create a figure if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    else:
        fig = None

    # Get the data from the program assignment .uns key
    if program is None:
        data = adata.uns[PROGRAM_KEY]

    # Or a program-specific time .uns key
    else:
        data = adata.uns[OBSM_PCA_KEY.format(prog=program)]

    data = data[MCV_LOSS_SUBKEY].ravel()

    n = data.shape[0]
    min_comp = np.argmin(data)

    ax.plot(
        np.arange(1, n),
        data[1:],
        color="gray"
    )

    ax.axvline(
        min_comp,
        color='black',
        linestyle="--"
    )

    ax.set_xticks(
        [min_comp, n - 1],
        [min_comp, n - 1],
        size=text_size
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=text_size,
        labelleft=False
    )

    if add_labels:
        ax.set_ylabel(
            "MSE",
            size=text_size
        )

        ax.set_xlabel(
            "# Components",
            size=text_size
        )

    if fig is None:
        return ax
    else:
        return fig, ax


def cumulative_variance_plot(
    adata,
    program=None,
    ax=None,
    add_labels=True,
    text_size=10
):
    """
    Generate a summary figure for a specific program from an AnnData object

    :param adata: Anndata object which `ifv.program_times()` has been called on
    :type adata: ad.AnnData
    :param program: Program name
    :type program: str
    :param ax: Matplotlib axis to draw into
    :type ax: Axes, optional
    :param add_labels: Add labels to axes
    :type add_labels: bool
    :return: Returns figure & axes objects if `ax=None` was passed, otherwise
        returns axes
    :rtype: matplotlib.Figure (optional), dict[Axes]
    """

    # Create a figure if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3, 3), dpi=300)

    else:
        fig = None

    # Get the data from the program assignment .uns key
    if program is None:

        try:
            data = adata.uns['pca']
        except KeyError:
            raise RuntimeError(
                "PCA not found. "
                "Run sc.pp.pca() "
                "on this object before plotting"
            )

        try:
            n = adata.uns[PROGRAM_KEY]['n_comps']
        except KeyError:
            raise RuntimeError(
                "Program information not found. "
                "Run ifv.assign_genes_to_programs()"
                "on this object before plotting"
            )

    # Or a program-specific time .uns key
    else:
        data = adata.uns[OBSM_PCA_KEY.format(prog=program)]
        n = data['variance_ratio'].shape[0]

    data = np.cumsum(data['variance_ratio'].ravel()[:n]) * 100

    ax.plot(
        np.arange(n + 1),
        np.insert(data, 0, 0),
        color="gray"
    )

    if add_labels:

        ax.set_ylabel(
            "% Variance",
            size=text_size
        )

        ax.set_xlabel(
            "# Components",
            size=text_size
        )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=text_size
    )

    ax.set_ylim(0., None)
    ax.set_xticks(
        [0, n],
        [0, n],
        size=text_size
    )

    if fig is None:
        return ax
    else:
        return fig, ax

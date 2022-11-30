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
    ax=None
):
    """
    Generate a summary figure for a specific program from an AnnData object

    :param adata: Anndata object which `ifv.program_times()` has been called on
    :type adata: ad.AnnData
    :param program: Program name
    :type program: str
    :param ax: Matplotlib axis to draw into
    :type ax: Axes, optional
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

    ax.set_ylabel(
        "MSE"
    )

    ax.set_xlabel(
        "# Components"
    )

    ax.annotate(
        f"{min_comp} Components",
        xy=(0.2, 0.8),
        xycoords='axes fraction'
    )

    if fig is None:
        return ax
    else:
        return fig, ax

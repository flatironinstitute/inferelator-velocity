import matplotlib.pyplot as plt
import numpy as np

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from ..utils.keys import PROGRAM_KEY


def programs_summary(
    adata,
    programs_key=PROGRAM_KEY,
    cmap='magma_r',
    ax=None
):
    """
    Plot a summary of the information distance between genes
    and the resulting clustering / merging

    :param adata: Anndata object which `ifv.program_select()`
        has been called on
    :type adata: ad.AnnData
    :param programs_key: Key for programs, defaults to 'programs'
    :type programs_key: str, optional
    :param cmap: Colormap for distance heatmaps,
        defaults to 'magma_r'
    :type cmap: str, optional
    :param ax: Dict of axes to draw images into.
        'info_dist' draws information distance heatmap
        'info_cbar' draws information distance colorbar
        'corr_rows' draws cluster to program assignment ribbon bar
        'corr_dist' draws cluster correlation heatmap
        'corr_cbar' draws cluster correlation distance colorbar
        if axis is not present it will be skipped,
        defaults to None (new figure will be drawn automatically)
    :type ax: dict, optional
    :return: Figure and axes references
    :rtype: plt.Figure, dict(plt.Axes)
    """

    fig_refs = {}

    if ax is None:

        # Create a default figure for this summary plot
        fig = plt.figure(figsize=(6, 3), dpi=300)

        ax = {
            'info_dist': fig.add_axes([0.08, 0.08, 0.36, 0.82]),
            'info_cbar': fig.add_axes([0.45, 0.08, 0.015, 0.82]),
            'corr_rows': fig.add_axes([0.55, 0.08, 0.02, 0.82]),
            'corr_dist': fig.add_axes([0.575, 0.08, 0.36, 0.82]),
            'corr_cbar': fig.add_axes([0.94, 0.08, 0.015, 0.82])
        }

    else:

        # Get the figure from the first axis
        fig = list(ax.keys())[0].figure

    if 'info_dist' in ax:

        _metric = adata.uns[programs_key]['metric']
        _matrix_info = adata.uns[programs_key][
            f"{_metric}_distance"
        ]
        _idx = _hclust(_matrix_info)
        _matrix_info = _matrix_info[:, _idx][_idx, :]

        fig_refs['info_dist'] = ax['info_dist'].pcolormesh(
            _matrix_info,
            cmap=cmap,
            vmin=0,
            vmax=1
        )

        _make_heatmap_axis(ax['info_dist'])

        ax['info_dist'].set_xlabel("Genes", size=8)
        ax['info_dist'].set_xlabel("Genes", size=8)
        ax['info_dist'].set_title(
            f"{_metric.capitalize()} Dist.", size=8
        )
        ax['info_dist'].set_title("A", loc='left', weight='bold', size=8)

    if 'info_cbar' in ax:

        fig_refs['info_cbar'] = ax['info_cbar'].figure.colorbar(
            fig_refs['info_dist'],
            cax=ax['info_cbar'],
            orientation='vertical',
            ticks=[0, 1]
        )

        ax['info_cbar'].yaxis.set_tick_params(pad=0)

    _corr_dist_matrix = 1 - adata.uns[programs_key]['leiden_correlation']
    _corr_idx = _hclust(_corr_dist_matrix)

    if 'corr_rows' in ax:

        _rowmap = adata.uns[programs_key]['cluster_program_map']
        _rows = [
            float(_rowmap[str(x)]) + 1
            for x in range(len(_rowmap) - 1)
        ]

        fig_refs['rows'] = ax['corr_rows'].pcolormesh(
            np.array(_rows)[_corr_idx].reshape(-1, 1),
            cmap='Set2'
        )

        _make_heatmap_axis(ax['corr_rows'])

    if 'corr_dist' in ax:

        _corr_dist_matrix = _corr_dist_matrix[:, _corr_idx][_corr_idx, :]

        fig_refs['corr_dist'] = ax['corr_dist'].pcolormesh(
            _corr_dist_matrix,
            cmap=cmap,
            vmin=0,
            vmax=1
        )

        _make_heatmap_axis(ax['corr_dist'])

        ax['corr_dist'].set_xlabel("Clusters", size=8)
        ax['corr_dist'].set_title("Correlation Dist.", size=8)
        ax['corr_dist'].set_title("B", loc='left', weight='bold', size=8)

    if 'corr_cbar' in ax:

        fig_refs['corr_cbar'] = ax['corr_cbar'].figure.colorbar(
            fig_refs['corr_dist'],
            cax=ax['corr_cbar'],
            orientation='vertical',
            ticks=[0, 1]
        )

        ax['corr_cbar'].yaxis.set_tick_params(pad=0)

    return fig, ax


def _make_heatmap_axis(ax):

    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)


def _hclust(dist_matrix):

    _idx = dendrogram(
        linkage(
            squareform(
                dist_matrix,
                checks=False
            )
        ),
        no_plot=True
    )['leaves']

    return _idx

import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform

from ..utils.keys import PROGRAM_KEY


def programs_summary(
    adata,
    programs_key=PROGRAM_KEY,
    cmap='magma_r',
    ax=None
):

    fig_refs = {}

    if ax is None:

        fig = plt.figure(figsize=(6, 3), dpi=300)

        ax = {
            'info_dist': fig.add_axes([0.08, 0.08, 0.36, 0.82]),
            'info_cbar': fig.add_axes([0.45, 0.08, 0.015, 0.82]),
            'corr_rows': fig.add_axes([0.55, 0.08, 0.02, 0.82]),
            'corr_dist': fig.add_axes([0.575, 0.08, 0.36, 0.82]),
            'corr_cbar': fig.add_axes([0.94, 0.08, 0.015, 0.82])
        }

    else:

        fig = list(ax.keys())[0].figure

    if 'info_dist' in ax:

        _idx = dendrogram(
            linkage(
                squareform(
                    adata.uns[programs_key]['information_distance'],
                    checks=False
                )
            ),
            no_plot=True
        )['leaves']

        _matrix_info = adata.uns[programs_key]['information_distance']
        _matrix_info = _matrix_info[:, _idx][_idx, :]

        fig_refs['info_dist'] = ax['info_dist'].pcolormesh(
            _matrix_info,
            cmap=cmap,
            vmin=0,
            vmax=1
        )

        ax['info_dist'].invert_yaxis()
        ax['info_dist'].set_xticks([])
        ax['info_dist'].set_yticks([])
        ax['info_dist'].spines['right'].set_visible(False)
        ax['info_dist'].spines['top'].set_visible(False)
        ax['info_dist'].spines['left'].set_visible(False)
        ax['info_dist'].spines['bottom'].set_visible(False)
        ax['info_dist'].set_xlabel("Genes", size=8)
        ax['info_dist'].set_xlabel("Genes", size=8)
        ax['info_dist'].set_title("Information Dist.", size=8)
        ax['info_dist'].set_title("A", loc='left', weight='bold', size=8)

    if 'info_cbar' in ax:

        fig_refs['info_cbar'] = ax['info_cbar'].figure.colorbar(
            fig_refs['info_dist'],
            cax=ax['info_cbar'],
            orientation='vertical',
            ticks=[0, 1]
        )

        ax['info_cbar'].yaxis.set_tick_params(pad=0)

    if 'corr_rows' in ax:

        _rowmap = adata.uns[programs_key]['cluster_program_map']
        _rows = [
            _rowmap[str(x)]
            for x in range(len(_rowmap) - 1)
        ]

        fig_refs['rows'] = ax['corr_rows'].pcolormesh(
            _rows,
            cmap='Set2'
        )

        ax['corr_rows'].invert_yaxis()
        ax['corr_rows'].set_yticks([])
        ax['corr_rows'].set_xticks([])
        ax['corr_rows'].spines['right'].set_visible(False)
        ax['corr_rows'].spines['top'].set_visible(False)
        ax['corr_rows'].spines['left'].set_visible(False)

    if 'corr_dist' in ax:

        _matrix_corr = 1 - adata.uns[programs_key]['leiden_correlation']

        _idx = dendrogram(
            linkage(
                squareform(
                    _matrix_corr,
                    checks=False
                )
            ),
            no_plot=True
        )['leaves']

        _matrix_corr = _matrix_corr[:, _idx][_idx, :]

        fig_refs['corr_dist'] = ax['corr_dist'].pcolormesh(
            _matrix_corr,
            cmap=cmap,
            vmin=0,
            vmax=1
        )

        ax['corr_dist'].invert_yaxis()
        ax['corr_dist'].set_xticks([])
        ax['corr_dist'].set_yticks([])
        ax['corr_dist'].spines['right'].set_visible(False)
        ax['corr_dist'].spines['top'].set_visible(False)
        ax['corr_dist'].spines['left'].set_visible(False)
        ax['corr_dist'].spines['bottom'].set_visible(False)
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

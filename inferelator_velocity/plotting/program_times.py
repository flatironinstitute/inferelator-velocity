import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import pandas as pd
import numpy as np
from matplotlib.ticker import FormatStrFormatter

DEFAULT_CMAP = 'plasma'


def program_time_summary(adata, program, cluster_order=None, cluster_colors=None, cbar_cmap=None, ax=None,
                         hist_bins=80, cbar_title=None, wrap_time=None, time_limits=None, ax_key_pref=None,
                         alpha=0.5):
    """
    Plot a summary of the program-specific times

    :param adata: Anndata which `times.program_times` has been called on
    :type adata: ad.AnnData
    :param program: _description_
    :type program: _type_
    :param cluster_order: _description_, defaults to None
    :type cluster_order: _type_, optional
    :param cluster_colors: _description_, defaults to None
    :type cluster_colors: _type_, optional
    :param cbar_cmap: _description_, defaults to None
    :type cbar_cmap: _type_, optional
    :param ax: _description_, defaults to None
    :type ax: _type_, optional
    :param hist_bins: _description_, defaults to 80
    :type hist_bins: int, optional
    :param cbar_title: _description_, defaults to None
    :type cbar_title: _type_, optional
    :param wrap_time: _description_, defaults to None
    :type wrap_time: _type_, optional
    :param time_limits: _description_, defaults to None
    :type time_limits: _type_, optional
    :param ax_key_pref: _description_, defaults to None
    :type ax_key_pref: _type_, optional
    :param alpha: _description_, defaults to 0.5
    :type alpha: float, optional
    :raises ValueError: _description_
    :return: _description_
    :rtype: _type_
    """

    # Get keys

    if ax_key_pref is None:
        ax_key_pref = ''

    uns_key = f"program_{program}_pca"
    if uns_key not in adata.uns:
        raise ValueError(
            f"Unable to find program {program} in .uns[{uns_key}]. "
            "Run program_times() before calling plotter."
        )

    obs_group_key = adata.uns[uns_key]['obs_group_key']
    obs_time_key = adata.uns[uns_key]['obs_time_key']
    obsm_key = adata.uns[uns_key]['obsm_key']

    # Set up colormappings if not provided

    if cluster_order is None:
        cluster_order = np.unique(adata.obs[obs_group_key])

    _panels = adata.uns[uns_key]['assignment_names']
    n = len(_panels)

    if cbar_cmap is None:
        cbar_cmap = DEFAULT_CMAP

    cbar_cmap = cm.get_cmap(cbar_cmap, n)

    if cluster_colors is None:
        cluster_colors = {cluster_order[i]: colors.rgb2hex(cbar_cmap(i)) for i in range(n)}

    _color_vector = _get_colors(adata.obs[obs_group_key].values, cluster_colors)

    # Set up figure
    if ax is None:

        _layout = [['pca1'], ['pca2'], ['hist'], ['cbar']]

        if n < 5:
            _groups = [_panels[i] if i < n else '.' for i in range(4)]
            _layout = [_layout[i] + [_groups[i]] for i in range(4)]

        elif n < 9:
            _groups = [_panels[i] if i < n else '.' for i in range(8)]
            _layout = [_layout[i] + [_groups[2*i], _groups[2*i + 1]] for i in range(4)]

        else:
            _groups = [_panels[i] if i < n else '.' for i in range(12)]
            _layout = [_layout[i] + [_groups[3*i], _groups[3*i + 1], _groups[3*i + 2]]
                       for i in range(4)]

        fig, ax = plt.subplot_mosaic(_layout,
                                     gridspec_kw=dict(width_ratios=[1] * len(_layout[0]),
                                                      height_ratios=[1, 1, 1, 1],
                                                      wspace=0.25, hspace=0.25),
                                     figsize=(8, 8), dpi=300)
    else:
        fig = None

    refs = {}

    _centroids = [adata.uns[uns_key]['centroids'][x] for x in cluster_order]

    if ax_key_pref + 'pca1' in ax:
        refs[ax_key_pref + 'pca1'] = _plot_pca(
            adata.obsm[obsm_key][:, 0:2],
            ax[ax_key_pref + 'pca1'],
            _color_vector,
            centroid_indices=_centroids,
            shortest_path=adata.uns[uns_key]['shortest_path'],
            alpha=alpha
        )

        ax[ax_key_pref + 'pca1'].set_title(f"Program {program} PCs")
        ax[ax_key_pref + 'pca1'].set_xlabel("PC1")
        ax[ax_key_pref + 'pca1'].set_ylabel("PC2")

    if ax_key_pref + 'pca2' in ax:
        refs[ax_key_pref + 'pca2'] = _plot_pca(
            adata.obsm[obsm_key][:, [0, 2]],
            ax[ax_key_pref + 'pca2'],
            _color_vector,
            alpha=alpha
        )

        ax[ax_key_pref + 'pca2'].set_xlabel("PC1")
        ax[ax_key_pref + 'pca2'].set_ylabel("PC3")

    for i, _pname in enumerate(_panels):

        if ax_key_pref + _pname in ax:
            _idx = adata.uns[uns_key]['closest_path_assignment'] == i
            refs[ax_key_pref + 'group'] = _plot_pca(
                adata.obsm[obsm_key][:, 0:2],
                ax[ax_key_pref + _pname],
                _color_vector,
                bool_idx=_idx,
                centroid_indices=adata.uns[uns_key]['assignment_centroids'][i],
                shortest_path=adata.uns[uns_key]['assignment_path'][i],
                alpha=alpha
            )
            ax[_pname].set_title(_pname)

    if ax_key_pref + 'hist' in ax:

        _times = adata.obs[obs_time_key].values

        if wrap_time is not None:
            _times[_times > wrap_time] = _times[_times > wrap_time] - wrap_time

        refs[ax_key_pref + 'hist'] = _plot_time_histogram(
            _times,
            adata.obs[obs_group_key].values,
            ax[ax_key_pref + 'hist'],
            group_order=cluster_order,
            group_colors=cluster_colors,
            bins=hist_bins
        )

        if time_limits is not None:
            _xlim_lower = (time_limits[0] - _times.min()) / hist_bins
            _xlim_higher = time_limits[1] / _times.max() * hist_bins
            ax[ax_key_pref + 'hist'].set_xlim((_xlim_lower,
                                               _xlim_higher))

    if ax_key_pref + 'cbar' in ax:
        _add_legend(
            ax[ax_key_pref + 'cbar'],
            [cluster_colors[x] for x in cluster_order],
            cluster_order,
            title=cbar_title
        )

    return fig, ax


def _plot_pca(comps, ax, colors, bool_idx=None, centroid_indices=None, shortest_path=None, s=1, alpha=0.5):

    _xlim = comps[:, 0].min(), comps[:, 0].max()
    _ylim = comps[:, 1].min(), comps[:, 1].max()

    if bool_idx is None:
        bool_idx = np.ones(comps.shape[0], dtype=bool)

    rgen = np.random.default_rng(123)
    overplot_shuffle = np.arange(np.sum(bool_idx))
    rgen.shuffle(overplot_shuffle)

    scatter_ref = ax.scatter(comps[bool_idx, 0][overplot_shuffle], comps[bool_idx, 1][overplot_shuffle],
                             c=colors[bool_idx][overplot_shuffle],
                             s=s, alpha=alpha)

    if centroid_indices is not None:
        ax.scatter(comps[centroid_indices, 0], comps[centroid_indices, 1],
                   c='None', edgecolor='black', s=150 * s, alpha=1)

        for i in range(len(centroid_indices) - 1):
            ax.plot(comps[[centroid_indices[i], centroid_indices[i + 1]], 0],
                    comps[[centroid_indices[i], centroid_indices[i + 1]], 1],
                    ls = '--', color='black',
                    alpha = 0.5,
                    lw = 1)

    if shortest_path is not None:
        ax.plot(comps[shortest_path, 0], comps[shortest_path, 1],
                '-ok', markersize=3, linewidth=1)

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(_xlim)
    ax.set_ylim(_ylim)

    return scatter_ref


def _get_colors(values, color_dict):

    c = np.empty_like(values, dtype=object)
    for k, col in color_dict.items():
        c[values == k] = col

    return c


def _get_time_hist_data(time_data, group_data, bins, group_order=None):

    if group_order is None:
        group_order = np.unique(group_data)

    cuts = np.linspace(np.min(time_data), np.max(time_data), bins)
    return [np.bincount(pd.cut(time_data[group_data == x],
                               cuts, labels=np.arange(len(cuts) - 1)).dropna(),
                        minlength=len(cuts) - 1) for x in group_order]


def _plot_time_histogram(time_data, group_data, ax, group_order=None, group_colors=None, bins=50, xtick_format="%.2f"):

    if group_order is None:
        group_order = np.unique(group_data)

    if group_colors is None:
         _cmap = cm.get_cmap(DEFAULT_CMAP, len(group_order))
         group_colors = [colors.rgb2hex(_cmap(i)) for i in range(len(group_order))]

    hist_limit, fref = [], []
    hist_labels = np.linspace(0.5, bins - 0.5, bins - 1)

    bottom_line = None
    for j, hist_data in enumerate(_get_time_hist_data(time_data, group_data, bins, group_order)):

        bottom_line = np.zeros_like(hist_data) if bottom_line is None else bottom_line

        fref.append(ax.bar(hist_labels,
                           hist_data,
                           bottom=bottom_line,
                           width=0.5,
                           color=group_colors[group_order[j]]))

        bottom_line = bottom_line + hist_data

        hist_limit.append(np.max(np.abs(bottom_line)))

    hist_limit = max(hist_limit)
    x_mod = (time_data.max() - time_data.min()) / bins
    n_xtick = int(bins/10) + 1

    ax.set_ylim(0, hist_limit)
    ax.set_xlim(0, bins)
    ax.set_xticks([x * 10 for x in range(n_xtick)])
    ax.set_xticklabels(
        FormatStrFormatter(xtick_format).format_ticks(
            [x * x_mod * 10 for x in range(n_xtick)]
        ), rotation = 90
    )

    return fref


def _add_legend(ax, colors, labels, title=None, **kwargs):
    ax.axis('off')
    _ = [ax.scatter([], [], c=c, label=l) for c, l in zip(colors, labels)]
    return ax.legend(frameon=False,
                     loc='center left',
                     ncol=1,
                     borderpad=0.1,
                     borderaxespad=0.1,
                     columnspacing=0,
                     mode=None,
                     title=title,
                     **kwargs)

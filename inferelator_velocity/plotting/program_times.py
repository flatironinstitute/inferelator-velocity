import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd
import numpy as np
import warnings

from inferelator_velocity.utils.keys import (
    OBSM_PCA_KEY,
    OBS_GROUP_KEY_KEY,
    OBS_TIME_KEY_KEY,
    OBSM_KEY_KEY,
    N_COMP_SUBKEY,
    CLUSTER_ORDER_SUBKEY,
    SHORTEST_PATH_SUBKEY,
    ASSIGNMENT_NAME_SUBKEY,
    ASSIGNMENT_PATH_SUBKEY,
    CENTROID_SUBKEY,
    CLOSEST_ASSIGNMENT_SUBKEY,
    ASSIGNMENT_CENTROID_SUBKEY
)

from inferelator_velocity.times import _wrap_time

DEFAULT_CMAP = 'plasma'


def program_time_summary(
    adata,
    program,
    cluster_order=None,
    cluster_colors=None,
    cbar_cmap=None,
    ax=None,
    hist_bins=80,
    cbar_title=None,
    cbar_horizontal=False,
    wrap_time=None,
    time_limits=None,
    ax_key_pref=None,
    panel_tags=None,
    alpha=0.5,
    text_size=10
):

    """
    Generate a summary figure for a specific program from an AnnData object

    :param adata: Anndata object which `ifv.program_times()` has been called on
    :type adata: ad.AnnData
    :param program: Program name
    :type program: str
    :param cluster_order: List of cluster labels in order,
        defaults to None
    :type cluster_order: list, optional
    :param cluster_colors: List of matplotlib colors to use for clusters,
        defaults to None
    :type cluster_colors: list, optional
    :param cbar_cmap: Matplotlib colormap to use for cluster colors,
        has no effect if cluster_colors is passed,
        will use 'plasma' if no colors are provided by default,
        defaults to None
    :type cbar_cmap: str, matplotlib.cm, optional
    :param ax: Dict of axes to draw images into.
        'pca1' draws PC1/PC2 plot
        'pca2' draws PC1/PC3 plot
        'hist' draws histogram of cells over time colored by cluster
        'cbar' draws legend colorbar,
        '{cluster_1_label} / {cluster_2_label} draws PC1/PC2 plot with cells
            from the cluster_1 / cluster_2 vector,
        if axis is not present it will be skipped,
        defaults to None (new figure will be drawn automatically)
    :type ax: dict[Axes], optional
    :param hist_bins: Number of bins to use for histogram, defaults to 80
    :type hist_bins: int, optional
    :param cbar_title: Title for colorbar/legend, defaults to None
    :type cbar_title: str, optional
    :param cbar_horizontal: Draw colorbar/legend horizontally,
        defaults to False
    :type cbar_horizontal: bool, optional
    :param wrap_time: Wrap times at a specific time, defaults to None
    :type wrap_time: float, optional
    :param time_limits: X axis limits for histogram, defaults to None
    :type time_limits: tuple(float, float), optional
    :param ax_key_pref: Prefix to add to axes dictionary keys, defaults to None
    :type ax_key_pref: str, optional
    :param panel_tags: Dict of panel tags to add to axes, keyed by axis,
        set to False to disable tags,
        defaults to None
    :type panel_tags: dict, bool, optional
    :param alpha: Alpha value for plots, defaults to 0.5
    :type alpha: float, optional
    :raises RuntimeError: Raises RuntimeError if `ifv.program_times()` keys
        are not present in the adata object
    :return: Returns figure & axes objects if `ax=None` was passed, otherwise
        returns axes
    :rtype: matplotlib.Figure (optional), dict[Axes]
    """

    # GET ADATA KEYS AND VERIFY PROGRAM_TIMES() ####
    if ax_key_pref is None:
        ax_key_pref = ''

    uns_key = OBSM_PCA_KEY.format(prog=program)
    if uns_key not in adata.uns:
        raise RuntimeError(
            f"Unable to find program {program} in .uns[{uns_key}]. "
            "Run program_times() before calling plotter."
        )

    uns = adata.uns[uns_key]

    obs_group_key = uns[OBS_GROUP_KEY_KEY]
    obs_time_key = uns[OBS_TIME_KEY_KEY]
    obsm_key = uns[OBSM_KEY_KEY]

    # GET CLUSTER-CLUSTER PATH LABELS ####
    _panels = uns[ASSIGNMENT_NAME_SUBKEY]
    n_comps = uns[N_COMP_SUBKEY]
    n = len(_panels)

    if panel_tags is None:
        panel_tags = {
            ax_key_pref + 'pca1': "A",
            ax_key_pref + 'hist': "B",
            ax_key_pref + _panels[0]: "C"
        }

    # SET UP COLORMAPS IF NOT PROVIDED ####
    if cluster_order is None:
        cluster_order = uns[CLUSTER_ORDER_SUBKEY]

    if cbar_cmap is None:
        cbar_cmap = DEFAULT_CMAP

    cbar_cmap = matplotlib.colormaps[cbar_cmap]

    if cluster_colors is None:
        cluster_colors = {
            cluster_order[i]: colors.rgb2hex(cbar_cmap(i))
            for i in range(len(cluster_order))
        }

    # VECTOR OF COLORS BASED ON CLUSTER LABELS ####
    _color_vector = _get_colors(
        adata.obs[obs_group_key].values,
        cluster_colors
    )

    # SET UP FIGURE IF NOT PROVIDED ####
    if ax is None:

        if n_comps >= 3:
            _layout = [['pca1'], ['pca2'], ['hist'], ['cbar']]
        elif n_comps == 2:
            _layout = [['pca1'], ['.'], ['hist'], ['cbar']]
        else:
            _layout = [['.'], ['.'], ['hist'], ['cbar']]

        # IF THERE ARE 4 OR FEWER CLUSTER-CLUSTER PAIRS, DRAW 4x2 ####
        if n < 5:
            _groups = [
                _panels[i] if i < n else '.'
                for i in range(4)
            ]
            _layout = [
                _layout[i] + [_groups[i]]
                for i in range(4)
            ]
            _figsize = (4, 8)

        # IF THERE ARE 8 OR FEWER CLUSTER-CLUSTER PAIRS, DRAW 4x3 ####
        elif n < 9:
            _groups = [
                _panels[i] if i < n else '.'
                for i in range(8)
            ]
            _layout = [
                _layout[i] + [_groups[2*i], _groups[2*i + 1]]
                for i in range(4)
            ]
            _figsize = (6, 8)

        # IF THERE MORE THAN 8 CLUSTER-CLUSTER PAIRS, DRAW 4x4 ####
        else:
            _groups = [_panels[i] if i < n else '.' for i in range(12)]
            _layout = [
                _layout[i] + [_groups[3*i], _groups[3*i + 1], _groups[3*i + 2]]
                for i in range(4)
            ]
            _figsize = (8, 8)

        # WITH MORE THAN 12 CLUSTER-CLUSTER PAIRS, WARN ####
        if n > 12:
            warnings.warn(
                "program_time_summary() can only autodraw up to 12 clusters, "
                f"{n} paths are present; generate a figure and provide axes "
                "to `ax` to plot other paths",
                RuntimeWarning
            )

        # BUILD GRIDSPEC AND CALL SUBPLOT MOSAIC ####
        _gridspec = dict(
            width_ratios=[1] * len(_layout[0]),
            height_ratios=[1, 1, 1, 1],
            wspace=0.25,
            hspace=0.25
        )

        fig, ax = plt.subplot_mosaic(
            _layout,
            gridspec_kw=_gridspec,
            figsize=_figsize,
            dpi=300
        )

    else:
        fig = None

    refs = {}

    # IDENTIFY CLUSTER CENTROIDS ####
    _centroids = [
        uns[CENTROID_SUBKEY][x]
        for x in cluster_order
    ]

    if wrap_time is not None:
        _centroids = _centroids + [
            uns[CENTROID_SUBKEY][cluster_order[0]]
        ]

    _var_exp = uns['variance_ratio'].copy()
    _var_exp /= np.sum(uns['variance_ratio'])
    _var_exp *= 100

    # BUILD PC1/PC2 PLOT ####
    if ax_key_pref + 'pca1' in ax:
        refs[ax_key_pref + 'pca1'] = _plot_pca(
            adata.obsm[obsm_key][:, 0:2],
            ax[ax_key_pref + 'pca1'],
            _color_vector,
            centroid_indices=_centroids,
            shortest_path=uns[SHORTEST_PATH_SUBKEY],
            alpha=alpha
        )

        _n_comps = len(uns['variance'])
        _total_var = np.sum(uns['variance_ratio']) * 100
        ax[ax_key_pref + 'pca1'].annotate(
            f"{_n_comps} PCS ({_total_var:.1f}%)",
            xy=(0, 0),
            xycoords='data',
            xytext=(0.25, 0.05),
            textcoords='axes fraction',
            size=text_size
        )

        ax[ax_key_pref + 'pca1'].set_xlabel(
            f"PC1 ({_var_exp[0]:.1f}%)",
            size=text_size
        )
        ax[ax_key_pref + 'pca1'].set_ylabel(
            f"PC2 ({_var_exp[1]:.1f}%)",
            size=text_size
        )

    # BUILD PC1/PC3 PLOT ####
    if ax_key_pref + 'pca2' in ax:
        refs[ax_key_pref + 'pca2'] = _plot_pca(
            adata.obsm[obsm_key][:, [0, 2]],
            ax[ax_key_pref + 'pca2'],
            _color_vector,
            centroid_indices=_centroids,
            shortest_path=uns[SHORTEST_PATH_SUBKEY],
            alpha=alpha
        )

        ax[ax_key_pref + 'pca2'].set_xlabel(
            f"PC1 ({_var_exp[0]:.1f}%)",
            size=text_size
        )
        ax[ax_key_pref + 'pca2'].set_ylabel(
            f"PC3 ({_var_exp[2]:.1f}%)",
            size=text_size
        )

    # BUILD CLUSTER-CLUSTER PC1/PC2 PLOTS ####
    for i, _pname in enumerate(_panels):

        if ax_key_pref + _pname in ax:
            _idx = uns[CLOSEST_ASSIGNMENT_SUBKEY] == i

            # REMOVE PADDING ON PATH ####
            _path = uns[ASSIGNMENT_PATH_SUBKEY][i]
            _path = _path[_path != -1]

            if n_comps > 1:
                refs[ax_key_pref + 'group' + _pname] = _plot_pca(
                    adata.obsm[obsm_key][:, 0:2],
                    ax[ax_key_pref + _pname],
                    _color_vector,
                    bool_idx=_idx,
                    centroid_indices=uns[ASSIGNMENT_CENTROID_SUBKEY][i],
                    shortest_path=_path,
                    alpha=alpha
                )

                ax[ax_key_pref + _pname].set_ylabel("PC2", size=text_size)

            else:
                refs[ax_key_pref + 'group' + _pname] = _plot_time_histogram(
                    adata.obsm[obsm_key][:, 0],
                    adata.obs[obs_group_key].values,
                    ax[ax_key_pref + _pname],
                    group_order=cluster_order,
                    group_colors=cluster_colors,
                )

                ax[ax_key_pref + _pname].set_ylabel("# Cells", size=text_size)

            ax[ax_key_pref + _pname].set_title(_pname, size=text_size)
            ax[ax_key_pref + _pname].set_xlabel("PC1", size=text_size)

    # BUILD TIME HISTOGRAM PLOT ####
    if ax_key_pref + 'hist' in ax:

        _times = adata.obs[obs_time_key].values

        if wrap_time is not None:
            _times = _wrap_time(
                _times[_times > wrap_time],
                wrap_time
            )

        # Just mask out unwanted times
        # Moderately easier than sorting it out after binning
        if time_limits is not None:
            _tidx = _times >= time_limits[0]
            _tidx &= _times <= time_limits[1]
        else:
            _tidx = np.ones_like(_times, dtype=bool)

        refs[ax_key_pref + 'hist'] = _plot_time_histogram(
            _times[_tidx],
            adata.obs[obs_group_key].values[_tidx],
            ax[ax_key_pref + 'hist'],
            group_order=cluster_order,
            group_colors=cluster_colors,
            bins=hist_bins
        )

    # BUILD LEGEND ####
    if ax_key_pref + 'cbar' in ax:
        _add_legend(
            ax[ax_key_pref + 'cbar'],
            [cluster_colors[x] for x in cluster_order],
            cluster_order,
            title=cbar_title,
            horizontal=cbar_horizontal
        )

    # ADD PANEL ANNOTATION TAGS ####
    if panel_tags:
        for _k, _title in panel_tags.items():
            if _k in ax:
                ax[_k].set_title(
                    _title,
                    loc='left',
                    weight='bold',
                    size=text_size
                )

    if fig is not None:
        return fig, ax

    else:
        return ax


def _plot_pca(
    comps,
    ax,
    colors,
    bool_idx=None,
    centroid_indices=None,
    shortest_path=None,
    s=1,
    alpha=0.5
):
    """
    Plot principal components as a scatter plot into a provided axis

    :param comps: Components to plot. Rows are observations,
        plot the first two columns
    :type comps: np.ndarray
    :param ax: Matplotlib axis for drawing
    :type ax: Axes
    :param colors: Vector of colors for each observation
    :type colors: _type_
    :param bool_idx: A boolean index of points to plot, None plots all,
        defaults to None
    :type bool_idx: np.ndarray, optional
    :param centroid_indices: Location of centroid points to highlight,
        defaults to None
    :type centroid_indices: np.ndarray, list, optional
    :param shortest_path: Ordered list of points to connect with black line,
        defaults to None
    :type shortest_path: np.ndarray, list, optional
    :param s: Point size, defaults to 1
    :type s: int, optional
    :param alpha: Point alpha, defaults to 0.5
    :type alpha: float, optional
    :raises ValueError: Raises a ValueError if colors and comps are different
        sizes
    :return: Returns the PathCollection created by matplotlib scatter
    :rtype: matplotlib.collections.PathCollection
    """

    if len(colors) != comps.shape[0]:
        raise ValueError(
            f"PCA comps size {comps.shape} != color vector [{len(colors)}]"
        )

    # GET LIMIT FROM ENTIRE DATASET WITHOUT MASKING ####
    _xlim = comps[:, 0].min(), comps[:, 0].max()
    _ylim = comps[:, 1].min(), comps[:, 1].max()

    if bool_idx is None:
        bool_idx = np.ones(comps.shape[0], dtype=bool)

    # MAKE SURE TO SHUFFLE TO MITIGATE OVERPLOTTING ####
    rgen = np.random.default_rng(123)
    overplot_shuffle = np.arange(np.sum(bool_idx))
    rgen.shuffle(overplot_shuffle)

    # SCATTER PLOT ####
    scatter_ref = ax.scatter(
        comps[bool_idx, 0][overplot_shuffle],
        comps[bool_idx, 1][overplot_shuffle],
        c=colors[bool_idx][overplot_shuffle],
        s=s,
        alpha=alpha
    )

    # HIGHLIGHT CENTROIDS ####
    if centroid_indices is not None:
        ax.scatter(comps[centroid_indices, 0], comps[centroid_indices, 1],
                   c='None', edgecolor='black', s=150 * s, alpha=1)

        for i in range(len(centroid_indices) - 1):
            ax.plot(
                comps[[centroid_indices[i], centroid_indices[i + 1]], 0],
                comps[[centroid_indices[i], centroid_indices[i + 1]], 1],
                ls='--',
                color='black',
                alpha=0.5,
                lw=1
            )

    # ADD PATH WALK ####
    if shortest_path is not None:
        ax.plot(
            comps[shortest_path, 0],
            comps[shortest_path, 1],
            '-ok',
            markersize=3,
            linewidth=1
        )

    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(_xlim)
    ax.set_ylim(_ylim)

    return scatter_ref


def _get_colors(values, color_dict):
    """
    Convert a vector of labels to a vector of colors
    using a dict of labels -> colors

    :param values: Vector of labels
    :type values: np.ndarray
    :param color_dict: Dict of labels -> colors
    :type color_dict: dict
    :return: Vector of colors
    :rtype: np.ndarray
    """

    c = np.empty_like(values, dtype=object)
    for k, col in color_dict.items():
        c[values == k] = col

    return c


def _get_time_hist_data(
    time_data,
    group_data,
    bins,
    group_order=None
):

    if group_order is None:
        group_order = np.unique(group_data)

    cuts = np.linspace(np.nanmin(time_data), np.nanmax(time_data), bins)

    return [
        np.bincount(
            pd.cut(
                time_data[group_data == x],
                cuts,
                labels=np.arange(len(cuts) - 1)
            ).dropna(),
            minlength=len(cuts) - 1
        )
        for x in group_order
    ]


def _plot_time_histogram(
    time_data,
    group_data,
    ax,
    group_order=None,
    group_colors=None,
    bins=None
):

    # Set a default number of bins
    # Integer range or 50, whichever is larger
    if bins is None:
        bins = max(
            int(np.nanmax(time_data) - np.nanmin(time_data)),
            50
        )

    # Set plotting values
    cuts = np.linspace(np.nanmin(time_data), np.nanmax(time_data), bins)
    hist_width = cuts[1] - cuts[0]
    hist_labels = cuts[:-1] + (0.5 * hist_width)

    hist_xlim = (
        cuts[0] - hist_width,
        cuts[-1] + hist_width
    )

    if group_order is None:
        group_order = np.unique(group_data)

    if group_colors is None:
        _cmap = matplotlib.colormaps[DEFAULT_CMAP]
        group_colors = [
            colors.rgb2hex(_cmap(i))
            for i in range(len(group_order))
        ]

    hist_limit, fref = [], []

    bottom_line = None
    for j, hist_data in enumerate(
        _get_time_hist_data(
            time_data,
            group_data,
            bins,
            group_order
        )
    ):

        if bottom_line is None:
            bottom_line = np.zeros_like(hist_data)

        fref.append(ax.bar(
            hist_labels,
            hist_data,
            bottom=bottom_line,
            width=0.5 * hist_width,
            color=group_colors[group_order[j]]
        ))

        bottom_line = bottom_line + hist_data

        hist_limit.append(np.max(np.abs(bottom_line)))

    hist_limit = max(hist_limit)

    ax.set_ylim(0, hist_limit)
    ax.set_xlim(*hist_xlim)
    ax.set_xticks(
        hist_labels[::10],
        hist_labels[::10].astype(int),
        rotation=90
    )

    return fref


def _add_legend(ax, colors, labels, title=None, horizontal=False, **kwargs):
    ax.axis('off')
    _ = [ax.scatter([], [], c=c, label=l) for c, l in zip(colors, labels)]
    return ax.legend(
        frameon=False,
        loc='center left',
        ncol=len(labels) if horizontal else 1,
        handletextpad=0.1 if horizontal else 0.8,
        borderpad=0.1,
        borderaxespad=0.1,
        columnspacing=1 if horizontal else 0,
        mode=None,
        title=title,
        **kwargs
    )

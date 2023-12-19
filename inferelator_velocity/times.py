import anndata as ad
import scanpy as sc
import numpy as np
import warnings

from scipy.sparse.csgraph import shortest_path

from .utils.graph import get_shortest_paths, get_total_path
from .utils.math import scalar_projection, get_centroids
from .utils.mcv import mcv_pcs

from .utils import (
    vprint,
    order_dict_to_lists,
    is_iterable_arg,
    standardize_data,
    ragged_lists_to_array
)

from .utils.keys import (
    OBS_TIME_KEY,
    OBSM_PCA_KEY,
    PROGRAM_KEY,
    MCV_LOSS_SUBKEY,
    CENTROID_SUBKEY,
    ASSIGNMENT_NAME_SUBKEY,
    ASSIGNMENT_CENTROID_SUBKEY,
    CLOSEST_ASSIGNMENT_SUBKEY,
    SHORTEST_PATH_SUBKEY,
    CLUSTER_ORDER_SUBKEY,
    CLUSTER_TIME_SUBKEY,
    ASSIGNMENT_PATH_SUBKEY,
    N_COMP_SUBKEY,
    OBS_GROUP_KEY_KEY,
    OBS_TIME_KEY_KEY,
    OBSM_KEY_KEY,
    get_program_ids
)


def program_times(
    data,
    cluster_obs_key_dict,
    cluster_order_dict,
    wrap_time=None,
    layer="X",
    program_var_key=PROGRAM_KEY,
    programs=None,
    n_comps=None,
    verbose=False,
    standardization_method='log',
    mcv_kwargs={},
    nan_on_error=False
):
    """
    Calcuate times for each cell based on known cluster time values

    :param data: AnnData object which `ifv.program_select()` has been called on
    :type data: ad.AnnData
    :param cluster_obs_key_dict: Dict, keyed by program ID, of cluster
        identifiers from metadata (e.g. {'0': 'Time'}, etc)
    :type cluster_obs_key_dict: dict
    :param cluster_order_dict: Dict, keyed by program ID, of cluster centroid
        time and order. For example:
        {'PROGRAM_ID':
            {'CLUSTER_ID':
                (
                    'NEXT_CLUSTER_ID',
                    time_at_first_centroid,
                    time_at_next_centroid
                )
            }
        }
    :type cluster_order_dict: dict[tuple]
    :param layer: Layer containing count data, defaults to "X"
    :type layer: str, optional
    :param program_var_key: Key to find program IDs in var data,
        defaults to 'program'
    :type program_var_key: str, optional
    :param programs: Program IDs to calculate times for, None will
        calculate times for all programs, defaults to None
    :type programs: tuple, optional
    :param n_comps: Dict, keyed by program ID, of number of components
        to use per program. If None, select number of components by
        molecular crossvalidation. Defaults to None.
    :type n_comps: dict[int]
    :param verbose: Print detailed status, defaults to False
    :type verbose: bool, optional
    :return: AnnData object with `program_{id}_pca` obsm and uns keys and
        `program_{id}_time` obs keys.
    :rtype: ad.AnnData
    """

    if programs is None:
        programs = get_program_ids(data)
    elif is_iterable_arg(programs):
        pass
    else:
        programs = [programs]

    if program_var_key not in data.var.columns:
        raise ValueError(
            f"Column {program_var_key} is not in data.var"
        )

    for prog in programs:

        if prog not in cluster_obs_key_dict:
            raise ValueError(
                "cluster_obs_key_dict does not have a key for "
                f"program {prog}"
            )

        if prog not in cluster_order_dict:
            raise ValueError(
                "cluster_order_dict does not have a key for "
                f"program {prog}"
            )

        if cluster_obs_key_dict[prog] not in data.obs.columns:
            raise ValueError(
                f"Column {cluster_obs_key_dict[prog]} is not found "
                "in data.obs"
            )

        # Put program name into data subkeys
        _obsk = OBS_TIME_KEY.format(prog=prog)
        _obsmk = OBSM_PCA_KEY.format(prog=prog)

        # Find features assigned to the program
        _var_idx = data.var[program_var_key] == prog

        vprint(
            f"Assigning time values for program {prog} "
            f"containing {np.sum(_var_idx)} genes",
            verbose=verbose
        )

        _cluster_labels = data.obs[cluster_obs_key_dict[prog]].values

        if np.sum(_var_idx) == 0:
            data.obs[_obsk] = np.nan
            return data

        lref = data.X if layer == "X" else data.layers[layer]

        data.obs[_obsk], data.obsm[_obsmk], data.uns[_obsmk] = calculate_times(
            lref[:, _var_idx],
            _cluster_labels,
            cluster_order_dict[prog],
            return_components=True,
            verbose=verbose,
            n_comps=n_comps if n_comps is None else n_comps[prog],
            wrap_time=wrap_time[prog] if wrap_time is not None else None,
            standardization_method=standardization_method,
            mcv_kwargs=mcv_kwargs,
            nan_on_error=nan_on_error
        )

        # Add keys to the .uns object
        data.uns[_obsmk][OBS_TIME_KEY_KEY] = _obsk
        data.uns[_obsmk][OBS_GROUP_KEY_KEY] = cluster_obs_key_dict[prog]
        data.uns[_obsmk][OBSM_KEY_KEY] = _obsmk

        # Put the cluster information into the .uns object
        _cluster_order, _cluster_times = order_dict_to_lists(
            cluster_order_dict[prog]
        )

        data.uns[_obsmk][CLUSTER_ORDER_SUBKEY] = _cluster_order
        data.uns[_obsmk][CLUSTER_TIME_SUBKEY] = _cluster_times

    return data


def calculate_times(
    count_data,
    cluster_vector,
    cluster_order_dict,
    standardization_method='log',
    wrap_time=None,
    n_neighbors=10,
    n_comps=None,
    graph_method="D",
    return_components=False,
    verbose=False,
    mcv_kwargs={},
    nan_on_error=False
):
    """
    Calculate times for each cell based on count data and a known set
    of anchoring time points.

    :param count_data: Integer count data
    :type count_data: np.ndarray
    :param cluster_vector: Vector of cluster labels
    :type cluster_vector: np.ndarray
    :param cluster_order_dict: Dict of cluster centroid time and
        order. For example:
        {'CLUSTER_ID':
            ('NEXT_CLUSTER_ID', time_at_first_centroid, time_at_next_centroid)
        }
    :type cluster_order_dict: dict
    :param n_neighbors: Number of neighbors for shortest-path to
        centroid assignment, defaults to 10
    :type n_neighbors: int, optional
    :param n_comps: Number of components to use for centroid assignment,
        defaults to None (selecting with molecular crossvalidation)
    :type n_comps: int, optional
    :param graph_method: Shortest-path graph method for
        scipy.sparse.csgraph.shortest_path,
        defaults to "D" (Dijkstra's algorithm)
    :type graph_method: str, optional
    :param return_components: Return PCs and metadata if True, otherwise
        return only a vector of times, defaults to False
    :type return_components: bool, optional
    :param verbose: Print detailed status, defaults to False
    :type verbose: bool, optional
    :raises ValueError: Raise ValueError if the cluster order keys and
        the values in the cluster_vector are not compatible.
    :return: An array of time values per cell. Also an array of PCs and
        a dict of metadata, if return_components is True.
    :rtype: np.ndarray, np.ndarray (optional), dict (optional)
    """

    n = count_data.shape[0]

    # Make sure the order dict and the clusters in the data
    # have the same cluster names
    if not np.all(
        np.isin(
            np.array(list(cluster_order_dict.keys())),
            np.unique(cluster_vector)
        )
    ):
        raise ValueError(
            f"Mismatch between cluster_order_dict keys "
            f"{list(cluster_order_dict.keys())} and "
            f"cluster_vector values {np.unique(cluster_vector)}"
        )

    # If the number of components to use is not provided,
    # Do molecular crossvalidation
    if n_comps is None:
        _mcv_error = mcv_pcs(count_data, n=1, **mcv_kwargs)
        n_comps = np.median(_mcv_error, axis=0).argmin()
    else:
        _mcv_error = None

    # Construct an adata object and normalize it
    adata = standardize_data(
        ad.AnnData(count_data.astype(float)),
        method=standardization_method
    )

    # Calculate chosen PCA & neighbor graph
    sc.pp.pca(adata, n_comps=n_comps, zero_center=True)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_comps)

    vprint(
        f"Preprocessed expression count data {count_data.shape}",
        verbose=verbose
    )

    # Find centroid points to each cluster
    centroids = {
        k: adata.obs_names.get_loc(
            adata.obs_names[cluster_vector == k][idx]
        )

        for k, idx in get_centroids(
            adata.obsm['X_pca'],
            cluster_vector
        ).items()
    }

    centroid_indices = [centroids[k] for k in centroids.keys()]

    vprint(
        f"Identified centroids for groups {', '.join(centroids.keys())}",
        verbose=verbose
    )

    # Scalar projections onto centroid-centroid vector
    times = np.full(n, np.nan, dtype=float)

    try:
        # Order the centroids and build an end-to-end path
        _total_path, _tp_centroids = get_total_path(
            get_shortest_paths(
                adata.obsp['distances'],
                centroid_indices,
                graph_method=graph_method
            ),
            cluster_order_dict,
            list(centroids.keys())
        )
    except RuntimeError:
        if nan_on_error and return_components:
            return times, adata.obsm['X_pca'], adata.uns['pca']
        elif nan_on_error:
            return times
        else:
            raise

    vprint(
        f"Built {len(_total_path)} length path connecting "
        f"{len(_tp_centroids)} groups",
        verbose=verbose
    )

    # Save path information between clusters into a dict
    group = {
        # Assigned index
        'index': np.zeros(n, dtype=int),
        'names': [],
        'centroids': [],
        'path': []
    }

    # Find the nearest points on the shortest path line for every point
    # As numeric position on _total_path
    _nearest_point_on_path = shortest_path(
        adata.obsp['distances'],
        directed=False,
        indices=_total_path,
        return_predecessors=False,
        method=graph_method
    ).argmin(axis=0)

    # Make a lookup dict to find splines that are connected to
    # specific centroids
    centroid_lookup = {
        centroid: [
            i
            for i, (start, (end, _, _))
            in enumerate(cluster_order_dict.items())
            if (start == centroid) or (end == centroid)
        ]
        for centroid in _tp_centroids.keys()
    }

    # Generate times for all observations against all splines
    all_times = np.hstack([
        _time_projection(
            adata.obsm['X_pca'],
            centroids[start],
            centroids[end],
            l_time,
            r_time,
            wrap_time=wrap_time
        ).reshape(-1, 1)
        for start, (end, l_time, r_time) in cluster_order_dict.items()
    ])

    # Build a lookup table for each spline
    spline_lookup = {
        (start, end): (i, centroids[start], centroids[end])
        for i, (start, (end, _, _)) in enumerate(cluster_order_dict.items())
    }

    dists = np.full(
        (adata.shape[0], len(spline_lookup)),
        np.nan,
        dtype=float
    )

    # Get the l2 norm between orthogonal distance to the spline
    # and projection distance from the endpoints of the spline
    for i, start, end in spline_lookup.values():
        dists[:, i] = np.sqrt(
            _array_distance_to_spline(
                adata.obsm['X_pca'],
                adata.obsm['X_pca'][start, :],
                adata.obsm['X_pca'][end, :]
            ) ** 2 + scalar_projection(
                adata.obsm['X_pca'],
                start,
                end,
                normalize=False,
                endpoint_distance=True
            ) ** 2
        )

    _spline_assign = dists.argmin(axis=1)

    del dists

    # Iterate through paths between cluster centroids
    for i, (start, (end, _, _)) in enumerate(cluster_order_dict.items()):

        # Wrap the centroids if needed
        if _tp_centroids[end] == 0:
            _right_centroid = len(_total_path)
        else:
            _right_centroid = _tp_centroids[end]

        # Boolean index for the observations that are closest to the
        # shortest path connecting these centroids
        _idx = _nearest_point_on_path > _tp_centroids[start]
        _idx &= _nearest_point_on_path < _right_centroid

        # Assign time values to observations that are
        # associated with only one spline
        times[_idx] = all_times[_idx, i]

        # But for things closest to the centroids themselves,
        # average the values for splines that join at that centroid
        for c in [start, end]:
            _centroid_idx = _nearest_point_on_path == _tp_centroids[c]

            times[_centroid_idx] = _wrap_mean(
                    all_times[:, centroid_lookup[c]],
                    wrap_time,
                    axis=1
            )[_centroid_idx]

            # Find the spline that these points are closest to
            # For plotting purposes later
            # If this is an endpoint, use all values attached to the centroid,
            # otherwise find the values closest by spline distance
            _associated_spline = np.array(centroid_lookup[c])

            if len(_associated_spline) > 1:
                _spline_idx = _associated_spline[_associated_spline != i]
                _centroid_idx &= _spline_assign == _spline_idx

            _idx |= _centroid_idx

        vprint(
            f"Assigned times to {np.sum(_idx)} cells [{start} - {end}] "
            f"conected by {_right_centroid - _tp_centroids[start]} points",
            verbose=verbose
        )

        # Append information about this path to lists in the dict
        # Line number for the closest observations
        group['index'][_idx] = i

        # Line name (cluster / cluster)
        group['names'].append(f"{start} / {end}")

        # The location of the centroids for this path
        group['centroids'].append(
            (centroids[start], centroids[end])
        )

        # The path that connects the centroids
        group['path'].append(
            _total_path[_tp_centroids[start]:_right_centroid + 1]
        )

    # Only do verbose message math if needed
    if verbose:
        vprint(
            f"Assigned times to {np.sum(~np.isnan(times))} cells "
            f"[{np.nanmin(times):.3f} - {np.nanmax(times):.3f}]",
            verbose=verbose
        )

    # PAD PATH WITH -1s AND CONVERT TO AN ARRAY ####
    group['path'] = ragged_lists_to_array(group['path'])

    # Assign result information to an .uns key
    adata.uns['pca'][CENTROID_SUBKEY] = centroids
    adata.uns['pca'][SHORTEST_PATH_SUBKEY] = _total_path
    adata.uns['pca'][CLOSEST_ASSIGNMENT_SUBKEY] = group['index']
    adata.uns['pca'][ASSIGNMENT_NAME_SUBKEY] = group['names']
    adata.uns['pca'][ASSIGNMENT_CENTROID_SUBKEY] = group['centroids']
    adata.uns['pca'][MCV_LOSS_SUBKEY] = _mcv_error
    adata.uns['pca'][ASSIGNMENT_PATH_SUBKEY] = group['path']
    adata.uns['pca'][N_COMP_SUBKEY] = n_comps

    if return_components:
        return times, adata.obsm['X_pca'], adata.uns['pca']
    else:
        return times


def wrap_times(
    data,
    program,
    wrap_time
):
    """
    Wrap a time value at a specific time
    Useful if time is a circle

    :param data: AnnData object which has been processed by
        program_times
    :type data: ad.AnnData
    :param program: Program Name
    :type program: str
    :param wrap_time: Time to wrap at
    :type wrap_time: numeric
    :return: AnnData with modified times for program
    :rtype: ad.AnnData
    """

    _obsk = OBS_TIME_KEY.format(prog=program)

    _times = data.obs[_obsk].values

    if _times.max() > 2 * wrap_time or _times.min() < -1 * wrap_time:
        warnings.warn(
            f"Wrapping times in .obs[{_obsk}] at {wrap_time} "
            f"May have undesired behavior because times range from "
            f"{_times.min():.3f} to {_times.max():3f}"
        )

    data.obs[_obsk] = _wrap_time(_times, wrap_time)

    return data


def _time_projection(
        x,
        start_position,
        end_position,
        start_time,
        end_time,
        wrap_time=None
):
    """
    Project data onto scalar and standardize to time anchors

    :param x: Data
    :type x: np.ndarray
    :param start_position: Row in data of spline start
    :type start_position: int
    :param end_position: Row in data of spline end
    :type end_position: int
    :param start_time: Time value at the spline start
    :type start_time: float
    :param end_time: Time value at the spline end
    :type end_time: float
    :param wrap_time: Wrapping time, defaults to None
    :type wrap_time: float, optional
    :return: Time vector
    :rtype: np.ndarray
    """

    time = scalar_projection(
        x,
        start_position,
        end_position,
    ) * (end_time - start_time) + start_time

    if wrap_time is not None:
        time = _wrap_time(time, wrap_time)

    return time


def _wrap_time(
    times,
    wrap_time
):
    """
    Wrap values above something and below zero
    """

    times[times > wrap_time] = times[times > wrap_time] - wrap_time
    times[times < 0] = times[times < 0] + wrap_time

    return times


def _wrap_mean(
    times,
    wrap_time,
    axis=1
):
    if wrap_time is None:
        return times.mean(axis=axis)

    t_diff = np.abs(np.diff(times, axis=axis)).ravel()

    _wrap_diff = t_diff > (wrap_time / 2)

    t_mean = np.min(times, axis=axis)

    t_mean[~_wrap_diff] += t_diff[~_wrap_diff] / 2
    t_mean[_wrap_diff] -= (wrap_time - t_diff[_wrap_diff]) / 2

    return _wrap_time(t_mean, wrap_time)


def _array_distance_to_spline(x, c1, c2):
    """
    Get the distance to a spline defined by two points
    (c1 and c2) for an array (x)

    :param x: Array of N points in m-space
    :type x: np.ndarray
    :param c1: Vector in m-space defining the first point
    :type c1: np.ndarray
    :param c2: Vector in m-space defining the second point
    :type c2: np.ndarray
    :return: N Distances to the spline defined by c1 and c2
    :rtype: np.ndarray
    """

    delta_c = c1 - c2

    t = np.dot(x - c2[None, :], delta_c) / np.dot(delta_c, delta_c)

    return np.linalg.norm(
        t[:, None] * delta_c[None, :] + c2[None, :] - x,
        axis=1
    )

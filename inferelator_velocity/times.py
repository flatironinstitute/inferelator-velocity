import anndata as ad
import scanpy as sc
import numpy as np
import warnings

from .utils.graph import get_shortest_paths, get_total_path
from .utils.math import scalar_projection, get_centroids
from .utils.mcv import mcv_pcs
from .utils import vprint, order_dict_to_lists

from scipy.sparse.csgraph import shortest_path


OBS_KEY = "program_{prog}_time"
OBSM_KEY ="program_{prog}_pca"


def program_times(data,
    cluster_obs_key_dict,
    cluster_order_dict,
    layer="X",
    program_var_key='program',
    programs=('0', '1'),
    n_comps=None,
    verbose=False
):
    """
    Calcuate times for each cell based on known cluster time values

    :param data: AnnData object which `ifv.program_select()` has been called on
    :type data: ad.AnnData
    :param cluster_obs_key_dict: Dict, keyed by program ID, of cluster identifiers from
        metadata (e.g. {'0': 'Time'}, etc)
    :type cluster_obs_key_dict: dict
    :param cluster_order_dict: Dict, keyed by program ID, of cluster centroid time and
        order. For example:
        {'PROGRAM_ID':
            {'CLUSTER_ID':
                ('NEXT_CLUSTER_ID', time_at_first_centroid, time_at_next_centroid)
            }
        }
    :type cluster_order_dict: dict[tuple]
    :param layer: Layer containing count data, defaults to "X"
    :type layer: str, optional
    :param program_var_key: Key to find program IDs in var data, defaults to 'program'
    :type program_var_key: str, optional
    :param programs: Program IDs to calculate times for, defaults to ('0', '1')
    :type programs: tuple, optional
    :param n_comps: Dict, keyed by program ID, of number of components to use per program,
        defaults to selecting with molecular crossvalidation
    :type n_comps: dict[int]
    :param verbose: Print detailed status, defaults to False
    :type verbose: bool, optional
    :return: AnnData object with `program_{id}_pca` obsm and uns keys and
        `program_{id}_time` obs keys.
    :rtype: ad.AnnData
    """

    if type(programs) == list or type(programs) == tuple or isinstance(programs, np.ndarray):
        pass
    else:
        programs = [programs]

    for prog in programs:

        _obsk = OBS_KEY.format(prog=prog)
        _obsmk = OBSM_KEY.format(prog=prog)

        _var_idx = data.var[program_var_key] == prog

        vprint(f"Assigning time values for program {prog} "
               f"containing {np.sum(_var_idx)} genes",
               verbose=verbose)

        _cluster_labels = data.obs[cluster_obs_key_dict[prog]].values

        if np.sum(_var_idx) == 0:
            data.obs[_obsk] = np.nan

        else:
            lref = data.X if layer == "X" else data.layers[layer]

            data.obs[_obsk], data.obsm[_obsmk], data.uns[_obsmk] = calculate_times(
                lref[:, _var_idx],
                _cluster_labels,
                cluster_order_dict[prog],
                return_components=True,
                verbose=verbose,
                n_comps=n_comps if n_comps is None else n_comps[prog]
            )

            data.uns[_obsmk]['obs_time_key'] = _obsk
            data.uns[_obsmk]['obs_group_key'] = cluster_obs_key_dict[prog]
            data.uns[_obsmk]['obsm_key'] = _obsmk

            _cluster_order, _cluster_times = order_dict_to_lists(
                cluster_order_dict[prog]
            )

            data.uns[_obsmk]['cluster_order'] = _cluster_order
            data.uns[_obsmk]['cluster_times'] = _cluster_times

    return data


def calculate_times(count_data,
                    cluster_vector,
                    cluster_order_dict,
                    n_neighbors=10,
                    n_comps=None,
                    graph_method="D",
                    return_components=False,
                    verbose=False
                    ):
    """
    Calculate times for each cell based on count data and a known set of anchoring
    time points.

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
    :param n_neighbors: Number of neighbors for shortest-path to centroid assignment,
        defaults to 10
    :type n_neighbors: int, optional
    :param n_comps: Number of components to use for centroid assignment,
        defaults to None (selecting with molecular crossvalidation)
    :type n_comps: int, optional
    :param graph_method: Shortest-path graph method for
        scipy.sparse.csgraph.shortest_path,
        defaults to "D" (Dijkstra's algorithm)
    :type graph_method: str, optional
    :param return_components: Return PCs and metadata if True, otherwise return only
        a vector of times, defaults to False
    :type return_components: bool, optional
    :param verbose: Print detailed status, defaults to False
    :type verbose: bool, optional
    :raises ValueError: Raise ValueError if the cluster order keys and the values in the
        cluster_vector are not compatible.
    :return: An array of time values per cell. Also an array of PCs and a dict of metadata,
        if return_components is True.
    :rtype: np.ndarray, np.ndarray (optional), dict (optional)
    """

    n = count_data.shape[0]

    if not np.all(np.isin(
        np.array(list(cluster_order_dict.keys())),
        np.unique(cluster_vector))
    ):
        raise ValueError(
            f"Mismatch between cluster_order_dict keys {list(cluster_order_dict.keys())} "
            f"And cluster_vector values {np.unique(cluster_vector)}"
        )

    adata = ad.AnnData(count_data, dtype=float)
    sc.pp.normalize_per_cell(adata, min_counts=0)
    sc.pp.log1p(adata)

    if n_comps is None:
        n_comps = np.median(mcv_pcs(count_data, n=1), axis=0).argmin()

    sc.pp.pca(adata, n_comps=n_comps, zero_center=True)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_comps)

    vprint(f"Preprocessed expression count data {count_data.shape}",
           verbose=verbose)

    centroids = {k: adata.obs_names.get_loc(adata.obs_names[cluster_vector == k][idx])
                 for k, idx in get_centroids(adata.obsm['X_pca'], cluster_vector).items()}

    centroid_indices = [centroids[k] for k in centroids.keys()]

    vprint(f"Identified centroids for groups {', '.join(centroids.keys())}",
           verbose=verbose)

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

    vprint(f"Built {len(_total_path)} length path connecting "
           f"{len(_tp_centroids)} groups",
           verbose=verbose)

    # Find the nearest points on the shortest path line for every point
    # As numeric position on _total_path
    _nearest_point_on_path = shortest_path(
        adata.obsp['distances'],
        directed=False,
        indices=_total_path,
        return_predecessors=False,
        method=graph_method
    ).argmin(axis=0)

    # Scalar projections onto centroid-centroid vector
    times = np.full(n, np.nan, dtype=float)

    group = {
        'index': np.zeros(n, dtype=int),
        'names': [],
        'centroids': [],
        'path': []
    }

    for start, (end, left_time, right_time) in cluster_order_dict.items():

        if _tp_centroids[end] == 0:
            _right_centroid = len(_total_path)
        else:
            _right_centroid = _tp_centroids[end]

        _idx = _nearest_point_on_path >= _tp_centroids[start]
        _idx &= _nearest_point_on_path <= _right_centroid

        vprint(f"Assigned times to {np.sum(_idx)} cells [{start} - {end}] "
               f"conected by {_right_centroid - _tp_centroids[start]} points",
               verbose=verbose)

        times[_idx] = scalar_projection(
            adata.obsm['X_pca'],
            centroids[start],
            centroids[end],
        )[_idx] * (right_time - left_time) + left_time

        group['index'][_idx] = len(group['names'])
        group['names'].append(f"{start} / {end}")
        group['centroids'].append((centroids[start], centroids[end]))
        group['path'].append(_total_path[max(0, _tp_centroids[start] - 1):_right_centroid])

    if verbose:
        vprint(f"Assigned times to {np.sum(~np.isnan(times))} cells "
               f"[{np.nanmin(times):.3f} - {np.nanmax(times):.3f}]",
               verbose=verbose)

    adata.uns['pca']['centroids'] = centroids
    adata.uns['pca']['shortest_path'] = _total_path
    adata.uns['pca']['closest_path_assignment'] = group['index']
    adata.uns['pca']['assignment_names'] = group['names']
    adata.uns['pca']['assignment_centroids'] = group['centroids']

    # PAD PATH WITH -1s AND CONVERT TO AN ARRAY FOR ANNDATA.WRITE() ####
    _path_max_len = max(map(len, group['path']))
    group['path'] = np.array([[x[c] if c < len(x) else -1
                               for c in range(_path_max_len)]
                              for x in group['path']])

    adata.uns['pca']['assignment_path'] = group['path']

    if return_components:
        return times, adata.obsm['X_pca'], adata.uns['pca']
    else:
        return times


def wrap_times(data, program, wrap_time):
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

    _obsk = OBS_KEY.format(prog=program)

    _times = data.obs[_obsk].values

    if _times.max() > 2 * wrap_time or _times.min() < -1 * wrap_time:
        warnings.warn(
            f"Wrapping times in .obs[{_obsk}] at {wrap_time} "
            f"May have undesired behavior because times range from "
            f"{_times.min():.3f} to {_times.max():3f}"
        )

    _times[_times > wrap_time] = _times[_times > wrap_time] - wrap_time
    _times[_times < 0] = _times[_times < 0] + wrap_time

    data.obs[_obsk] = _times

    return data

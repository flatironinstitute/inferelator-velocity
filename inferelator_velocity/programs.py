import numpy as np
import anndata as ad
import warnings

import scanpy as sc
from scanpy.neighbors import compute_neighbors_umap, _compute_connectivities_umap

from sklearn.cluster import AgglomerativeClustering
from scipy.stats import spearmanr
from sklearn.metrics import pairwise_distances

from inferelator.regression.mi import _make_array_discrete
from .utils.mcv import mcv_pcs
from .utils import vprint
from .metrics import information_distance

import pandas.api.types as pat

mi_bins = 10


def program_select(
    data,
    n_programs=2,
    n_comps=None,
    layer="X",
    mcv_loss_arr=None,
    n_jobs=-1,
    verbose=False,
    filter_to_hvg=True,
    metric='information'
):
    """
    Find a specific number of gene programs based on information distance between genes.
    Use raw counts as input.

    :param data: AnnData expression data object
    :type data: ad.AnnData
    :param n_programs: Number of gene programs, defaults to 2
    :type n_programs: int, optional
    :param n_comps: Number of components to use,
        overrides molecular crossvalidation,
        *only set a value for testing purposes*,
        defaults to None
    :type n_comps: int, optional
    :param layer: Data layer to use, defaults to "X"
    :type layer: str, optional
    :param mcv_loss_arr: An array of molecular crossvalidation
        loss values [n x n_pcs],
        will be calculated if not provided,
        defaults to None
    :type mcv_loss_arr: np.ndarray, optional
    :param n_jobs: Number of CPU cores to use for parallelization,
        defaults to -1 (all cores)
    :type n_jobs: int, optional
    :param verbose: Print status, defaults to False
    :type verbose: bool, optional
    :param filter_to_hvg: Filter to only high-dispersion genes,
        as defined by seurat, defaults to True.
        If False, filter genes that do not appear in 10+ cells
    :type filter_to_hvg: bool, optional
    :param metric: Which metric to use for distance.
        Accepts 'information', and any metric which is accepted
        by sklearn.metrics.pairwise_distances.
        Defaults to 'information'.
    :type metric: str, optional
    :return: Data object with new attributes:
        .obsm['program_PCs']: Principal component for each program
        .var['leiden']: Leiden cluster ID
        .var['program']: Program ID
        .uns['MI_program']: {
            'metric': Metric name,
            'leiden_correlation': Absolute value of spearman rho
                between PC1 of each leiden cluster,
            'metric_genes': Gene labels for distance matrix
            '{metric}_distance': Distance matrix for {metric},
            'cluster_program_map': Dict mapping gene clusters to gene programs,
            'program_PCs_variance_ratio': Variance explained by program PCs,
            'n_comps': Number of PCs selected by molecular crossvalidation,
            'molecular_cv_loss': Loss values for molecular crossvalidation
        }
    :rtype: AnnData object
    """

    #### CREATE A NEW DATA OBJECT FOR THIS ANALYSIS ####

    lref = data.X if layer == 'X' else data.layers[layer]

    if not pat.is_integer_dtype(lref.dtype):
        warnings.warn(
            "program_select expects count data "
            f"but {lref.dtype} data has been passed. "
            "This data will be normalized and processed "
            "as count data. If it is not count data, "
            "these results will be nonsense."
        )

    d = ad.AnnData(lref, dtype=float)
    d.layers['counts'] = lref.copy()
    d.var = data.var.copy()

    #### PREPROCESSING / NORMALIZATION ####

    sc.pp.normalize_per_cell(d)
    sc.pp.log1p(d)

    if filter_to_hvg:
        sc.pp.highly_variable_genes(d, max_mean=np.inf, min_disp=0.01)
        d._inplace_subset_var(d.var['highly_variable'].values)

        vprint(f"Normalized and kept {d.shape[1]} highly variable genes",
               verbose=verbose)
    else:
        sc.pp.filter_genes(d, min_cells=10)

        vprint(f"Normalized and kept {d.shape[1]} expressed genes",
               verbose=verbose)


    #### PCA / COMPONENT SELECTION BY MOLECULAR CROSSVALIDATION ####
    if n_comps is None:

        if mcv_loss_arr is None:
            mcv_loss_arr = mcv_pcs(
                d.layers['counts'],
                n=1,
                n_pcs=min(d.shape[1] - 1, 100)
            )

        if mcv_loss_arr.ndim == 2:
            n_comps = np.median(mcv_loss_arr, axis=0).argmin()
        else:
            n_comps = mcv_loss_arr.argmin()

    sc.pp.pca(d, n_comps=n_comps)

    vprint(f"Using {n_comps} components", verbose=verbose)

    # Rotate back to expression space
    pca_expr = d.obsm['X_pca'] @ d.varm['PCs'].T

    #### CALCULATING MUTUAL INFORMATION & GENE CLUSTERING ####

    vprint(f"Calculating information distance for {pca_expr.shape} array",
           verbose=verbose)

    if metric != 'information':
        vprint(f"Calculating {metric} distance for {pca_expr.shape} array",
               verbose=verbose)

        dists = pairwise_distances(pca_expr.T, metric=metric)
        mutual_info = np.array([])

    else:
        dists, mutual_info = information_distance(
            _make_array_discrete(pca_expr, mi_bins, axis=0),
            mi_bins,
            n_jobs=n_jobs,
            logtype=np.log2,
            return_information=True
        )

    vprint(f"Calculating k-NN and Leiden for {dists.shape} distance array",
           verbose=verbose)

    ### k-NN & LEIDEN - 15 <= N_GENES / 100 <= 100 neighbors
    d.var['leiden'] = _leiden_cluster(
        dists,
        min(100, max(int(dists.shape[0] / 100), 15)),
        leiden_kws={'random_state': 50}
    )

    _n_l_clusts = d.var['leiden'].nunique()

    vprint(f"Found {_n_l_clusts} unique gene clusters",
           verbose=verbose)

    _cluster_pc1 = np.zeros((d.shape[0], _n_l_clusts), dtype=float)
    for i in range(_n_l_clusts):
        _cluster_pc1[:, i] = _get_pcs(
            d.layers['counts'][:, d.var['leiden'] == i],
            return_var_explained=False
        ).ravel()

    #### SECOND ROUND OF CLUSTERING TO MERGE GENE CLUSTERS INTO PROGRAMS ####

    _rho_pc1 = np.abs(spearmanr(_cluster_pc1))[0]

    vprint(f"Merging {_n_l_clusts} gene clusters into {n_programs} programs",
           verbose=verbose)

    # Merge clusters based on correlation distance (1 - abs(spearman rho))
    clust_2 = AgglomerativeClustering(n_clusters=n_programs,
                                      affinity='precomputed',
                                      linkage='complete').fit_predict(1 - _rho_pc1)

    clust_map = {str(k): str(clust_2[k]) for k in range(_n_l_clusts)}
    clust_map[str(-1)] = str(-1)

    d.var['program'] = d.var['leiden'].astype(str).map(clust_map)

    #### LOAD FINAL DATA INTO INITIAL DATA OBJECT AND RETURN IT ####

    data.var['leiden'] = str(-1)
    data.var.loc[d.var_names, 'leiden'] = d.var['leiden'].astype(str)
    data.var['program'] = data.var['leiden'].map(clust_map)

    #### ADD RESULTS OBJECT TO UNS ####

    data.uns['programs'] = {
        'metric': metric,
        'leiden_correlation': _rho_pc1,
        'metric_genes': d.var_names.values,
        f'{metric}_distance': dists,
        'cluster_program_map': clust_map,
        'n_comps': n_comps,
        'n_programs': n_programs,
        'program_names': np.unique(data.var['program']),
        'molecular_cv_loss': mcv_loss_arr
    }

    if metric == 'information':
        data.uns['programs']['mutual_information'] = mutual_info

    return data


def program_pcs(data, program_id_vector, program_id_levels=None,
                skip_program_ids=(str(-1)), normalize=True, n_pcs=1):
    """
    Calculate principal components for a set of expression programs

    :param data: Expression data [Obs x Features]
    :type data: np.ndarray, sp.spmatrix
    :param program_id_vector: List mapping Features to Program ID
    :type program_id_vector: pd.Series, np.ndarray, list
    :param skip_program_ids: Program IDs to skip, defaults to (str(-1))
        Ignored if program_id_levels is set.
    :type skip_program_ids: tuple, list, pd.Series, np.ndarray, optional
    :param program_id_levels: Program IDs and order for output array
    :type program_id_levels: pd.Series, np.ndarray, list, optional
    :param normalize: Normalize expression data, defaults to False
    :type normalize: bool, optional
    :param n_pcs: Number of PCs to include for each program,
        defaults to 1
    :type n_pcs: int, optional
    :returns: A numpy array with the program PCs for each program,
        a numpy array with the program PCs variance ratio for each program,
        and a list of Program IDs if program_id_levels is not set
    :rtype: np.ndarray, np.ndarray, list (optional)
    """

    if program_id_levels is None:
        use_ids = [i for i in np.unique(program_id_vector)
                   if i not in skip_program_ids]
    else:
        use_ids = program_id_levels

    p_pcs = np.zeros((data.shape[0], len(use_ids) * n_pcs), dtype=float)
    vr_pcs = np.zeros(len(use_ids) * n_pcs, dtype=float)

    for i, prog_id in enumerate(use_ids):
        _idx, _r_idx = i * n_pcs, i * n_pcs + n_pcs

        p_pcs[:, _idx:_r_idx], vr_pcs[_idx:_r_idx] = _get_pcs(
            data[:, program_id_vector == prog_id],
            normalize=normalize,
            n_pcs=n_pcs
        )

    if program_id_levels is not None:
        return p_pcs, vr_pcs

    else:
        return p_pcs, vr_pcs, use_ids


def _get_pcs(data, n_pcs=1, normalize=True, return_var_explained=True):
    """
    Get the values for PC1

    :param data: Data array or matrix [Obs x Var]
    :type data: np.ndarray, sp.spmatrix
    :param n_pcs: Number of PCs to include, defaults to 1
    :type n_pcs: int
    :param normalize: Normalize depth & log transform, defaults to True
    :type normalize: bool, optional
    :return: PCs [Obs, n_pcs]
    :rtype: np.ndarray
    """

    if normalize:
        data = sc.pp.log1p(
            sc.pp.normalize_per_cell(
                data.astype(float),
                copy=True,
                min_counts=0
            )
        )
    else:
        data = data.astype(float)

    _pca_X, _, _pca_var_ratio, _ = sc.pp.pca(
        data,
        n_comps=n_pcs,
        zero_center=True,
        return_info=True
    )

    if return_var_explained:
        return _pca_X[:, 0:n_pcs], _pca_var_ratio[0:n_pcs]
    else:
        return _pca_X[:, 0:n_pcs]


def _leiden_cluster(dist_array, n_neighbors, random_state=100, leiden_kws=None):

    # Calculate neighbors using scanpy internals
    # (Needed as there's no way to provide a distance matrix)
    knn_i, knn_dist, _ = compute_neighbors_umap(
        dist_array,
        n_neighbors,
        random_state,
        metric='precomputed'
    )

    knn_dist, knn_connect = _compute_connectivities_umap(
        knn_i, knn_dist, dist_array.shape[0], n_neighbors
    )

    leiden_kws = {} if leiden_kws is None else leiden_kws
    leiden_kws['adjacency'] = knn_connect
    leiden_kws['random_state'] = leiden_kws.get('random_state', random_state)

    ad_arr = ad.AnnData(dist_array, dtype=float)

    sc.tl.leiden(ad_arr, **leiden_kws)

    return ad_arr.obs['leiden'].astype(int).values

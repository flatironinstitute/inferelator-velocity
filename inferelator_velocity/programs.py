import numpy as np
import anndata as ad
from scipy import sparse

import scanpy as sc
from scanpy.neighbors import (
    compute_neighbors_umap,
    _compute_connectivities_umap
)

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

from inferelator.regression.mi import _make_array_discrete
from .utils.mcv import mcv_pcs

from .utils import (
    vprint,
    copy_count_layer,
    standardize_data
)
from .metrics import (
    information_distance,
    circular_rank_correlation
)

from .utils.keys import (
    N_BINS,
    PROGRAM_KEY,
    METRIC_SUBKEY,
    METRIC_GENE_SUBKEY,
    METRIC_DIST_SUBKEY,
    MCV_LOSS_SUBKEY,
    LEIDEN_CORR_SUBKEY,
    PROGRAM_CLUST_SUBKEY,
    N_COMP_SUBKEY,
    N_PROG_SUBKEY,
    PROG_NAMES_SUBKEY
)


def program_select(
    data,
    n_programs=2,
    n_comps=None,
    layer="X",
    count_layer=None,
    standardization_method='log',
    mcv_loss_arr=None,
    n_jobs=-1,
    verbose=False,
    filter_to_hvg=True,
    metric='cosine',
    random_state=50,
    mcv_kwargs={}
):
    """
    Find a specific number of gene programs based on information distance
    between genes. Use raw counts as input.

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
    :param normalize: Normalize per cell and log, defaults to True
    :type normalize: bool
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
    :param random_state: Random seed for leiden, defaults to 50
    :type random_state: int
    :return: Data object with new attributes:
        .var['leiden']: Leiden cluster ID
        .var['programs']: Program ID
        .uns['programs']: {
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

    # CREATE A NEW DATA OBJECT FOR THIS ANALYSIS #

    d = copy_count_layer(data, layer, counts_layer=count_layer)

    # PREPROCESSING / NORMALIZATION #

    standardize_data(
        d,
        method=standardization_method
    )

    if filter_to_hvg:

        # Use seurat-type HVG
        sc.pp.highly_variable_genes(
            d,
            max_mean=np.inf,
            min_disp=0.0
        )

        d._inplace_subset_var(d.var['highly_variable'].values)

        vprint(
            f"Kept {d.shape[1]} high variance genes",
            verbose=verbose
        )

    else:
        sc.pp.filter_genes(d, min_cells=10)

        vprint(
            f"Kept {d.shape[1]} expressed genes",
            verbose=verbose
        )

    # PCA / COMPONENT SELECTION BY MOLECULAR CROSSVALIDATION #
    if n_comps is None:

        if mcv_loss_arr is None:
            mcv_loss_arr = mcv_pcs(
                d.layers['counts'],
                n=1,
                n_pcs=min(d.shape[1] - 1, 100),
                **mcv_kwargs
            )

        if mcv_loss_arr.ndim == 2:
            n_comps = np.median(mcv_loss_arr, axis=0).argmin()
        else:
            n_comps = mcv_loss_arr.argmin()

    sc.pp.pca(d, n_comps=n_comps)

    vprint(
        f"Selecting programs from expression based on {n_comps} components",
        verbose=verbose
    )

    # Rotate back to expression space
    pca_expr = d.obsm['X_pca'] @ d.varm['PCs'].T

    # CALCULATING MUTUAL INFORMATION & GENE CLUSTERING #

    if metric != 'information':
        vprint(
            f"Calculating {metric} distance for {pca_expr.shape} array",
            verbose=verbose
        )

        dists = pairwise_distances(pca_expr.T, metric=metric)
        mutual_info = np.array([])

    else:
        vprint(
            f"Calculating information distance for {pca_expr.shape} array",
            verbose=verbose
        )

        dists, mutual_info = information_distance(
            _make_array_discrete(pca_expr, N_BINS, axis=0),
            N_BINS,
            n_jobs=n_jobs,
            logtype=np.log2,
            return_information=True
        )

    vprint(
        f"Clustering on {dists.shape} {metric} distance array",
        verbose=verbose
    )

    # k-NN & LEIDEN - 2 <= N_GENES / 100 <= 100 neighbors
    d.var['leiden'] = _leiden_cluster(
        dists,
        min(100, max(int(dists.shape[0] / 100), 2)),
        leiden_kws={'random_state': random_state}
    )

    _n_l_clusts = d.var['leiden'].nunique()

    vprint(
        f"Clustered into {_n_l_clusts} unique gene clusters",
        verbose=verbose
    )

    # Get the first PC for each cluster of genes
    _cluster_pc1 = np.zeros((d.shape[0], _n_l_clusts), dtype=float)
    for i in range(_n_l_clusts):
        _lidx = d.var['leiden'] == i
        _cluster_pc1[:, i] = d.X[:, _lidx] @ d.varm['PCs'][_lidx, 0]

    # SECOND ROUND OF CLUSTERING TO MERGE GENE CLUSTERS INTO PROGRAMS #

    # Spearman rho for the first PC for each cluster
    _rho_pc1 = np.abs(
        circular_rank_correlation(
            _cluster_pc1
        )
    )

    if _n_l_clusts > n_programs:
        vprint(
            f"Merging {_n_l_clusts} gene clusters into {n_programs} programs",
            verbose=verbose
        )

        # Merge clusters based on correlation distance (1 - abs(circcorr rho))
        clust_2 = AgglomerativeClustering(
            n_clusters=n_programs
        ).fit_predict(1 - _rho_pc1)

    else:
        clust_2 = {k: str(k) for k in range(_n_l_clusts)}

    # Generate a map that links cluters to programs
    clust_map = {
        str(k): str(clust_2[k])
        for k in range(_n_l_clusts)
    }
    clust_map[str(-1)] = str(-1)

    d.var[PROGRAM_KEY] = d.var['leiden'].astype(str).map(clust_map)

    # LOAD FINAL DATA INTO INITIAL DATA OBJECT AND RETURN IT #

    data.var['leiden'] = str(-1)
    data.var.loc[d.var_names, 'leiden'] = d.var['leiden'].astype(str)
    data.var[PROGRAM_KEY] = data.var['leiden'].map(clust_map)

    # ADD RESULTS OBJECT TO UNS #

    data.uns[PROGRAM_KEY] = {
        METRIC_SUBKEY: metric,
        LEIDEN_CORR_SUBKEY: _rho_pc1,
        METRIC_GENE_SUBKEY: d.var_names.values,
        METRIC_DIST_SUBKEY.format(metric=metric): dists,
        PROGRAM_CLUST_SUBKEY: clust_map,
        N_COMP_SUBKEY: n_comps,
        N_PROG_SUBKEY: n_programs,
        PROG_NAMES_SUBKEY: np.unique(data.var[PROGRAM_KEY]),
        MCV_LOSS_SUBKEY: mcv_loss_arr
    }

    if metric == 'information':
        data.uns[PROGRAM_KEY]['mutual_information'] = mutual_info

    return data


def _leiden_cluster(
    dist_array,
    n_neighbors,
    random_state=100,
    leiden_kws=None
):

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

    if knn_connect.size < int(1e6) and sparse.issparse(knn_connect):
        knn_connect = knn_connect.A

    leiden_kws = {} if leiden_kws is None else leiden_kws
    leiden_kws['adjacency'] = knn_connect
    leiden_kws['random_state'] = leiden_kws.get('random_state', random_state)

    ad_arr = ad.AnnData(dist_array.astype(float))

    sc.tl.leiden(ad_arr, **leiden_kws)

    return ad_arr.obs['leiden'].astype(int).values

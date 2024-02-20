import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sps
import numba
import tqdm
from pynndescent import PyNNDescentTransformer
from sklearn.neighbors import NearestNeighbors


from .graph import (
    set_diag,
    local_optimal_knn
)
from .math import (
    dot,
    pairwise_metric,
    array_sum,
    _csr_row_divide
)
from .misc import (
    vprint,
    standardize_data,
    is_csr
)


N_PCS = np.arange(5, 115, 10)
N_NEIGHBORS = np.arange(15, 115, 10)


def knn_noise2self(
    count_data,
    neighbors=None,
    npcs=None,
    verbose=False,
    metric='euclidean',
    loss='mse',
    loss_kwargs={},
    return_errors=False,
    connectivity=False,
    standardization_method='log',
    pc_data=None,
    chunk_size=10000,
    use_sparse=None
):
    """
    Select an optimal set of graph parameters based on noise2self

    :param count_data: Count data
    :type count_data: np.ndarray, sp.spmatrix
    :param neighbors: k values to search for global graph,
        defaults to None (5 to 105 by 10s)
    :type neighbors: np.ndarray, optional
    :param npcs: Number of PCs to use to embed graph,
        defaults to None (5 to 105 by 10s)
    :type npcs: np.ndarray, optional
    :param verbose: Verbose output to stdout,
        defaults to False
    :type verbose: bool, optional
    :param metric: Distance metric to use for k-NN graph construction,
        supports any metric sklearn Neighbors does, defaults to 'euclidean'
    :type metric: str, optional
    :param loss: Loss function for comparing reconstructed expression data to
        actual expression data, supports `mse`, `mae`, and `log_loss`, or any
        callable of the form func(x, y, **kwargs). Defaults to `mse`.
    :type loss: str, func, optional
    :param loss_kwargs: Dict of kwargs for the loss function, defalts to {}
    :type loss_kwargs: dict, optional
    :param return_errors: Return the mean square errors for global
        neighbor/nPC search, defaults to False
    :type return_errors: bool, optional
    :param use_sparse: Deprecated; always keep sparse now
    :type use_sparse: bool
    :param connectivity: Calculate row stochastic matrix from connectivity,
        not from distance
    :type connectivity: bool
    :param standardization_method: How to standardize provided count data,
        None disables. Options are `log`, `scale`, and `log_scale`. Defaults
        to 'log'.
    :type standardization_method: str, optional,
    :param pc_data: Precalculated principal components, defaults to None
    :type pc_data: np.ndarray, optional
    :return: Optimal k-NN graph
        global optimal # of PCs,
        global optimal k,
        local optimal k for each observation
    :rtype: sp.sparse.csr_matrix, int, int, np.ndarray [int]
    """

    # Get default search parameters and check dtypes
    if neighbors is None:
        neighbors = N_NEIGHBORS
    else:
        neighbors = np.asanyarray(neighbors).reshape(-1)

    if npcs is None:
        npcs = N_PCS
    else:
        npcs = np.asanyarray(npcs).reshape(-1)

    _max_pcs = np.max(npcs)

    if not np.issubdtype(neighbors.dtype, np.integer):
        raise ValueError(
            "k-NN graph requires k to be integers; "
            f"{neighbors.dtype} provided"
        )

    if not np.issubdtype(npcs.dtype, np.integer):
        raise ValueError(
            "n_pcs must be integers; "
            f"{npcs.dtype} provided"
        )

    # Check input data sizes
    if pc_data is not None and pc_data.shape[1] < _max_pcs:
        raise ValueError(
            f"Cannot search through {_max_pcs} PCs; only "
            f"{pc_data.shape[1]} components provided"
        )
    elif min(count_data.shape) < _max_pcs:
        raise ValueError(
            f"Cannot search through {_max_pcs} PCs for "
            f"data {count_data.shape} provided"
        )

    vprint(
        f"Searching {len(npcs)} PC x {len(neighbors)} Neighbors space",
        verbose=verbose
    )

    # Standardize data if necessary and create an anndata object
    # Keep separate reference to expression data and force float32
    if standardization_method is not None:
        data_obj = standardize_data(
            ad.AnnData(count_data.astype(np.float32)),
            method=standardization_method
        )
        expr_data = data_obj.X

    else:
        data_obj = ad.AnnData(
            sps.csr_matrix(count_data.shape, dtype=np.float32)
        )
        expr_data = count_data.astype(np.float32)

    if pc_data is not None:
        vprint(
            f"Using existing PCs {pc_data.shape}",
            verbose=verbose
        )
        data_obj.obsm['X_pca'] = pc_data

    else:
        vprint(
            f"Calculating {np.max(npcs)} PCs",
            verbose=verbose
        )
        data_obj.obsm['X_pca'] = sc.pp.pca(
            expr_data,
            n_comps=np.max(npcs),
            zero_center=False
        )

    mses = np.zeros((len(npcs), len(neighbors)))

    if len(npcs) > 1:
        # Create a progress bar
        tqdm_pbar = tqdm.tqdm(
            enumerate(npcs),
            postfix=f"{npcs[0]} PCs",
            bar_format='{l_bar}{bar}{r_bar}',
            total=len(npcs) * len(neighbors)
        )

        # Search for the smallest MSE for each n_pcs / k combination
        # Outer loop does PCs, because the distance graph has to be
        # recalculated when PCs changes
        for i, pc in tqdm_pbar:

            # Calculate neighbor graph with the max number of neighbors
            # Faster to select only a subset of edges than to recalculate
            # (obviously)
            _neighbor_graph(
                data_obj,
                pc,
                np.max(neighbors),
                metric=metric
            )

            # Update the progress bar
            tqdm_pbar.postfix = f"{pc} PCs Neighbor Search"
            tqdm_pbar.update(0)

            # Search through the neighbors space
            mses[i, :] = _search_k(
                expr_data,
                data_obj.obsp['distances'],
                neighbors,
                connectivity=connectivity,
                loss=loss,
                loss_kwargs=loss_kwargs,
                chunk_size=chunk_size,
                pbar=tqdm_pbar
            )

        # Get the index of the optimal PCs and k based on
        # minimizing MSE
        op_pc = np.argmin(np.min(mses, axis=1))
        op_k = np.argmin(mses[op_pc, :])

        vprint(
            f"Global optimal graph at {npcs[op_pc]} PCs "
            f"and {neighbors[op_k]} neighbors",
            verbose=verbose
        )
    else:
        vprint(
            "Skipping global optimal graph search and "
            f"using {npcs[0]} PCs",
            verbose=verbose
        )
        op_pc = 0
        op_k = None

    # Recalculate a k-NN graph from the optimal # of PCs
    _neighbor_graph(
        data_obj,
        npcs[op_pc],
        np.max(neighbors),
        metric=metric
    )

    # Search space for k-neighbors
    local_neighbors = np.arange(
        np.min(neighbors) if len(neighbors) > 1 else 1,
        np.max(neighbors)
    )

    # Search for the optimal number of k for each obs
    # For the global optimal n_pc
    local_k = local_neighbors[np.argmin(
        _search_k(
            expr_data,
            data_obj.obsp['distances'],
            local_neighbors,
            by_row=True,
            pbar=True,
            connectivity=connectivity,
            loss=loss,
            loss_kwargs=loss_kwargs,
            chunk_size=chunk_size
        ),
        axis=0
    )]

    # Pack return object:
    # Optimal (variable-k) graph
    # Optimal # of PCs
    # Optimal # of neighbors (global)
    # Optimal # of neighbors (local)
    optimals = (
        local_optimal_knn(
            data_obj.obsp['distances'],
            local_k
        ),
        npcs[op_pc],
        neighbors[op_k] if op_k is not None else None,
        local_k
    )

    if return_errors:
        return optimals, mses

    else:
        return optimals


def _neighbor_graph(adata, pc, k, metric='euclidean'):
    """
    Build neighbor graph in an AnnDaya object
    """

    if adata.n_obs < 25000:

        adata.obsp['distances'] = NearestNeighbors(
            n_neighbors=k,
            metric=metric,
            n_jobs=-1
        ).fit(adata.obsm['X_pca'][:, :pc]).kneighbors_graph()

    else:
        adata.obsp['distances'] = PyNNDescentTransformer(
            n_neighbors=k,
            n_jobs=None,
            metric=metric,
            n_trees=min(64, 5 + int(round((adata.n_obs) ** 0.5 / 20.0))),
            n_iters=max(5, int(round(np.log2(adata.n_obs)))),
        ).fit_transform(adata.obsm['X_pca'][:, :pc])

    # Enforce diagonal zeros on graph
    # Single precision floats
    set_diag(adata.obsp['distances'], 0)
    adata.obsp['distances'].data = adata.obsp['distances'].data.astype(
        np.float32
    )

    return adata


def _search_k(
    X,
    graph,
    k,
    by_row=False,
    loss='mse',
    loss_kwargs={},
    X_compare=None,
    pbar=False,
    connectivity=False,
    chunk_size=10000
):
    """
    Find optimal number of neighbors for a given graph

    :param X: Data [M x N]
    :type X: np.ndarray, sp.spmatrix
    :param graph: Graph
    :type graph: np.ndarray, sp.spmatrix
    :param k: k values to search
    :type k: np.ndarray [K]
    :param by_row: Get optimal k for each observation,
        defaults to False
    :type by_row: bool, optional
    :param pbar: Show a progress bar, defaults to False
    :type pbar: bool
    :return: Mean Squared Error for each k [K] or
        for each k and each observation [K x M]
    :rtype: np.ndarray
    """

    n, _ = X.shape
    n_k = len(k)

    X_compare = X_compare if X_compare is not None else X

    mses = np.zeros(n_k) if not by_row else np.zeros((n_k, n))

    rfunc = tqdm.trange if pbar is True else range

    if hasattr(pbar, 'postfix'):
        _postfix = pbar.postfix

    if connectivity:
        row_normalize = _connect_to_row_stochastic
    else:
        row_normalize = _dist_to_row_stochastic

    for i in rfunc(n_k):

        if hasattr(pbar, 'postfix'):
            pbar.postfix = _postfix + f" ({k[i]} N)"
            pbar.update(1)

        # Extract k non-zero neighbors from the graph
        k_graph = local_optimal_knn(
            graph.copy(),
            np.full(n, k[i]),
            keep='smallest'
        )

        # Convert to a row stochastic graph
        k_graph = row_normalize(k_graph)

        # Calculate mean squared error
        mses[i] = _noise_to_self_error(
            X,
            k_graph,
            by_row=by_row,
            metric=loss,
            chunk_size=chunk_size,
            **loss_kwargs
        )

    return mses


def _dist_to_row_stochastic(graph):

    if sps.isspmatrix(graph):

        rowsum = array_sum(graph, axis=1).astype(float)
        rowsum[rowsum == 0] = 1.

        # Dot product between inverse rowsum diagonalized
        # and graph.
        # Somehow faster then element-wise \_o_/

        if is_csr(graph):
            _csr_row_divide(
                graph.data,
                graph.indptr,
                rowsum
            )
            return graph
        else:
            return dot(
                sps.diags(
                    (1. / rowsum),
                    offsets=0,
                    shape=graph.shape,
                    format='csr',
                    dtype=graph.dtype
                ),
                graph
            )
    else:

        rowsum = graph.sum(axis=1)
        rowsum[rowsum == 0] = 1.

        return np.multiply(
            graph,
            (1 / rowsum)[:, None],
            out=graph
        )


def _connect_to_row_stochastic(graph):

    graph_dtype = graph.dtype

    if sps.issparse(graph):
        graph.data[:] = 1
    else:
        graph = graph != 0
        graph = graph.astype(graph_dtype)

    return _dist_to_row_stochastic(graph)


def _noise_to_self_error(
    X,
    k_graph,
    by_row=False,
    metric='mse',
    chunk_size=10000,
    **loss_kwargs
):

    if (metric == 'mse' and is_csr(X) and chunk_size is not None):
        _n_row = X.shape[0]
        _row_mse = np.zeros(X.shape[0], dtype=float)

        for i in range(int(np.ceil(_n_row / chunk_size))):
            _start = i * chunk_size
            _end = min(_start + chunk_size, _n_row)

            _row_mse[_start:_end] = _chunk_graph_mse(
                X,
                k_graph,
                _start,
                _end
            )

        if by_row:
            return _row_mse
        else:
            return np.mean(_row_mse)

    else:
        return pairwise_metric(
            X,
            dot(k_graph, X, dense=not sps.issparse(X)),
            by_row=by_row,
            metric=metric,
            **loss_kwargs
        )


def _chunk_graph_mse(
    X,
    k_graph,
    row_start=0,
    row_end=None
):
    if row_end is None:
        row_end = k_graph.shape[0]
    else:
        row_end = min(k_graph.shape[0], row_end)

    return _mse_rowwise(
        X.data,
        X.indices,
        X.indptr,
        dot(
            k_graph[row_start:row_end, :],
            X,
            dense=True
        )
    )


@numba.njit(parallel=True)
def _mse_rowwise(
    a_data,
    a_indices,
    a_indptr,
    b
):

    n_row = b.shape[0]
    output = np.zeros(n_row, dtype=float)

    for i in numba.prange(n_row):

        _idx_a = a_indices[a_indptr[i]:a_indptr[i + 1]]
        _nnz_a = _idx_a.shape[0]

        row = b[i, :]

        if _nnz_a == 0:
            pass

        else:
            row = row.copy()
            row[_idx_a] -= a_data[a_indptr[i]:a_indptr[i + 1]]

        output[i] = np.mean(row ** 2)

    return output

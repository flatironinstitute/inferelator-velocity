import numpy as np
import anndata as ad
import scanpy as sc
import scipy.sparse as sps
import tqdm

from .graph import set_diag, local_optimal_knn
from .math import mean_squared_error
from .misc import vprint

try:
    from sparse_dot_mkl import dot_product_mkl as dot

except ImportError as err:

    import warnings

    warnings.warn(
        "Unable to use MKL for sparse matrix math, "
        "defaulting to numpy/scipy matmul: "
        f"{str(err)}"
    )

    def dot(x, y, dense=False, cast=False):

        z = x @ y

        if dense and sps.issparse(z):
            z = z.A

        return z


N_PCS = np.arange(5, 115, 10)
N_NEIGHBORS = np.arange(15, 115, 10)


def knn_noise2self(
    count_data,
    neighbors=None,
    npcs=None,
    verbose=False,
    metric='euclidean',
    return_errors=False,
    use_sparse=True
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
    :param metric: Distance metric to use, defaults to 'euclidean'
    :type metric: str, optional
    :param return_errors: Return the mean square errors for global
        neighbor/nPC search, defaults to False
    :type return_errors: bool, optional
    :return: Optimal k-NN graph
        global optimal # of PCs,
        global optimal k,
        local optimal k for each observation
    :rtype: sp.sparse.csr_matrix, int, int, np.ndarray [int]
    """

    neighbors = N_NEIGHBORS if neighbors is None else neighbors
    npcs = N_PCS if npcs is None else npcs

    vprint(f"Searching {len(npcs)} PC x {len(neighbors)} Neighbors space",
           verbose=verbose)

    data_obj = ad.AnnData(count_data, dtype=np.float32)

    sc.pp.normalize_per_cell(data_obj)
    sc.pp.log1p(data_obj)
    sc.pp.pca(data_obj, n_comps=np.max(npcs), zero_center=True)

    mses = np.zeros((len(npcs), len(neighbors)))

    expr_data = data_obj.X if use_sparse or not sps.issparse(data_obj.X) else data_obj.X.A

    # Search for the smallest MSE for each n_pcs / k combination
    for i, pc in tqdm.tqdm(enumerate(npcs)):
        tqdm.tqdm.write(f"Searching graphs from {pc} PCs")

        sc.pp.neighbors(
            data_obj,
            n_neighbors=np.max(neighbors),
            n_pcs=pc,
            metric=metric
        )

        # Enforce diagonal zeros on graph
        # Single precision floats
        set_diag(data_obj.obsp['distances'], 0)
        data_obj.obsp['distances'] = data_obj.obsp['distances'].astype(np.float32)

        mses[i, :] = _search_k(
            expr_data,
            data_obj.obsp['distances'],
            neighbors,
            X_compare=expr_data
        )

    op_pc = np.argmin(np.min(mses, axis=1))
    op_k = np.argmin(mses[op_pc, :])

    vprint(
        f"Global optimal graph at {npcs[op_pc]} PCs "
        f"and {neighbors[op_k]} neighbors",
        verbose=verbose
    )

    sc.pp.neighbors(
        data_obj,
        n_neighbors=np.max(neighbors),
        n_pcs=npcs[op_pc],
        metric=metric
    )

    # Enforce diagonal zeros on graph
    # Single precision floats
    set_diag(data_obj.obsp['distances'], 0)
    data_obj.obsp['distances'] = data_obj.obsp['distances'].astype(np.float32)

    local_neighbors = np.arange(np.min(neighbors), np.max(neighbors))

    # Search for the optimal number of k for each obs
    # For the global optimal n_pc
    local_k = local_neighbors[np.argmin(
        _search_k(
            expr_data,
            data_obj.obsp['distances'],
            local_neighbors,
            by_row=True,
            X_compare=expr_data
        ),
        axis=0
    )]

    optimals = (
        local_optimal_knn(data_obj.obsp['distances'], local_k),
        npcs[op_pc],
        neighbors[op_k],
        local_k
    )

    if return_errors:
        return optimals, mses

    else:
        return optimals


def _search_k(
    X,
    graph,
    k,
    by_row=False,
    X_compare=None
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
    :return: Mean Squared Error for each k [K] or
        for each k and each observation [K x M]
    :rtype: np.ndarray
    """

    n, _ = X.shape
    n_k = len(k)

    X_compare = X_compare if X_compare is not None else X

    mses = np.zeros(n_k) if not by_row else np.zeros((n_k, n))

    for i in tqdm.trange(n_k):

        # Extract k non-zero neighbors from the graph
        k_graph = local_optimal_knn(
            graph.copy(),
            np.full(n, k[i]),
            keep='smallest'
        )

        # Convert to a row stochastic graph
        k_graph = _dist_to_row_stochastic(
            k_graph
        )

        # Calculate mean squared error
        mses[i] = mean_squared_error(
            X_compare,
            dot(k_graph, X, dense=not sps.issparse(X_compare)),
            by_row=by_row
        )

    return mses


def _dist_to_row_stochastic(graph):

    if sps.issparse(graph):

        rowsum = graph.sum(axis = 1).A1
        rowsum[rowsum == 0] = 1.

        # Dot product between inverse rowsum diagonalized
        # and graph.
        # Somehow faster then element-wise \_o_/
        return dot(
            sps.diags(
                (1 / rowsum),
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

        return np.multiply(graph, (1 / rowsum)[:, None])

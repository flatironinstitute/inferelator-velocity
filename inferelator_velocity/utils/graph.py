import numpy as np
import numba
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import (
    issparse as _is_sparse,
    isspmatrix_csr as _is_csr,
    csr_matrix as _csr
)
import itertools


def get_shortest_paths(graph, select_nodes, graph_method="D"):
    """
    Find the pairwise shortest path between specific nodes in
    an undirected graph

    :param graph: N x N graph connecting N nodes
    :type graph: np.ndarray, sp.spmatrix
    :param select_nodes: Indices of nodes to connect
    :type select_nodes: np.ndarray, list
    :param graph_method: _description_, defaults to "D"
    :type graph_method: str, optional
    :return: Array of shortest-path lists
    :rtype: np.ndarray [list] [N x N]
    """

    # Get the predecessor array from all data points to each node
    _, graph_pred = shortest_path(
        graph,
        directed=False,
        indices=select_nodes,
        return_predecessors=True,
        method=graph_method
    )

    _n_nodes = len(select_nodes)

    # Build an N x N array of lists
    shortest_paths = np.zeros((_n_nodes, _n_nodes), dtype=object)

    # Add 1-len lists to the diagonal
    for i in range(_n_nodes):
        shortest_paths[i, i] = [select_nodes[i]]

    # For every combination of nodes
    for end_idx, start_idx in itertools.combinations(range(_n_nodes), 2):

        # Find the endpoint at the left node
        end = select_nodes[end_idx]

        # Find the start point at the right node
        current_loc = select_nodes[start_idx]
        path = [current_loc]

        # While the current position is different from the end position
        # Walk backwards through the predecessor array
        # Putting each location in the path list
        try:
            while current_loc != end:
                current_loc = graph_pred[end_idx, current_loc]
                path.append(current_loc)
        except IndexError:
            raise RuntimeError("Graph is not fully connected; pathing failed")

        # Put the list in the output array in the correct position
        # And then reverse it for the other direction
        shortest_paths[end_idx, start_idx] = path[::-1]
        shortest_paths[start_idx, end_idx] = path

    return shortest_paths


def get_total_path(shortest_paths, centroid_order_dict, centroid_order_list):
    """
    Take an array of shortest paths between key nodes and
    find the total path that connects all key nodes

    :param shortest_paths: [N x N] Array of lists, where the list is the
        shortest path connecting key nodes
    :type shortest_paths: np.ndarray [list] [N x N]
    :param centroid_order_dict: Dict keyed by node labels. Values are
        ('next_node_label', time_left_node, time_right_node)
    :type centroid_order_dict: dict
    :param centroid_order_list: Node labels for shortest_paths array
    :type centroid_order_list: np.ndarray, list
    :return: A list of nodes that connect every key node
        A dict keyed by key node label with the position of that label
        on the total path list
    :rtype: list, dict
    """

    total_path = []
    total_path_centroids = {}

    # Path goes from left to right node
    for i, (start_label, (end_label, _, _)) in enumerate(
        centroid_order_dict.items()
    ):

        # Get the shortest path that starts at left and ends at right
        _link_path = shortest_paths[
            [start_label == x for x in centroid_order_list],
            [end_label == x for x in centroid_order_list]
        ][0]

        # Set the position of the key node on the total path
        total_path_centroids[start_label] = max(len(total_path) - 1, 0)
        total_path.extend(_link_path[int(i > 0):])

        if end_label not in total_path_centroids:
            total_path_centroids[end_label] = len(total_path) - 1

    return total_path, total_path_centroids


def local_optimal_knn(
    neighbor_graph,
    nn_vector,
    keep='smallest'
):
    """
    Modify a k-NN graph *in place* to have a specific number of
    non-zero values k per row based on a vector of k-values

    :param neighbor_graph: N x N matrix with edge values.
    :type neighbor_graph: np.ndarray, sp.sparse.csr_matrix
    :param nn_vector: Vector of `k` values per row
    :type nn_vector: np.ndarray
    :param keep: Keep the 'largest' or 'smallest' non-zero values
    :type keep: str
    :raise ValueError: Raise a ValueError if neighbor_graph is a
        non-CSR sparse matrix or if keep is not 'smallest' or
        'largest'
    :return: Return a reference to neighbor_graph
    :rtype: np.ndarray, sp.sparse.csr_matrix
    """

    n, m = neighbor_graph.shape

    neighbor_sparse = _is_sparse(neighbor_graph)

    if neighbor_sparse and not _is_csr(neighbor_graph):
        raise ValueError("Sparse matrices must be CSR")

    if n != len(nn_vector):
        raise ValueError(
            f"{len(nn_vector)}-length vector wrong size "
            f" for graph {neighbor_graph.shape}"
        )

    # Define functions that select the inverse of
    # the k largest or smallest values
    # So those values can be zeroed
    if keep == 'smallest':
        def _nn_slice(k, n):
            return slice(k, None)

    elif keep == 'largest':
        def _nn_slice(k, n):
            return slice(None, n - k)
    else:
        raise ValueError("keep must be 'smallest' or 'largest'")

    _smallest = keep == 'smallest'

    if neighbor_sparse:
        _shrink_sparse_graph_k(
            neighbor_graph.data,
            neighbor_graph.indptr,
            np.asanyarray(nn_vector),
            smallest=_smallest
        )

        neighbor_graph.eliminate_zeros()

        return neighbor_graph

    for i in range(n):

        k = nn_vector[i]

        if k >= n:
            continue

        n_slice = neighbor_graph[i, :]

        # Use a masked array to block out zeros
        droppers = np.ma.masked_array(
            n_slice,
            mask=n_slice == 0
        ).argsort(endwith=_smallest)[_nn_slice(k, m)]

        # Write the new data into the original array
        neighbor_graph[i, droppers] = 0.

    return neighbor_graph


def set_diag(X, diag):
    """
    Set diagonal of matrix X.
    Safe for dense or sparse matrices

    :param X: Numeric matrix
    :type X: np.ndarray, sp.spmatrix
    :param diag: Value to fill
    :type diag: Numeric
    :return: Numeric matrix with set diagonal
    :rtype: np.ndarray, sp.spmatrix
    """

    if diag is None:
        pass

    elif _is_sparse(X):
        # UGLY HACK TO AVOID SPARSITY CHANGES #
        _remove_diag = X.diagonal()

        if np.all(_remove_diag == 0.):
            return X

        _remove_idx = np.arange(len(_remove_diag))

        X -= _csr(
            (_remove_diag, (_remove_idx, _remove_idx)),
            shape=X.shape
        )

    else:
        np.fill_diagonal(X, diag)

    return X


@numba.njit(parallel=True)
def _shrink_sparse_graph_k(
    graph_data,
    graph_indptr,
    k_vec,
    smallest=True
):

    n = graph_indptr.shape[0] - 1

    for i in numba.prange(n):

        _left = graph_indptr[i]
        _right = graph_indptr[i+1]
        _n = _right - _left

        k = k_vec[i]

        if _n <= k:
            pass

        else:
            _data = graph_data[_left:_right]

            if smallest:
                _data[np.argsort(_data)[k - _n:]] = 0
            else:
                _data[np.argsort(_data)[:_n - k]] = 0

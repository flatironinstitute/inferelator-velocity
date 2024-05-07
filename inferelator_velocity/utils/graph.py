import numpy as np
from scipy.sparse.csgraph import shortest_path
from umap.umap_ import nearest_neighbors
from scself._noise2self.graph import local_optimal_knn

import itertools

try:
    from scanpy.neighbors import _compute_connectivities_umap as _connect_umap
except ImportError:
    from scanpy.neighbors._connectivity import umap as _scanpy_connect

    def _connect_umap(i, d, **kwargs):

        return d, _scanpy_connect(
            i, d, **kwargs
        )


def compute_neighbors(
    array,
    n_neighbors,
    random_state,
    metric='precomputed',
    **metric_kwargs
):

    knn_i, knn_dist, _ = nearest_neighbors(
        array,
        n_neighbors,
        random_state=random_state,
        metric=metric,
        metric_kwds=metric_kwargs,
        n_jobs=-1,
        angular=False
    )

    knn_dist, knn_connect = _connect_umap(
        knn_i, knn_dist, n_obs=array.shape[0], n_neighbors=n_neighbors
    )

    return knn_dist, knn_connect


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

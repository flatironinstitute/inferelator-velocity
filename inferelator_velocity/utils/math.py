import numpy as np


def scalar_projection(data, center_point, off_point, normalize=True, weights=None):
    """
    Scalar projection of data onto a line defined by two points

    :param data: Data
    :type data: np.ndarray
    :param center_point: Integer index of starting point for line
    :type center_point: int
    :param off_point: Integer index of ending point for line
    :type off_point: int
    :param normalize: Normalize distance between start and end of line to 1,
        defaults to True
    :type normalize: bool, optional
    :param weights: How much weight to put on each dimension, defaults to None
    :type weights: np.ndarray, optional
    :return: Scalar projection array
    :rtype: np.ndarray
    """

    data = data - data[center_point, :]

    if weights is not None:
        data = np.multiply(data, weights[None, :])

    vec = data[off_point, :]

    # Calculate scalar projection: (data (dot) vec) / ||vec||
    scalar_proj = np.dot(data, vec) / np.linalg.norm(vec)

    # Normalize projection such that vec is unit length
    if normalize:
        scalar_proj = scalar_proj / np.linalg.norm(vec)

    return scalar_proj


def get_centroids(comps, cluster_vector):
    return {k: _get_centroid(comps[cluster_vector == k, :])
            for k in np.unique(cluster_vector)}


def _get_centroid(comps):
    return np.sum((comps - np.mean(comps, axis=0)[None, :]) ** 2, axis=1).argmin()

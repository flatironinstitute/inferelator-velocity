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

    vec = data[off_point, :] - data[center_point, :]
    data = data - data[center_point, :]

    if weights is not None:
        vec *= weights
        data = np.multiply(data, weights[None, :])

    scalar_proj = np.dot(data, vec) / np.linalg.norm(vec)

    if normalize:
        _center_scale = scalar_proj[center_point]
        _off_scale = scalar_proj[off_point]
        scalar_proj = (scalar_proj - _center_scale) / (_off_scale - _center_scale)

    return scalar_proj


def get_centroids(comps, cluster_vector):
    return {k: _get_centroid(comps[cluster_vector == k, :])
            for k in np.unique(cluster_vector)}


def _get_centroid(comps):
    return np.sum((comps - np.mean(comps, axis=0)[None, :]) ** 2, axis=1).argmin()

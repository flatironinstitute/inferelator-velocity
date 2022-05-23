import numpy as np
from .misc import make_vector_2D

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


def least_squares(x, y):
    """
    OLS wrapper that also calculates SE of slope estimate

    :param x: Design vector
    :type x: np.ndarray
    :param y: Response vector
    :type y: np.ndarray
    :return: OLS beta estimate, OLS beta SE estimate
    :rtype: float, float
    """

    if x.shape[0] == 0:
        return 0., 0.

    x = make_vector_2D(x)
    y = make_vector_2D(y)

    slope = np.linalg.lstsq(x, y, rcond=None)[0][0][0]

    return slope, _calc_se(x, y, slope)


def _calc_se(x, y, slope):

    if x.shape[0] == 0:
        return 0

    mse_x = np.sum(np.square(x - np.nanmean(x)))
    if mse_x == 0:
        return 0

    elif slope == 0:
        return np.mean(np.square(y - np.nanmean(y))) / mse_x

    else:
        mse_y = np.sum(np.square(y - np.dot(x, slope)))
        se_y = mse_y / (len(y) - 1)
        return se_y / mse_x

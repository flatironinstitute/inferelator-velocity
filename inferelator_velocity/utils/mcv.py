import tqdm
import numpy as np
import scipy.sparse as sps
import scanpy as sc
import anndata as ad

from inferelator_velocity.utils.misc import (
    standardize_data
)
from inferelator_velocity.utils.math import (
    pairwise_metric,
    mcv_mse
)


def mcv_pcs(
    count_data,
    n=5,
    n_pcs=100,
    random_seed=800,
    p=0.5,
    metric='mse',
    standardization_method='log',
    metric_kwargs={}
):
    """
    Calculate a loss metric based on molecular crossvalidation

    :param count_data: Integer count data
    :type count_data: np.ndarray, sp.sparse.csr_matrix, sp.sparse.csc_matrix
    :param n: Number of crossvalidation resplits, defaults to 5
    :type n: int, optional
    :param n_pcs: Number of PCs to search, defaults to 100
    :type n_pcs: int, optional
    :param random_seed: Random seed for split, defaults to 800
    :type random_seed: int, optional
    :param p: Split probability, defaults to 0.5
    :type p: float, optional
    :param metric: Metric to use - accepts 'mse', 'mae', and 'r2' as strings,
        or any callable of the type metric(pred, true), defaults to 'mse'
    :type metric: str, func, optional
    :return: An n x n_pcs array of metric results
    :rtype: np.ndarray
    """

    n_pcs = min(n_pcs, *map(lambda x: x - 1, count_data.shape))
    size = count_data.shape[0] * count_data.shape[1]

    metric_arr = np.zeros((n, n_pcs + 1), dtype=float)

    # Use a single progress bar for nested loop
    with tqdm.tqdm(total=n * (n_pcs + 1)) as pbar:

        for i in range(n):
            A, B, n_counts = _molecular_split(
                count_data,
                random_seed=random_seed,
                p=p
            )

            A = standardize_data(
                A,
                target_sum=n_counts,
                method=standardization_method
            )

            B = standardize_data(
                B,
                target_sum=n_counts,
                method=standardization_method
            )

            # Calculate PCA
            sc.pp.pca(
                A,
                n_comps=n_pcs
            )

            # Null model (no PCs)

            if sps.issparse(B.X):
                metric_arr[i, 0] = np.sum(B.X.data ** 2) / size
            else:
                metric_arr[i, 0] = np.sum(B.X ** 2) / size

            pbar.update(1)

            # Calculate metric for 1-n_pcs number of PCs
            for j in range(1, n_pcs + 1):
                metric_arr[i, j] = mcv_comp(
                    B.X,
                    A.obsm['X_pca'][:, 0:j],
                    A.varm['PCs'][:, 0:j].T,
                    metric=metric,
                    **metric_kwargs
                )
                pbar.update(1)

    return metric_arr


def _molecular_split(count_data, random_seed=800, p=0.5):
    """
    Break an integer count matrix into two count matrices.
    These will sum to the original count matrix and are
    selected randomly from the binomial distribution

    :param count_data: Integer count data
    :type count_data: np.ndarray, sp.sparse.csr_matrix, sp.sparse.csc_matrix
    :param random_seed: Random seed for generator, defaults to 800
    :type random_seed: int, optional
    :param p: Split probability, defaults to 0.5
    :type p: float, optional
    :return: Two count matrices A & B of the same type as the input count_data,
        where A + B = count_data
    :rtype: np.ndarray or sp.sparse.csr_matrix or sp.sparse.csc_matrix
    """

    rng = np.random.default_rng(random_seed)

    if sps.issparse(count_data):

        normalization_depth = np.median(
            count_data.sum(axis=1).A1
        )

        if sps.isspmatrix_csr(count_data):
            mat_func = sps.csr_matrix
        else:
            mat_func = sps.csc_matrix

        cv_data = mat_func((
            rng.binomial(count_data.data, p=p),
            count_data.indices,
            count_data.indptr),
            shape=count_data.shape
        )

        count_data = mat_func((
            count_data.data - cv_data.data,
            count_data.indices,
            count_data.indptr),
            shape=count_data.shape
        )

    else:

        normalization_depth = np.median(
            count_data.sum(axis=1)
        )

        cv_data = np.zeros_like(count_data)

        for i in range(count_data.shape[0]):
            cv_data[i, :] = rng.binomial(count_data[i, :], p=p)

        count_data = count_data - cv_data

    count_data = ad.AnnData(count_data)
    cv_data = ad.AnnData(cv_data)

    return count_data, cv_data, normalization_depth


def mcv_comp(x, pc, rotation, metric, **metric_kwargs):

    if metric != 'mse':
        return pairwise_metric(
            x,
            pc @ rotation,
            metric=metric,
            **metric_kwargs
        )
    else:
        return mcv_mse(
            x,
            pc,
            rotation,
            **metric_kwargs
        )

import warnings
from functools import partial

import numpy as np
from scipy import sparse, linalg
from scipy.sparse.linalg import svds

from sparse_dot_mkl import dot_product_mkl

from sklearn.decomposition import TruncatedSVD
from sklearn.utils import check_random_state
from sklearn.utils._array_api import get_namespace, device
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.extmath import svd_flip
from sklearn.utils._arpack import _init_arpack_v0
from sklearn.utils.validation import check_is_fitted


#################################################
#                                               #
#   This code is taken directly from sklearn    #
#   and modified to use the intel MKL for       #
#   sparse matrix multiplication                #
#                                               #
#   Code in this file is licensed under BSD     #
#   3-clause for compatibility with sklearn     #
#                                               #
#################################################

def safe_sparse_dot(a, b, *, dense_output=False):

    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = dot_product_mkl(a, b_2d, dense=True)
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = dot_product_mkl(a_2d, b, dense=True)
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)

    # Special check to make sure b is contiguous
    if not sparse.issparse(b):
        ret = dot_product_mkl(a, np.ascontiguousarray(b), dense=dense_output)
    else:
        ret = dot_product_mkl(a, b, dense=dense_output)

    if (
        sparse.issparse(a)
        and sparse.issparse(b)
        and dense_output
        and hasattr(ret, "toarray")
    ):
        return ret.toarray()

    return ret


def randomized_range_finder(
    A, *, size, n_iter, power_iteration_normalizer="auto", random_state=None
):

    xp, is_array_api_compliant = get_namespace(A)
    random_state = check_random_state(random_state)

    # Generating normal random vectors with shape: (A.shape[1], size)
    # XXX: generate random number directly from xp if it's possible
    # one day.
    Q = xp.asarray(random_state.normal(size=(A.shape[1], size)))
    if hasattr(A, "dtype") and xp.isdtype(A.dtype, kind="real floating"):
        # Use float32 computation and components if A has a float32 dtype.
        Q = xp.astype(Q, A.dtype, copy=False)

    # Move Q to device if needed only after converting to float32 if needed to
    # avoid allocating unnecessary memory on the device.

    # Note: we cannot combine the astype and to_device operations in one go
    # using xp.asarray(..., dtype=dtype, device=device) because downcasting
    # from float64 to float32 in asarray might not always be accepted as only
    # casts following type promotion rules are guarateed to work.
    # https://github.com/data-apis/array-api/issues/647
    if is_array_api_compliant:
        Q = xp.asarray(Q, device=device(A))

    # Deal with "auto" mode
    if power_iteration_normalizer == "auto":
        if n_iter <= 2:
            power_iteration_normalizer = "none"
        elif is_array_api_compliant:
            # XXX: https://github.com/data-apis/array-api/issues/627
            warnings.warn(
                "Array API does not support LU factorization, falling back to QR"
                " instead. Set `power_iteration_normalizer='QR'` explicitly to silence"
                " this warning."
            )
            power_iteration_normalizer = "QR"
        else:
            power_iteration_normalizer = "LU"
    elif power_iteration_normalizer == "LU" and is_array_api_compliant:
        raise ValueError(
            "Array API does not support LU factorization. Set "
            "`power_iteration_normalizer='QR'` instead."
        )

    if is_array_api_compliant:
        qr_normalizer = partial(xp.linalg.qr, mode="reduced")
    else:
        # Use scipy.linalg instead of numpy.linalg when not explicitly
        # using the Array API.
        qr_normalizer = partial(linalg.qr, mode="economic")

    if power_iteration_normalizer == "QR":
        normalizer = qr_normalizer
    elif power_iteration_normalizer == "LU":
        normalizer = partial(linalg.lu, permute_l=True)
    else:
        normalizer = lambda x: (x, None)

    # Perform power iterations with Q to further 'imprint' the top
    # singular vectors of A in Q
    for _ in range(n_iter):
        Q, _ = normalizer(
            safe_sparse_dot(
                A,
                Q,
                dense_output=True
            )
        )
        Q, _ = normalizer(
            safe_sparse_dot(
                np.ascontiguousarray(Q.T),
                A,
                dense_output=True
            ).T
        )

    # Sample the range of A using by linear projection of Q
    # Extract an orthonormal basis
    Q, _ = qr_normalizer(safe_sparse_dot(A, Q, dense_output=True))

    return Q


def randomized_svd(
    M,
    n_components,
    *,
    n_oversamples=10,
    n_iter="auto",
    power_iteration_normalizer="auto",
    transpose="auto",
    flip_sign=True,
    random_state=None,
    svd_lapack_driver="gesdd",
):
    if sparse.issparse(M) and M.format in ("lil", "dok"):
        warnings.warn(
            "Calculating SVD of a {} is expensive. "
            "csr_matrix is more efficient.".format(type(M).__name__),
            sparse.SparseEfficiencyWarning,
        )

    random_state = check_random_state(random_state)
    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    if n_iter == "auto":
        # Checks if the number of iterations is explicitly specified
        # Adjust n_iter. 7 was found a good compromise for PCA. See #5299
        n_iter = 7 if n_components < 0.1 * min(M.shape) else 4

    if transpose == "auto":
        transpose = n_samples < n_features
    if transpose:
        # this implementation is a bit faster with smaller shape[1]
        M = M.T

    Q = randomized_range_finder(
        M,
        size=n_random,
        n_iter=n_iter,
        power_iteration_normalizer=power_iteration_normalizer,
        random_state=random_state,
    )

    # project M to the (k + p) dimensional space using the basis vectors
    B = safe_sparse_dot(Q.T, M, dense_output=True)

    # compute the SVD on the thin matrix: (k + p) wide
    xp, is_array_api_compliant = get_namespace(B)
    if is_array_api_compliant:
        Uhat, s, Vt = xp.linalg.svd(B, full_matrices=False)
    else:
        # When when array_api_dispatch is disabled, rely on scipy.linalg
        # instead of numpy.linalg to avoid introducing a behavior change w.r.t.
        # previous versions of scikit-learn.
        Uhat, s, Vt = linalg.svd(
            B, full_matrices=False, lapack_driver=svd_lapack_driver
        )
    del B
    U = safe_sparse_dot(Q, Uhat, dense_output=True)

    if flip_sign:
        if not transpose:
            U, Vt = svd_flip(U, Vt)
        else:
            # In case of transpose u_based_decision=false
            # to actually flip based on u and not v.
            U, Vt = svd_flip(U, Vt, u_based_decision=False)

    if transpose:
        # transpose back the results according to the input convention
        return Vt[:n_components, :].T, s[:n_components], U[:, :n_components].T
    else:
        return U[:, :n_components], s[:n_components], Vt[:n_components, :]


class TruncatedSVDMKL(TruncatedSVD):

    def fit_transform(self, X, y=None):
        X = self._validate_data(
            X,
            accept_sparse=["csr", "csc"],
            ensure_min_features=2,
            dtype=[np.float64, np.float32, float]
        )
        random_state = check_random_state(self.random_state)

        if self.algorithm == "arpack":
            v0 = _init_arpack_v0(min(X.shape), random_state)
            U, Sigma, VT = svds(X, k=self.n_components, tol=self.tol, v0=v0)
            # svds doesn't abide by scipy.linalg.svd/randomized_svd
            # conventions, so reverse its outputs.
            Sigma = Sigma[::-1]
            U, VT = svd_flip(U[:, ::-1], VT[::-1])

        elif self.algorithm == "randomized":
            if self.n_components > X.shape[1]:
                raise ValueError(
                    f"n_components({self.n_components}) must be <="
                    f" n_features({X.shape[1]})."
                )
            U, Sigma, VT = randomized_svd(
                X,
                self.n_components,
                n_iter=self.n_iter,
                n_oversamples=self.n_oversamples,
                power_iteration_normalizer=self.power_iteration_normalizer,
                random_state=random_state,
            )

        self.components_ = VT

        # As a result of the SVD approximation error on X ~ U @ Sigma @ V.T,
        # X @ V is not the same as U @ Sigma
        if self.algorithm == "randomized" or (
            self.algorithm == "arpack" and self.tol > 0
        ):
            X_transformed = safe_sparse_dot(X, self.components_.T, dense_output=True)
        else:
            X_transformed = U * Sigma

        # Calculate explained variance & explained variance ratio
        self.explained_variance_ = exp_var = np.var(X_transformed, axis=0)
        if sparse.issparse(X):
            _, full_var = mean_variance_axis(X, axis=0)
            full_var = full_var.sum()
        else:
            full_var = np.var(X, axis=0).sum()
        self.explained_variance_ratio_ = exp_var / full_var
        self.singular_values_ = Sigma  # Store the singular values.

        return X_transformed

    def transform(self, X):
        """Perform dimensionality reduction on X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Reduced version of X. This will always be a dense array.
        """
        check_is_fitted(self)
        X = self._validate_data(X, accept_sparse=["csr", "csc"], reset=False)
        return safe_sparse_dot(X, self.components_.T, dense_output=True)

import numpy as np
import scipy.stats
from tqdm import trange, tqdm

from .utils.math import least_squares
from .utils.misc import get_bins

MAX_ITER = 100
TOL = 1e-6

DECAY_QUANT = (0.0, 0.05)
ALPHA_QUANT = 0.975


def calc_decay_sliding_windows(expression_data, velocity_data, time_data, n_windows=None, centers=None, width=None,
                               include_alpha=True, bootstrap_estimates=False, **kwargs):
    """
    Calculate decay constants in a sliding window across a time axis

    :param expression_data: Gene expression data [N Observations x M Genes]
    :type expression_data: np.ndarray (float)
    :param velocity_data: Gene velocity data [N Observations x M Genes]
    :type velocity_data: np.ndarray (float)
    :param time_data: Time data [N Observations, ]
    :type time_data: np.ndarray (float)
    :param n_windows: Number of windows on the time_data axis, defaults to None.
        Either this or centers & width must be set
    :type n_windows: integer, optional
    :param centers: Center points for the windows, defaults to None
    :type centers: np.ndarray, optional
    :param width: Width of the windows, defaults to None
    :type width: float, optional
    :param include_alpha: Include estimates of alpha parameter, defaults to True
    :type include_alpha: bool, optional
    :param bootstrap_estimates: Bootstrap estimates of decay & confidence interval, defaults to False
    :type bootstrap_estimates: bool, optional
    :raises ValueError: Raises ValueError when the wrong combination of kwargs is set
    :return: Returns
    :rtype: _type_
    """

    n, m = expression_data.shape

    centers, half_width = get_bins(
        time_data,
        n_bins=n_windows,
        centers=centers,
        width=width
    )

    def _calc_window_decay(center):
        lowend, highend = center - half_width, center + half_width

        keep_idx = (time_data >= lowend) & (time_data <= highend) & ~np.isnan(time_data)

        if np.sum(keep_idx) < 2:
            return (np.full(expression_data.shape[1], np.nan),
                    np.full(expression_data.shape[1], np.nan),
                    np.full(expression_data.shape[1], np.nan) if include_alpha else None)

        decay_func = calc_decay_bootstraps if bootstrap_estimates else calc_decay

        return decay_func(expression_data[keep_idx, :],
                          velocity_data[keep_idx, :],
                          lstatus=False,
                          include_alpha=include_alpha,
                          **kwargs)

    results = [_calc_window_decay(x) for x in tqdm(centers)]
    return [x[0] for x in results], [x[1] for x in results], [x[2] for x in results], centers


def calc_decay_bootstraps(expression_data, velocity_data, n_bootstraps=15, bootstrap_ratio=1.0,
                          random_seed=8675309, lstatus=True, confidence_interval = 0.95,
                          **kwargs):
    """
    Estimate decay constant lambda for dX/dt = -lambda X + alpha and calculate
    confidence intervals by bootstrapping.

    :param expression_data: Gene expression data [N Observations x M Genes]
    :type expression_data: np.ndarray (float)
    :param velocity_data: Gene velocity data [N Observations x M Genes]
    :type velocity_data: np.ndarray (float)
    :param n_bootstraps: Number of bootstraps, defaults to 15
    :type n_bootstraps: int, optional
    :param bootstrap_ratio: Fraction of samples to select for each bootstrap,
        defaults to 1.0
    :type bootstrap_ratio: float, optional
    :param random_seed: Seed for bootstrapping RNG, defaults to 8675309
    :type random_seed: int, optional
    :param lstatus: Display status bar, defaults to True
    :type lstatus: bool, optional
    :param confidence_interval: Confidence interval between 0 and 1, defaults to 0.95
    :type confidence_interval: float, optional
    """

    if n_bootstraps < 2:
        raise ValueError(f'n_bootstraps must be > 1, {n_bootstraps} provided')

    lstatus = trange if lstatus else range
    rng = np.random.RandomState(seed=random_seed)

    # Number to select per bootstrap
    # Minimum of 1
    n_to_choose = max(1, int(bootstrap_ratio * expression_data.shape[0]))

    def _calc_boot():
        pick_idx = rng.choice(np.arange(expression_data.shape[0]), size=n_to_choose)
        return calc_decay(expression_data[pick_idx, :],
                          velocity_data[pick_idx, :],
                          lstatus=False, **kwargs)

    bootstrap_results = [_calc_boot() for _ in range(n_bootstraps)]
    decays = np.vstack([x[0] for x in bootstrap_results])
    alphas = np.vstack([x[2] for x in bootstrap_results]) if bootstrap_results[0][2] is not None else None

    t = scipy.stats.t.ppf((1 + confidence_interval) / 2, n_bootstraps - 1)
    ci = t * np.nanstd(decays, axis=0) / np.sqrt(n_bootstraps)

    return np.nanmean(decays, axis=0), ci, alphas


def calc_decay(expression_data, velocity_data, include_alpha=True,
               decay_quantiles=DECAY_QUANT, alpha_quantile=ALPHA_QUANT,
               lstatus=True):
    """
    Estimate decay constant lambda for dX/dt = -lambda X + alpha

    :param expression_data: Gene expression data [N Observations x M Genes]
    :type expression_data: np.ndarray (float)
    :param velocity_data: Gene velocity data [N Observations x M Genes]
    :type velocity_data: np.ndarray (float)
    :param decay_quantiles: The quantile of observations to fit lambda,
        defaults to (0.00, 0.05)
    :type decay_quantiles: tuple, optional
    :param alpha_quantile: The quantile of observations to estimate alpha,
        defaults to 0.975
    :type alpha_quantile: float, optional
    :param lstatus: Display status bar, defaults to True
    :type lstatus: bool, optional
    :raises ValueError: Raises a ValueError if arguments are invalid
    :return: Returns estimates for lambda [M,],
        standard error of lambda estimate [M,],
        and estimates of alpha [M,]
    :rtype: np.ndarray, np.ndarray, np.ndarray
    """

    lstatus = trange if lstatus else range

    if expression_data.shape != velocity_data.shape:
        raise ValueError(f"Expression data {expression_data.shape} ",
                         f"and velocity data {velocity_data.shape} ",
                         "are not the same size")

    if ((len(decay_quantiles) != 2) or not isinstance(decay_quantiles, (tuple, list))):
        raise ValueError(f"decay_quantiles must be a tuple of two floats; {decay_quantiles} passed")

    n, m = expression_data.shape

    # Estimate parameters for each gene individually
    results = np.array([
        _estimate_for_gene(
            expression_data[:, i],
            velocity_data[:, i],
            decay_quantiles,
            alpha_quantile=alpha_quantile if include_alpha else None,
        ) for i in lstatus(m)
        ])

    return results[:, 0] * -1, results[:, 2], results[:, 1]


def _estimate_for_gene(
    expression_data,
    velocity_data,
    decay_quantiles,
    alpha_quantile=None,
):
    """
    Estimate parameters for a single gene,
    based on dx/dt = - \\lambda * x + \\alpha,
    where \\lambda and \\alpha > 0

    :param expression_data: Expression data
    :type expression_data: np.ndarray
    :param velocity_data: Velocity Data
    :type velocity_data: np.ndarray
    :param decay_quantiles: Min & max percentiles
        for expression/velocity surface estimate
    :type decay_quantiles: tuple(float, float)
    :param alpha_quantile: Percentile for alpha surface estimate,
        defaults to None
    :type alpha_quantile: float, optional
    :return: Decay parameter lambda, production parameter alpha,
        decay paramater standard error
    :rtype: float, float, float
    """

    # Estimate alpha based on maximum velovity
    if alpha_quantile is not None:
        a = _estimate_alpha(
            velocity_data,
            alpha_quantile,
        )

    else:
        a = 0.

    d, dse = _estimate_decay(
        expression_data,
        velocity_data,
        decay_quantiles,
        alpha_est=a
    )

    return d, a, dse


def _estimate_decay(
    expression_data,
    velocity_data,
    decay_quantiles,
    alpha_est=None
):
    """
    Estimate decay constant for a single gene.
    Remove estimate of transcriptional output before estimating
    decay constant, if provided.

    :param expression_data: Expression data
    :type expression_data: np.ndarray
    :param velocity_data: Velocity Data
    :type velocity_data: np.ndarray
    :param decay_quantiles: Min & max percentiles
        for expression/velocity surface estimate
    :type decay_quantiles: tuple(float, float)
    :param alpha_est: Estimate of alpha, defaults to None
    :type alpha_est: float, optional
    :return: Estimate of decay parameter & standard error
    :rtype: float, float
    """

    # If there is an estimate for alpha,
    # modify velocity to remove the alpha component
    if alpha_est is not None:
        velocity_data = np.subtract(
            velocity_data,
            alpha_est
        )

    # Get the velocity / expression ratio
    # Mask to NaN where expression is 0
    ratio_data = np.divide(
        velocity_data,
        expression_data,
        out=np.full_like(velocity_data, np.nan, dtype=float),
        where=expression_data != 0
    )

    # Find the quantile cutoffs for decay curve fitting
    ratio_cuts = np.nanquantile(ratio_data, decay_quantiles)

    # Find the observations at the edge of the velocity/expression surface
    # Use those to estimate the decay constant
    keep_observations = np.greater_equal(ratio_data, ratio_cuts[0])
    keep_observations &= np.less_equal(ratio_data, ratio_cuts[1])

    # Estimate lambda_hat via OLS slope and enforce positive lambda
    ols_slope_se = least_squares(
        expression_data[keep_observations],
        velocity_data[keep_observations]
    )

    # Get decay standard errors
    # Throw away errors that are for rectified decays
    decay_se = ols_slope_se[1] if ols_slope_se[0] <= 0 else 0.

    # Get decay coefficients
    # Ceiling at zero
    decay = min(ols_slope_se[0], 0)

    return decay, decay_se


def _estimate_alpha(
    velocity_data,
    alpha_quantile,
    decay_est=None,
    expression_data=None
):
    """
    Estimate transcriptional output
    If an estimate of decay is provided, remove
    decay component before estimating transcriptional output

    :param velocity_data: Velocity Data
    :type velocity_data: np.ndarray
    :param alpha_quantile: Percentile for alpha estimate
    :type alpha_quantile: float
    :param decay_est: Estimate of lambda (decay constant),
        defaults to None
    :type decay_est: float, optional
    :param expression_data: Expression Data, defaults to None
    :type expression_data: np.ndarray, optional
    :return: Estimate of alpha
    :rtype: float
    """

    if decay_est is not None and expression_data is None:
        raise ValueError("expression_data must be set if decay_est is")

    if decay_est is not None and decay_est != 0:
        velocity_data = velocity_data - expression_data * decay_est

    alpha_est = np.nanquantile(
        velocity_data,
        alpha_quantile
    )

    return max(alpha_est, 0)

import numpy as np

from .misc import get_bins


def aggregate_sliding_window_times(
    expression_data,
    times,
    agg_func=np.mean,
    agg_kwargs={},
    n_windows=None,
    centers=None,
    width=None
):

    centers, half_width = get_bins(
        times,
        n_bins=n_windows,
        centers=centers,
        width=width
    )

    def _agg(center):
        lowend, highend = center - half_width, center + half_width

        keep_idx = (times >= lowend) & (times <= highend) & ~np.isnan(times)
        data = expression_data[keep_idx, ...]

        try:
            return agg_func(data, **agg_kwargs)
        except TypeError:
            return [a(data, **agg_kwargs) for a in agg_func]

    return np.array([_agg(x) for x in centers]), centers

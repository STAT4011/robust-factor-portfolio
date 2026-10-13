"""selection — stock selection by total score and cluster labels.

Rules are registered in ``SELECTORS``, with a common signature::

    sel(score: Series[ticker], cluster: Series[ticker], **params) -> list[str]

score is the cross-sectional total score on the decision date (the score column of the score table), with NaN already removed by the caller; cluster is the cluster column of the score table (``cluster.hierarchical``).
Returns the selected symbols in descending score order. Only decides which stocks to hold; produces no weights. Parameters are registered in ``config.SelectionParams``.

Implemented: cluster_cap (greedy selection from highest to lowest total score, capped per cluster, stopping at a score gap once the minimum count is reached).
"""

from collections.abc import Callable

import pandas as pd


def cluster_cap(
    score: pd.Series,
    cluster: pd.Series,
    max_per_cluster: int,
    min_n: int,
    max_n: int,
    max_gap: float,
) -> list[str]:
    """Greedy selection: consider stocks one by one in descending total score, skip if their cluster already has max_per_cluster stocks, stop once ≥ min_n are selected and a stock scores more than max_gap below the previous selected one, at most max_n stocks.

    Parameters
    ----------
    score : Series[ticker]
        Cross-sectional total score on the decision date (dimensionless), without NaN.
    cluster : Series[ticker]
        Cluster labels (str), index ⊇ score.index.
    max_per_cluster : int
        Maximum number of stocks per cluster, ≥ 1.
    min_n : int
        Minimum number of stocks, 1 ≤ min_n ≤ max_n: score gaps are not checked until min_n stocks are selected (the cluster cap always applies).
    max_n : int
        Maximum number of stocks, ≥ 1.
    max_gap : float
        Maximum allowed score gap, ≥ 0: stop once a candidate scores more than max_gap below the previous selected stock, and no later stocks are considered; a very large value means no limit.

    Returns
    -------
    list[str]
        Selected symbols in descending score order; ties go to whichever appears first in score.
        Length within [min_n, max_n], unless there are not enough available stocks or the cluster caps bind (then fewer than min_n, until exhausted); empty list when score is empty.

    Raises
    ------
    ValueError
        Parameters out of range; a stock in score is not in cluster, or its cluster is NaN (missing prices in the clustering window; try reducing ``cfg.cluster.window``).
    """
    if max_per_cluster < 1 or max_n < 1 or not 1 <= min_n <= max_n or max_gap < 0:
        raise ValueError(f"max_per_cluster ≥ 1, 1 ≤ min_n ≤ max_n and max_gap ≥ 0 are required, got max_per_cluster={max_per_cluster}, "
                         f"min_n={min_n}, max_n={max_n}, max_gap={max_gap}")
    labels = cluster.reindex(score.index)
    if labels.isna().any():
        raise ValueError(f"cluster missing or NaN (try reducing the clustering window): {labels.index[labels.isna()].tolist()}")
    selected: list[str] = []
    counts: dict[str, int] = {}
    # stable: ties keep the order in score.
    for ticker, s in score.sort_values(ascending=False, kind="stable").items():
        if len(selected) == max_n:
            break
        c = labels[ticker]
        if counts.get(c, 0) == max_per_cluster:
            continue
        # The gap is measured against the previous selected stock, not the stock immediately above in the ranking: stocks skipped by the cluster cap do not count.
        if len(selected) >= min_n and score[selected[-1]] - s > max_gap:
            break
        selected.append(ticker)
        counts[c] = counts.get(c, 0) + 1
    return selected


SELECTORS: dict[str, Callable[..., list[str]]] = {
    "cluster_cap": cluster_cap,
}

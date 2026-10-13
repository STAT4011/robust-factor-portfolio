"""cluster — stock clustering on as_of (agglomerative hierarchical clustering), providing "same risk" labels for selection.

Input: truncated prices / benchmarks, as_of, clustering parameters (``config.ClusterParams``).
Output: ``hierarchical`` → cluster labels Series[ticker] ("A", "B", …), merged into the cluster column of the score table by ``backtest.decide``;
``linkage_tree`` → merge tree (scipy linkage matrix), for drawing dendrograms in notebooks.

Distance d = √(2(1 − ρ)), where ρ is the correlation of the window daily returns up to as_of (with remove_market, the residuals of an OLS on the market index).
Under complete linkage, cutting the tree at min_corr is equivalent to "any two stocks in a cluster have ρ ≥ min_corr".

Selection (``selection.SELECTORS``) uses this label to cap the number of stocks per cluster.
"""

from string import ascii_uppercase

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

METHODS: tuple[str, ...] = ("single", "complete", "average", "ward")


def linkage_tree(
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame | None,
    as_of: pd.Timestamp,
    window: int,
    remove_market: bool,
    mkt: str,
    method: str,
) -> tuple[np.ndarray, list[str]]:
    """Merge tree: agglomerative hierarchical clustering of stocks with complete prices in the window, using distance d_ij = √(2(1 − ρ_ij)).

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices. Only as_of and the window + 1 rows before it are used.
    benchmarks : DataFrame[date × benchmark] | None
        Index closing levels; only the mkt column on as_of and the window + 1 trading days of prices before it is used; may be None when remove_market is False.
    as_of : Timestamp
        Decision date, ∈ prices.index.
    window : int
        Number of daily returns, ≥ 3.
    remove_market : bool
        If True, each stock's daily returns are first regressed (OLS with intercept) on mkt's daily returns and ρ is computed from residuals; if False, daily returns are used directly.
    mkt : str
        Market proxy, a price index ∈ benchmarks.columns (cannot be ^VIX); unused when remove_market is False.
    method : str
        Linkage rule, ∈ {"single", "complete", "average", "ward"}.

    Returns
    -------
    Z : ndarray
        scipy linkage matrix of shape (len(tickers) − 1, 4), column 3 is the merge height (distance); shape (0, 4) with fewer than 2 stocks.
    tickers : list[str]
        Stocks taking part in clustering (leaf order of Z), in the order of prices.columns: stocks with none of the window + 1 prices missing.
        Empty list if there are fewer than window rows before as_of, or if mkt has missing values in the window when remove_market is set.

    Raises
    ------
    ValueError
        window < 3, invalid method; remove_market is True while benchmarks is None or lacks the mkt column.
    """
    if window < 3:
        raise ValueError(f"window must be ≥ 3, got {window}")
    if method not in METHODS:
        raise ValueError(f"method must be ∈ {METHODS}, got {method!r}")
    if remove_market and (benchmarks is None or mkt not in benchmarks.columns):
        raise ValueError(f"benchmarks must contain column {mkt!r} when remove_market=True")
    t = prices.index.get_loc(as_of)
    if t < window:
        return np.empty((0, 4)), []
    p = prices.iloc[t - window : t + 1]
    r = (p / p.shift(1)).iloc[1:] - 1
    r = r.loc[:, r.notna().all()]
    if remove_market:
        # benchmarks dates are not identical to prices (e.g. ^VIX has values on market holidays); align to the stocks' trading days.
        m = benchmarks[mkt].reindex(p.index)
        m = (m / m.shift(1)).iloc[1:] - 1
        if m.isna().any():
            return np.empty((0, 4)), []
        x = np.column_stack([np.ones(window), m.to_numpy()])
        beta = np.linalg.lstsq(x, r.to_numpy(), rcond=None)[0]
        r = r - x @ beta
    tickers = r.columns.tolist()
    if len(tickers) < 2:
        return np.empty((0, 4)), tickers
    rho = np.corrcoef(r.to_numpy(), rowvar=False)
    # Floating-point error can make 1 − ρ slightly below 0.
    d = np.sqrt(np.clip(2 * (1 - rho), 0, None))
    np.fill_diagonal(d, 0)
    return linkage(squareform(d, checks=False), method=method), tickers


def hierarchical(
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame | None,
    as_of: pd.Timestamp,
    window: int,
    remove_market: bool,
    mkt: str,
    method: str,
    min_corr: float | None,
    n_clusters: int | None,
) -> pd.Series:
    """Cluster labels: cut the merge tree of ``linkage_tree`` at height h = √(2(1 − min_corr)), or into n_clusters clusters.

    Parameters
    ----------
    prices, benchmarks, as_of, window, remove_market, mkt, method
        Same as ``linkage_tree``.
    min_corr : float | None
        Cut threshold, ∈ [−1, 1]: merges with height ≤ h stay in the same cluster. Under complete linkage, any two stocks in a cluster have ρ ≥ min_corr;
        under average, the average inter-cluster distance is ≤ h; single / ward have no such interpretation. Exactly one of this and n_clusters is non-None.
    n_clusters : int | None
        Number of clusters, ≥ 1: cut into at most n_clusters clusters (possibly fewer when merge heights tie). Exactly one of this and min_corr is non-None.

    Returns
    -------
    Series[ticker]
        Cluster labels (str) "A", "B", …, "Z", "AA", "AB", …; same index as prices.columns.
        A is the cluster with the most stocks; ties are ordered by the alphabetically smallest symbol in the cluster.
        Stocks not taking part in clustering (missing prices in the window, fewer than window rows before as_of, missing mkt, see ``linkage_tree``) are NaN;
        if only 1 stock takes part, it is "A".

    Raises
    ------
    ValueError
        min_corr and n_clusters are not exactly one non-None, or out of range; otherwise same as ``linkage_tree``.
    """
    if (min_corr is None) == (n_clusters is None):
        raise ValueError(f"exactly one of min_corr and n_clusters must be non-None, got min_corr={min_corr}, n_clusters={n_clusters}")
    if min_corr is not None and not -1 <= min_corr <= 1:
        raise ValueError(f"min_corr must be ∈ [−1, 1], got {min_corr}")
    if n_clusters is not None and n_clusters < 1:
        raise ValueError(f"n_clusters must be ≥ 1, got {n_clusters}")
    z, tickers = linkage_tree(prices, benchmarks, as_of, window, remove_market, mkt, method)
    if len(tickers) < 2:
        return pd.Series(dict.fromkeys(tickers, "A"), dtype="str").reindex(prices.columns)
    if min_corr is not None:
        ids = fcluster(z, np.sqrt(2 * (1 - min_corr)), criterion="distance")
    else:
        ids = fcluster(z, n_clusters, criterion="maxclust")
    members = pd.Series(tickers).groupby(ids).agg(list)
    order = sorted(members, key=lambda g: (-len(g), min(g)))
    labels = {}
    for k, g in enumerate(order):
        # Bijective base-26 (Excel column names): 25 → Z, 26 → AA.
        name, q = "", k + 1
        while q:
            q, rem = divmod(q - 1, 26)
            name = ascii_uppercase[rem] + name
        labels.update(dict.fromkeys(g, name))
    return pd.Series(labels, dtype="str").reindex(prices.columns)

"""baselines — build baseline portfolio weights for evaluation, one as_of per call.

Input: truncated prices DataFrame[date × ticker], as_of, and the unpacked parameters of ``config.EvalSpec``.
Output: a stock pool / selection list ``list[str]``, or equal weights ``Series[ticker]`` (fully invested, no CASH).

This only decides what the baselines hold and in what proportion; holding-period daily returns are computed by ``backtest`` with ``holding_returns``,
and index baselines (weight 1) and reference stocks (single stock, equal weight) are also assembled by ``backtest``. Baselines are only for evaluation and never enter any decision.
"""

import numpy as np
import pandas as pd

from pf.factors import momentum


def pool(prices: pd.DataFrame, as_of: pd.Timestamp) -> list[str]:
    """All stocks U(t0) with a close price on as_of, in alphabetical order of symbol.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices; only the as_of row is used.
    as_of : Timestamp
        Decision date, ∈ prices.index.

    Returns
    -------
    list[str]
        Symbols whose close is non-NaN on as_of, alphabetical; stocks not yet listed are excluded.
    """
    return sorted(prices.columns[prices.loc[as_of].notna()])


def equal_weights(tickers: list[str]) -> pd.Series:
    """Equal weight, fully invested: each stock gets weight 1 / len(tickers).

    Parameters
    ----------
    tickers : list[str]
        Symbols of the stocks (or indices) held; non-empty, no duplicates.

    Returns
    -------
    Series[ticker]
        Weights (decimal), same index as tickers (order unchanged), summing to 1; no CASH.
    """
    return pd.Series(1.0 / len(tickers), index=pd.Index(tickers, name="ticker"))


def momentum_top(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    k: int,
    lookback: int,
    skip: int,
) -> list[str] | None:
    """Selection for the mom baseline: the k stocks with the highest ``factors.momentum`` among those with non-NaN values.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices; only rows on and before as_of are used.
    as_of : Timestamp
        Decision date, ∈ prices.index.
    k : int
        Number of stocks to select, ≥ 1.
    lookback, skip : int
        Passed to ``factors.momentum``, lookback > skip ≥ 0.

    Returns
    -------
    list[str] | None
        k stocks in descending momentum, ties broken by alphabetical symbol; None if fewer than k stocks have non-NaN momentum.
    """
    mom = momentum(prices, None, as_of, lookback=lookback, skip=skip).dropna()
    if len(mom) < k:
        return None
    # Sort by symbol first so that a stable sort keeps ties in alphabetical order.
    return mom.sort_index().sort_values(ascending=False, kind="stable").index[:k].tolist()


def random_draws(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    k: int,
    n_random: int,
    random_seed: int,
) -> pd.DataFrame:
    """Random portfolios: draw k stocks from U(t0) without replacement, n_random portfolios in total, each equally weighted.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices; only the as_of row is used (to determine U(t0), see ``pool``).
    as_of : Timestamp
        Decision date, ∈ prices.index; together with random_seed it determines the generator, so draws are unaffected by the rolling rule.
    k : int
        Number of stocks per portfolio, 1 ≤ k ≤ len(U(t0)).
    n_random : int
        Number of portfolios, ≥ 1.
    random_seed : int
        Random seed; the generator is ``np.random.default_rng([random_seed, YYYYMMDD])``.

    Returns
    -------
    DataFrame[draw × slot]
        index is draw = 0 … n_random − 1, columns are slot = 0 … k − 1; each row is one portfolio's k symbols,
        alphabetical within the row, all distinct, all in U(t0). Each weight is 1 / k.
    """
    universe = np.array(pool(prices, as_of))
    rng = np.random.default_rng([random_seed, int(as_of.strftime("%Y%m%d"))])
    draws = np.sort(np.stack([rng.choice(universe, size=k, replace=False) for _ in range(n_random)]), axis=1)
    return pd.DataFrame(draws, index=pd.RangeIndex(n_random, name="draw"), columns=pd.RangeIndex(k, name="slot"))

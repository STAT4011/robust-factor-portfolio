"""factors — computation of raw factor values, one as_of per call.

One function per factor, registered in ``FACTORS`` (name → function). Common signature::

    f(prices, fundamentals, as_of, **params) -> Series[ticker]

- prices: DataFrame[date × ticker], already truncated (passing the full sample gives the same result: only rows on and before as_of are used).
- fundamentals: long-format fundamentals table or None; factors that do not need it ignore it. Factors that need it only use records with ann_date < as_of.
- as_of: decision date, ∈ prices.index.
- Output: factor values on as_of, same index as prices.columns; when a stock has insufficient data, only that stock is NaN.

Exception: factors in ``EXTRA_DATA_FACTORS`` (currently only ret) also take a keyword-only argument benchmarks,
forwarded by ``scoring``; ret returns KNN's predicted_return, ``ret_details`` additionally exposes the uncertainty from the same computation, and the horizon parameter reuses the B-class holding_days.

All factors are higher-is-better: factors that need reversing are negated here, and ``scoring`` no longer handles direction.
Standardization belongs to ``scoring``. Parameters are registered in ``config.FactorParams``.

Implemented: momentum, downside_vol, revenue_growth, and ret (return) computed by the two-stage KNN.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from pf.config import KNNParams
from pf.knn import KNNResult, predict_return_details


def momentum(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    as_of: pd.Timestamp,
    lookback: int,
    skip: int,
) -> pd.Series:
    """Momentum: P[t−skip] / P[t−lookback] − 1 on as_of, where t is the row position of as_of.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices. Only rows on and before as_of are used.
    fundamentals : long-format fundamentals table | None
        Not used.
    as_of : Timestamp
        Decision date, ∈ prices.index.
    lookback : int
        Lookback in trading days, > skip.
    skip : int
        Number of most recent trading days to skip, ≥ 0.

    Returns
    -------
    Series[ticker]
        Cumulative simple return on as_of (decimal, not annualized); same index as prices.columns.
        NaN for stocks missing P[t−lookback] or P[t−skip] (e.g. CRWV, listed fewer than lookback days);
        all NaN when there are fewer than lookback rows before as_of.

    Raises
    ------
    ValueError
        lookback > skip ≥ 0 is not satisfied.
    """
    if not lookback > skip >= 0:
        raise ValueError(f"lookback > skip ≥ 0 is required, got lookback={lookback}, skip={skip}")
    t = prices.index.get_loc(as_of)
    if t < lookback:
        return pd.Series(np.nan, index=prices.columns)
    return prices.iloc[t - skip] / prices.iloc[t - lookback] - 1


def downside_vol(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    as_of: pd.Timestamp,
    window: int,
    tau: float,
    periods_per_year: int,
) -> pd.Series:
    """Negative downside volatility: −√(A · (1/window) · Σ min(r − tau, 0)²), summed over the window daily returns up to as_of.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices. Only rows on and before as_of are used.
    fundamentals : long-format fundamentals table | None
        Not used.
    as_of : Timestamp
        Decision date, ∈ prices.index.
    window : int
        Number of daily returns, ≥ 1; needs window + 1 prices.
    tau : float
        Daily return threshold (decimal). Days above the threshold count as 0 but still count in the denominator window.
    periods_per_year : int
        Number of periods per year A, > 0.

    Returns
    -------
    Series[ticker]
        Negative of the annualized semi-deviation on as_of (decimal), ≤ 0, closer to 0 is better; same index as prices.columns.
        NaN for stocks missing any daily return in the window (e.g. CRWV, listed fewer than window days);
        all NaN when there are fewer than window rows before as_of.

    Raises
    ------
    ValueError
        window < 1 or periods_per_year ≤ 0.
    """
    if window < 1 or periods_per_year <= 0:
        raise ValueError(f"window ≥ 1 and periods_per_year > 0 are required, got window={window}, "
                         f"periods_per_year={periods_per_year}")
    t = prices.index.get_loc(as_of)
    if t < window:
        return pd.Series(np.nan, index=prices.columns)
    p = prices.iloc[t - window : t + 1]
    r = (p / p.shift(1)).iloc[1:] - 1
    # skipna=False: missing a single day in the window makes the whole stock NaN, rather than averaging over the remaining days.
    lpm2 = ((r - tau).clip(upper=0) ** 2).mean(skipna=False)
    return -np.sqrt(periods_per_year * lpm2)


# revenue_growth method → number of months to the base period (period in the fundamentals table is a month).
REVENUE_GROWTH_LAGS: dict[str, int] = {"yoy": 12, "qoq": 3}


def revenue_growth(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    as_of: pd.Timestamp,
    method: str,
) -> pd.Series:
    """Revenue growth: (Rev[q] − Rev[q−k]) / Rev[q−k], where q is the month of the latest effective fiscal quarter at as_of; yoy uses k = 12 months, qoq uses k = 3 months.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Only its columns are used as the output index.
    fundamentals : long-format fundamentals table
        Revenue (single quarter, USD), period is ``"YYYY-MM"``. A record takes effect from the trading day after its announcement date (ann_date < as_of);
        records with a missing ann_date never take effect.
    as_of : Timestamp
        Decision date.
    method : str
        ∈ ``REVENUE_GROWTH_LAGS`` ("yoy", "qoq").

    Returns
    -------
    Series[ticker]
        Growth rate on as_of (decimal, not annualized); same index as prices.columns. NaN for stocks with:
        no effective fiscal quarter; base period q−k without a record or not yet effective; Rev[q] or Rev[q−k] missing; Rev[q] ≤ 0 or Rev[q−k] ≤ 0;
        not present in fundamentals.

    Raises
    ------
    ValueError
        method is not in ``REVENUE_GROWTH_LAGS``, or fundamentals is None.
    """
    if method not in REVENUE_GROWTH_LAGS:
        raise ValueError(f"method must be one of {tuple(REVENUE_GROWTH_LAGS)}, got {method!r}")
    if fundamentals is None:
        raise ValueError("revenue_growth requires fundamentals")
    lag = REVENUE_GROWTH_LAGS[method]
    # NaT < as_of is False, so records missing an announcement date are naturally excluded.
    eff = fundamentals.loc[fundamentals["ann_date"] < as_of]
    if eff.empty:
        return pd.Series(np.nan, index=prices.columns)
    # Month "2025-06" → consecutive month number; q − 12 is the same quarter of the previous fiscal year.
    ym = eff["period"].str.split("-", expand=True).astype("int64")
    eff = eff.assign(q=ym[0] * 12 + ym[1] - 1)
    latest = eff.loc[eff.groupby("ticker")["q"].idxmax()].set_index("ticker")
    values = eff.set_index(["ticker", "q"])["value"]
    base_key = pd.MultiIndex.from_arrays([latest.index, latest["q"] - lag])
    base = values.reindex(base_key).set_axis(latest.index)
    # Revenue ≤ 0 can only be a derivation error in the data source (e.g. a negative Q4 for EOG in some year); treat the period as unusable as well.
    growth = ((latest["value"] - base) / base).where((base > 0) & (latest["value"] > 0))
    return growth.reindex(prices.columns).rename(None)


def _ret_knn_params(
    market: str,
    market_pool_size: int,
    n_neighbors: int,
    min_neighbors: int,
    neighbor_gap: int | None,
    market_momentum_window: int,
    market_vol_window: int,
    market_drawdown_window: int,
    breadth_window: int,
    dispersion_window: int,
    stock_momentum_window: int,
    stock_vol_window: int,
    stock_drawdown_window: int,
    market_distance_weight: float,
    stock_distance_weight: float,
) -> KNNParams:
    """Assemble the ret factor parameters into KNNParams."""
    return KNNParams(
        market=market,
        market_pool_size=market_pool_size,
        n_neighbors=n_neighbors,
        min_neighbors=min_neighbors,
        neighbor_gap=neighbor_gap,
        market_momentum_window=market_momentum_window,
        market_vol_window=market_vol_window,
        market_drawdown_window=market_drawdown_window,
        breadth_window=breadth_window,
        dispersion_window=dispersion_window,
        stock_momentum_window=stock_momentum_window,
        stock_vol_window=stock_vol_window,
        stock_drawdown_window=stock_drawdown_window,
        market_distance_weight=market_distance_weight,
        stock_distance_weight=stock_distance_weight,
    )


def ret_details(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    as_of: pd.Timestamp,
    horizon: int,
    market: str,
    market_pool_size: int,
    n_neighbors: int,
    min_neighbors: int,
    neighbor_gap: int | None,
    market_momentum_window: int,
    market_vol_window: int,
    market_drawdown_window: int,
    breadth_window: int,
    dispersion_window: int,
    stock_momentum_window: int,
    stock_vol_window: int,
    stock_drawdown_window: int,
    market_distance_weight: float,
    stock_distance_weight: float,
    *,
    benchmarks: pd.DataFrame | None,
) -> KNNResult:
    """Full KNN result of the return factor, so the decision path can reuse both the prediction and the uncertainty."""
    params = _ret_knn_params(
        market=market,
        market_pool_size=market_pool_size,
        n_neighbors=n_neighbors,
        min_neighbors=min_neighbors,
        neighbor_gap=neighbor_gap,
        market_momentum_window=market_momentum_window,
        market_vol_window=market_vol_window,
        market_drawdown_window=market_drawdown_window,
        breadth_window=breadth_window,
        dispersion_window=dispersion_window,
        stock_momentum_window=stock_momentum_window,
        stock_vol_window=stock_vol_window,
        stock_drawdown_window=stock_drawdown_window,
        market_distance_weight=market_distance_weight,
        stock_distance_weight=stock_distance_weight,
    )
    return predict_return_details(
        prices, fundamentals, benchmarks, as_of, horizon, params
    )


def ret(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    as_of: pd.Timestamp,
    horizon: int,
    market: str,
    market_pool_size: int,
    n_neighbors: int,
    min_neighbors: int,
    neighbor_gap: int | None,
    market_momentum_window: int,
    market_vol_window: int,
    market_drawdown_window: int,
    breadth_window: int,
    dispersion_window: int,
    stock_momentum_window: int,
    stock_vol_window: int,
    stock_drawdown_window: int,
    market_distance_weight: float,
    stock_distance_weight: float,
    *,
    benchmarks: pd.DataFrame | None,
) -> pd.Series:
    """return factor: the two-stage KNN's raw return prediction at as_of for the next horizon days."""
    return ret_details(
        prices,
        fundamentals,
        as_of,
        horizon=horizon,
        market=market,
        market_pool_size=market_pool_size,
        n_neighbors=n_neighbors,
        min_neighbors=min_neighbors,
        neighbor_gap=neighbor_gap,
        market_momentum_window=market_momentum_window,
        market_vol_window=market_vol_window,
        market_drawdown_window=market_drawdown_window,
        breadth_window=breadth_window,
        dispersion_window=dispersion_window,
        stock_momentum_window=stock_momentum_window,
        stock_vol_window=stock_vol_window,
        stock_drawdown_window=stock_drawdown_window,
        market_distance_weight=market_distance_weight,
        stock_distance_weight=stock_distance_weight,
        benchmarks=benchmarks,
    ).predicted_return

# Factors that take a keyword-only argument benchmarks besides the common signature; scoring forwards this data only to them.
EXTRA_DATA_FACTORS: frozenset[str] = frozenset({"ret"})


FACTORS: dict[str, Callable[..., pd.Series]] = {
    "momentum": momentum,
    "downside_vol": downside_vol,
    "revenue_growth": revenue_growth,
    "ret": ret,
}

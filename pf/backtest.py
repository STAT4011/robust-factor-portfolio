"""backtest — single-period decisions, holding-period realization and rolling backtests.

- Decision ``decide``: truncate data at as_of → factors → scoring → cluster → selection → weights; outputs the score table and weights w.
  run.ipynb calls it with the B-class parameter ``decision_date`` as as_of to get the final submitted portfolio.
- Realization ``holding_returns``: build the position at the as_of close according to w, hold for H trading days without rebalancing (weights drift), and get daily portfolio returns.
- Baselines ``baseline_returns``: build baseline weights from truncated data via ``baselines`` (indices, equal-weight pool, equal-weight selection, momentum, reference stocks,
  random portfolios), then compute daily returns over the same holding period with the ``holding_returns`` formula; used only for evaluation, never in decisions.
- Risk-free rate ``rf_daily``: effective ^IRX value (annualized percent) / 100 / A; on a missing day, use the rate of the most recent earlier date that has a value.
- Rolling ``run_backtest``: loop over ``as_of_dates`` (the rolling rule of the C-class ``RunSpec``), calling the steps above each period,
  and collect into a ``BacktestResult`` (per-period daily returns, portfolio daily returns Series[date], per-period returns, per-period weights, baseline daily returns, daily risk-free rate).

**Look-ahead truncation happens only here**: prices and benchmark indices (benchmarks) use ``date ≤ as_of``, fundamentals use ``ann_date < as_of``.
Only the holding-period realization uses prices (and the daily risk-free rate) after as_of. No metrics are computed and no files are written.

Selection uses the score and cluster columns of the score table; the rule is given by ``cfg.selection.method``.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pf.baselines import equal_weights, momentum_top, pool, random_draws
from pf.cluster import hierarchical
from pf.config import Config, EvalSpec, RunSpec
from pf.factors import ret_details
from pf.scoring import combine, factor_values
from pf.selection import SELECTORS
from pf.weights import CASH, WEIGHTERS


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Result of a rolling backtest; as_of is ascending and aligned across all tables."""

    # DataFrame[date × as_of]: each column is one period's portfolio daily returns (decimal, daily); NaN outside the holding period.
    daily: pd.DataFrame
    # Series[date]: portfolio daily returns of all periods concatenated end to end; None when step < H (periods overlap) and they cannot form one series.
    returns: pd.Series | None
    # DataFrame[as_of × {end, period_return}]: last day of the holding period, period return (1 + r).prod() − 1 (decimal, not annualized).
    periods: pd.DataFrame
    # DataFrame[as_of × ticker]: entry weights; columns are prices.columns + CASH, NaN if not selected, each row's non-NaN values sum to 1.
    weights: pd.DataFrame
    # Baseline key (``baseline_keys``, excluding rand) → DataFrame[date × as_of]: each baseline's daily returns per period, same shape as daily.
    baseline_daily: dict[str, pd.DataFrame]
    # as_of → DataFrame[date × draw]: daily returns of that period's n_random random portfolios; rows are the period's H trading days.
    random_daily: dict[pd.Timestamp, pd.DataFrame]
    # Series[date]: daily risk-free rate (decimal, daily), same index as prices.index; NaN on dates before any ^IRX value. Used only for evaluation.
    rf_daily: pd.Series


def as_of_dates(
    index: pd.DatetimeIndex,
    start: str,
    end: str,
    step: int,
    horizon: int,
) -> list[pd.Timestamp]:
    """Rolling rule: starting from the first trading day ≥ start, take an as_of every step trading days, keeping only those whose following horizon trading days are all ≤ end.

    Parameters
    ----------
    index : DatetimeIndex
        All trading days (prices.index), ascending and unique.
    start, end : str
        YYYY-MM-DD, start ≤ end; [start, end] covers all as_of dates and their holding periods.
    step : int
        Number of trading days between adjacent as_of dates, ≥ 1.
    horizon : int
        Holding period H (trading days), ≥ 1.

    Returns
    -------
    list[Timestamp]
        as_of ascending, all ∈ index, adjacent ones step rows apart; row number of the last one + horizon ≤ row number of the last trading day before end.

    Raises
    ------
    ValueError
        step < 1, horizon < 1, start > end, or not a single as_of can be taken.
    """
    if step < 1 or horizon < 1:
        raise ValueError(f"step ≥ 1 and horizon ≥ 1 are required, got step={step}, horizon={horizon}")
    t_start, t_end = pd.Timestamp(start), pd.Timestamp(end)
    if t_start > t_end:
        raise ValueError(f"start={start} is later than end={end}")
    first = index.searchsorted(t_start, side="left")
    last = index.searchsorted(t_end, side="right") - 1
    # When end falls within the first horizon rows of the data, last − horizon + 1 is negative and slicing would count from the end, so clip to 0.
    dates = list(index[first : max(last - horizon + 1, 0) : step])
    if not dates:
        raise ValueError(f"no as_of in [{start}, {end}] is followed by {horizon} more trading days")
    return dates


def decide(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    benchmarks: pd.DataFrame | None,
    as_of: pd.Timestamp,
    cfg: Config,
) -> tuple[pd.DataFrame, pd.Series]:
    """One decision after the as_of close: truncate → factors → scoring → clustering → selection → weights.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices, may be the full sample; only rows with date ≤ as_of are used.
    fundamentals : long-format fundamentals table | None
        May be the full sample; only records with ann_date < as_of are used. Must not be None when a factor that needs it is enabled.
    benchmarks : DataFrame[date × benchmark] | None
        Index closing levels, may be the full sample; only rows with date ≤ as_of are used. Used by ret (knn) and by remove_market in clustering.
    as_of : Timestamp
        Decision date, ∈ prices.index.
    cfg : Config
        Uses D-class ``factor_weights``, E-class ``factors`` / ``standardize`` / ``cluster`` / ``selection`` / ``weights``,
        as well as B-class ``periods_per_year`` (downside_vol) and ``holding_days`` (the horizon of ret and weights).

    Returns
    -------
    table : DataFrame[ticker × {score, predicted_return, prediction_uncertainty, knn_neighbor_count, cluster}]
        Score table on as_of: the output of ``scoring.combine``, plus the prediction uncertainty and neighbor count from the same KNN run, and cluster.
    w : Series[ticker]
        Entry weights (decimal), index = selected stocks (by total score descending) + CASH, summing to 1; same as the output of ``weights.WEIGHTERS``.

    Raises
    ------
    ValueError
        as_of is not in prices.index; cfg.factor_weights contains an unregistered factor; not a single stock is selected
        (as_of too early, every stock is missing some factor). Incomplete data for selected stocks etc. is raised by downstream functions and not caught here.
    """
    if as_of not in prices.index:
        raise ValueError(f"as_of={as_of.date()} is not a trading day in prices")
    unknown = [k for k in cfg.factor_weights if not hasattr(cfg.factors, k)]
    if unknown:
        raise ValueError(f"cfg.factor_weights contains unregistered factors: {unknown}")
    p = prices.loc[:as_of]
    b = benchmarks.loc[:as_of] if benchmarks is not None else None
    # The announcement time is unknown and may not be available at that day's close, so use strictly less than.
    f = fundamentals[fundamentals["ann_date"] < as_of] if fundamentals is not None else None

    horizon = cfg.task.holding_days
    factor_params = {k: asdict(getattr(cfg.factors, k)) for k in cfg.factor_weights}
    # B-class parameters are merged in by the caller and not registered again in the E class.
    if "downside_vol" in factor_params:
        factor_params["downside_vol"]["periods_per_year"] = cfg.task.periods_per_year
    if "ret" in factor_params:
        factor_params["ret"]["horizon"] = horizon

    knn_result = None
    precomputed: dict[str, pd.Series] = {}
    if "ret" in factor_params:
        knn_result = ret_details(
            p, f, as_of, benchmarks=b, **factor_params["ret"]
        )
        precomputed["ret"] = knn_result.predicted_return

    values = factor_values(
        p,
        f,
        as_of,
        factor_params,
        benchmarks=b,
        precomputed=precomputed,
    )
    table = combine(values, cfg.factor_weights, cfg.standardize.n)
    if knn_result is None:
        table["prediction_uncertainty"] = float("nan")
        table["knn_neighbor_count"] = 0
    else:
        table["prediction_uncertainty"] = knn_result.uncertainty
        table["knn_neighbor_count"] = knn_result.neighbor_count
    table["cluster"] = hierarchical(p, b, as_of, **asdict(cfg.cluster))
    sel = cfg.selection.method
    selected = SELECTORS[sel](table["score"].dropna(), table["cluster"], **asdict(getattr(cfg.selection, sel)))
    # Without this check, an empty selection would only surface in weights as "selected must be non-empty", hiding that as_of is too early.
    if not selected:
        raise ValueError(f"as_of={as_of.date()}: no stock has a total score (insufficient factor history), so none can be selected; move the backtest start later")
    method = cfg.weights.method
    w = WEIGHTERS[method](
        p,
        table["predicted_return"],
        table["prediction_uncertainty"],
        selected,
        as_of,
        **asdict(getattr(cfg.weights, method)),
        horizon=horizon,
    )
    return table, w


def holding_returns(
    prices: pd.DataFrame,
    w: pd.Series,
    as_of: pd.Timestamp,
    horizon: int,
) -> pd.Series:
    """Buy-and-hold portfolio daily returns: V_t = w_cash + Σ_i w_i · P_{i,t} / P_{i,t0}, V_{t0} = 1, r_t = V_t / V_{t−1} − 1.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices; must contain at least horizon rows after as_of (full sample, not truncated).
    w : Series[ticker]
        Entry weights (decimal) at the as_of close; stock index ⊆ prices.columns, may additionally contain ``CASH``.
        Cash return is fixed at 0; no rebalancing during the period, so actual weights drift with stock prices.
    as_of : Timestamp
        Entry date t0, ∈ prices.index.
    horizon : int
        Holding period H (trading days), ≥ 1.

    Returns
    -------
    Series[date]
        Portfolio daily simple returns (decimal, daily); index is the horizon trading days after as_of.
        Period return = (1 + r).prod() − 1 = V_{t0+H} − 1. If a held stock is missing a price on some day,
        V is NaN on that day, and r is NaN on that day and the next.

    Raises
    ------
    ValueError
        horizon < 1, as_of is not in prices.index, or fewer than horizon rows after as_of.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")
    if as_of not in prices.index:
        raise ValueError(f"as_of={as_of.date()} is not a trading day in prices")
    t = prices.index.get_loc(as_of)
    if t + horizon >= len(prices.index):
        raise ValueError(f"fewer than horizon={horizon} rows after as_of={as_of.date()}")
    cash_weight = float(w.get(CASH, 0.0))
    risky_w = w.drop(labels=CASH, errors="ignore")
    window_index = prices.index[t : t + horizon + 1]
    if risky_w.empty:
        risky_value = pd.Series(0.0, index=window_index)
    else:
        p = prices.iloc[t : t + horizon + 1][risky_w.index]
        # skipna=False: if any held stock is missing a price, that day's NAV is unknown; do not renormalize over the remaining stocks.
        risky_value = (p / p.iloc[0]).mul(risky_w, axis=1).sum(axis=1, skipna=False)
    v = cash_weight + risky_value
    return (v / v.shift(1) - 1).iloc[1:]


def baseline_keys(ev: EvalSpec) -> list[str]:
    """Keys of the deterministic baselines, in the order: index baselines, ew_pool, ew_sel, mom, reference stocks.

    Parameters
    ----------
    ev : EvalSpec
        Uses ``index_baselines`` and ``reference_tickers``.

    Returns
    -------
    list[str]
        Keys for indices and reference stocks are the symbol with ``^`` removed, lowercased (``^GSPC`` → ``gspc``, ``NVDA`` → ``nvda``).
    """
    return ([_key(n) for n in ev.index_baselines] + ["ew_pool", "ew_sel", "mom"]
            + [_key(t) for t in ev.reference_tickers])


def baseline_returns(
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame | None,
    as_of: pd.Timestamp,
    w: pd.Series,
    cfg: Config,
) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """Daily returns of each baseline portfolio for one as_of period: weights use only data with date ≤ as_of, buy and hold for H days, fully invested.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices, full sample: baseline weights use only the part truncated at as_of; the holding period uses the H rows after as_of.
    benchmarks : DataFrame[date × benchmark] | None
        Index closing levels, full sample; must contain the risk_free and index_baselines columns of ``cfg.task.eval``.
    as_of : Timestamp
        Entry date t0, ∈ prices.index.
    w : Series[ticker]
        Strategy entry weights at as_of (output of ``decide``); sel is all stocks except CASH, including those with weight 0, k = len(sel) ≥ 1.
    cfg : Config
        Uses ``task.holding_days`` (H), ``task.eval`` and ``data.benchmark``.

    Returns
    -------
    daily : dict[str, Series[date]]
        Keys are ``baseline_keys(cfg.task.eval)``; values are that baseline's portfolio daily simple returns (decimal, daily), index is the
        H trading days after as_of (same as ``holding_returns``). All NaN when the reference stock has no price at as_of, or fewer than mom_k stocks have non-NaN momentum;
        missing prices of a holding on some day are handled as in ``holding_returns``.
    random : DataFrame[date × draw]
        Daily returns of n_random random portfolios (``baselines.random_draws``, k stocks equally weighted); index as above, columns are draw.

    Raises
    ------
    ValueError
        ``cfg.task.eval`` does not match the data (see ``run_backtest``); same as ``holding_returns``.
    """
    ev, horizon = cfg.task.eval, cfg.task.holding_days
    _check_eval(prices, benchmarks, cfg)
    if as_of not in prices.index:
        raise ValueError(f"as_of={as_of.date()} is not a trading day in prices")
    p = prices.loc[:as_of]
    # Alphabetical order: the summation order of equal weights is independent of the ordering of w, so recomputing from the saved (alphabetical) weights matches value for value.
    selected = sorted(t for t in w.index if t != CASH)
    t = prices.index.get_loc(as_of)
    window_index = prices.index[t + 1 : t + horizon + 1]

    daily: dict[str, pd.Series] = {}
    for name in ev.index_baselines:
        # benchmarks dates are not identical to prices dates (^VIX has values on market holidays); without alignment the holding period would count an extra row.
        daily[_key(name)] = holding_returns(benchmarks[[name]].reindex(prices.index), equal_weights([name]), as_of, horizon)
    daily["ew_pool"] = holding_returns(prices, equal_weights(pool(p, as_of)), as_of, horizon)
    daily["ew_sel"] = holding_returns(prices, equal_weights(selected), as_of, horizon)
    mom = momentum_top(p, as_of, ev.mom_k, ev.mom_lookback, ev.mom_skip)
    daily["mom"] = (holding_returns(prices, equal_weights(mom), as_of, horizon) if mom is not None
                    else pd.Series(np.nan, index=window_index))
    for ticker in ev.reference_tickers:
        # When as_of has no price, P/P_{t0} is all NaN, and holding_returns naturally yields NaN for the whole period.
        daily[_key(ticker)] = holding_returns(prices, equal_weights([ticker]), as_of, horizon)

    draws = random_draws(p, as_of, len(selected), ev.n_random, ev.random_seed)
    window = prices.iloc[t : t + horizon + 1]
    ratio = (window / window.iloc[0]).to_numpy()
    # Same formula and same order of operations as holding_returns: P/P0 × (1/k) summed across stocks in row order (NaN propagates), cash is 0.
    held = ratio[:, prices.columns.get_indexer(draws.to_numpy().ravel())].reshape(len(window), *draws.shape)
    v = 0.0 + (held * (1.0 / len(selected))).sum(axis=2)
    random = pd.DataFrame(v[1:] / v[:-1] - 1, index=window_index, columns=draws.index)
    return daily, random


def rf_daily(
    benchmarks: pd.DataFrame,
    index: pd.DatetimeIndex,
    risk_free: str,
    periods_per_year: int,
) -> pd.Series:
    """Daily risk-free rate rf_d = IRX^eff_d / 100 / A; IRX^eff_d is the most recent non-NaN rate with date ≤ d.

    Parameters
    ----------
    benchmarks : DataFrame[date × benchmark]
        Full sample; the risk_free column is in annualized percent (4.25 means 4.25%).
    index : DatetimeIndex
        Output dates (prices.index), ascending and unique.
    risk_free : str
        Rate column name (``EvalSpec.risk_free``), ∈ benchmarks.columns.
    periods_per_year : int
        Number of periods per year A, > 0.

    Returns
    -------
    Series[date]
        Daily risk-free rate (decimal, daily), index is index; NaN on days before any rate value. Used only for evaluation.
    """
    # Days when the bond market is closed but the stock market is open have no rate; take the most recent earlier "effective value", just as fundamentals take effect on their announcement date.
    rate = benchmarks[risk_free].dropna().reindex(index, method="ffill")
    return (rate / 100 / periods_per_year).rename("rf").rename_axis("date")


def run_backtest(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    benchmarks: pd.DataFrame | None,
    cfg: Config,
    run: RunSpec,
) -> BacktestResult:
    """Rolling backtest: call ``decide`` for each as_of in ``as_of_dates``, then realize the H-day holding with ``holding_returns``.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices, full sample (truncated at decision time; the holding period uses rows after as_of).
    fundamentals : long-format fundamentals table | None
        Full sample; same as ``decide``.
    benchmarks : DataFrame[date × benchmark] | None
        Full sample; same as ``decide``.
    cfg : Config
        Strategy parameters; the holding period H is ``cfg.task.holding_days``.
    run : RunSpec
        Rolling rule ``start``, ``end``, ``step``.

    Returns
    -------
    BacktestResult
        See its field descriptions. When step ≥ H, ``returns`` is a Series[date] (when step > H there are gaps between periods, and gap days do not appear);
        otherwise None. Baselines and random portfolios are computed each period by ``baseline_returns`` and do not affect the strategy's decisions or returns.

    Raises
    ------
    ValueError
        Same as ``as_of_dates``; ``cfg.task.eval`` does not match the data: benchmarks is None or has no risk_free column (no constant is substituted),
        index_baselines are not in benchmarks, contain risk_free or do not contain ``cfg.data.benchmark``, reference_tickers are not in
        prices.columns, mom_k < 1, n_random < 1, or quantiles do not satisfy 0 < first < second < 1;
        an error raised by ``decide`` in any period is re-raised as is (e.g. no stock can be selected, incomplete data for selected stocks).
    """
    horizon = cfg.task.holding_days
    _check_eval(prices, benchmarks, cfg)
    dates = as_of_dates(prices.index, run.start, run.end, run.step, horizon)
    daily, weights, base, random = {}, {}, {}, {}
    for as_of in dates:
        _, w = decide(prices, fundamentals, benchmarks, as_of, cfg)
        daily[as_of] = holding_returns(prices, w, as_of, horizon)
        weights[as_of] = w
        base[as_of], random[as_of] = baseline_returns(prices, benchmarks, as_of, w, cfg)

    daily_df = pd.DataFrame(daily).rename_axis(index="date", columns="as_of")
    periods = pd.DataFrame(
        {"end": [r.index[-1] for r in daily.values()],
         "period_return": [(1 + r).prod(skipna=False) - 1 for r in daily.values()]},
        index=pd.DatetimeIndex(dates, name="as_of"),
    )
    # Only when periods do not overlap does each day belong to a single period, so concatenation makes sense.
    returns = pd.concat(daily.values()).rename_axis("date") if run.step >= horizon else None
    weights_df = pd.DataFrame(weights).T.reindex(columns=[*prices.columns, CASH]).rename_axis(index="as_of", columns="ticker")
    baseline_daily = {
        k: pd.DataFrame({a: base[a][k] for a in dates}).rename_axis(index="date", columns="as_of")
        for k in baseline_keys(cfg.task.eval)
    }
    return BacktestResult(
        daily=daily_df, returns=returns, periods=periods, weights=weights_df,
        baseline_daily=baseline_daily, random_daily=random,
        rf_daily=rf_daily(benchmarks, prices.index, cfg.task.eval.risk_free, cfg.task.periods_per_year),
    )


def _key(name: str) -> str:
    """Index or reference stock symbol → baseline key: remove ``^`` and lowercase.

    Parameters
    ----------
    name : str
        Index name or stock symbol.

    Returns
    -------
    str
        Baseline key, e.g. ``^GSPC`` → ``gspc``.
    """
    return name.replace("^", "").lower()


def _check_eval(prices: pd.DataFrame, benchmarks: pd.DataFrame | None, cfg: Config) -> None:
    """Check that the evaluation setup ``cfg.task.eval`` matches the data; raise ValueError if not (conditions in ``run_backtest``).

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices; columns are used.
    benchmarks : DataFrame[date × benchmark] | None
        Index closing levels; columns are used.
    cfg : Config
        Uses ``task.eval`` and ``data.benchmark``.
    """
    ev = cfg.task.eval
    if benchmarks is None or ev.risk_free not in benchmarks.columns:
        raise ValueError(f"no risk-free rate column {ev.risk_free!r} in benchmarks (no constant or 0 is substituted)")
    bad = [n for n in ev.index_baselines if n not in benchmarks.columns or n == ev.risk_free]
    if bad:
        raise ValueError(f"index_baselines must be price indices in benchmarks, got {bad}")
    if cfg.data.benchmark not in ev.index_baselines:
        raise ValueError(f"index_baselines must include the market index {cfg.data.benchmark!r} (used for beta)")
    missing = [t for t in ev.reference_tickers if t not in prices.columns]
    if missing:
        raise ValueError(f"reference_tickers must be in the candidate pool, got {missing}")
    if ev.mom_k < 1 or ev.n_random < 1:
        raise ValueError(f"mom_k ≥ 1 and n_random ≥ 1 are required, got mom_k={ev.mom_k}, n_random={ev.n_random}")
    q_lo, q_hi = ev.quantiles
    if not 0 < q_lo < q_hi < 1:
        raise ValueError(f"quantiles must satisfy 0 < first < second < 1, got {ev.quantiles}")

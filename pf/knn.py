"""knn — two-stage return prediction from historically similar states, one as_of per call.

Stage one filters similar market states by market trend, risk, drawdown, pool breadth and cross-sectional dispersion;
stage two re-ranks within the market candidate pool by each stock's momentum relative to the market, market-regression residual momentum,
annualized idiosyncratic volatility and stock drawdown.
Input data may be the full sample; this module first truncates to as_of, and historical states s must satisfy s + horizon < as_of.

``predict_return`` is the repository's common entry point and outputs the raw horizon-day predicted returns Series[ticker];
``predict_return_details`` additionally returns KNN prediction uncertainty and neighbor dates, for robust MVO and debugging.

Time-series standardization of the distance features is done in this module, using only historical candidate states that were valid at the time;
same-day cross-sectional standardization of predictions is still done by ``scoring.standardize``.
"""

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pf.config import KNNParams


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True, slots=True)
class KNNResult:
    """Full result of the two-stage KNN."""

    predicted_return: pd.Series
    outcome_dispersion: pd.Series
    uncertainty: pd.Series
    neighbor_count: pd.Series
    neighbors: dict[str, tuple[pd.Timestamp, ...]] = field(default_factory=dict)


def _validate_inputs(
    prices: pd.DataFrame,
    benchmarks: pd.DataFrame | None,
    as_of: pd.Timestamp,
    horizon: int,
    params: KNNParams,
) -> None:
    """Validate the inputs and parameters of the two-stage KNN."""
    if horizon < 1:
        raise ValueError(f"horizon must be ≥ 1, got {horizon}")
    if as_of not in prices.index:
        raise ValueError(f"as_of={as_of.date()} is not a trading day in prices")
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError("prices.index must be ascending and unique")
    if benchmarks is None:
        raise ValueError("two-stage KNN requires benchmarks")
    if params.market not in benchmarks.columns:
        raise ValueError(f"market index {params.market!r} is not in benchmarks")
    if params.market_pool_size < params.n_neighbors:
        raise ValueError("market_pool_size must be ≥ n_neighbors")
    if not 1 <= params.min_neighbors <= params.n_neighbors:
        raise ValueError("1 ≤ min_neighbors ≤ n_neighbors is required")
    windows = {
        "market_momentum_window": params.market_momentum_window,
        "market_vol_window": params.market_vol_window,
        "market_drawdown_window": params.market_drawdown_window,
        "breadth_window": params.breadth_window,
        "dispersion_window": params.dispersion_window,
        "stock_momentum_window": params.stock_momentum_window,
        "stock_vol_window": params.stock_vol_window,
        "stock_drawdown_window": params.stock_drawdown_window,
    }
    bad_windows = {name: value for name, value in windows.items() if value < 2}
    if bad_windows:
        raise ValueError(f"all windows must be ≥ 2, got {bad_windows}")
    if params.neighbor_gap is not None and params.neighbor_gap < 1:
        raise ValueError(f"neighbor_gap must be None or ≥ 1, got {params.neighbor_gap}")
    if params.market_distance_weight < 0 or params.stock_distance_weight < 0:
        raise ValueError("market_distance_weight and stock_distance_weight must be ≥ 0")
    if params.market_distance_weight + params.stock_distance_weight == 0:
        raise ValueError("the two distance weights cannot both be 0")


def _monthly_first_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the first trading day of each month that appears in index."""
    months = index.to_period("M")
    return index[~months.duplicated()]


def _market_features(
    prices: pd.DataFrame,
    market: pd.Series,
    params: KNNParams,
) -> pd.DataFrame:
    """Compute the market state features for each trading day."""
    market_return = market.pct_change(fill_method=None)
    market_momentum = market / market.shift(params.market_momentum_window) - 1
    market_vol = market_return.rolling(
        params.market_vol_window, min_periods=params.market_vol_window
    ).std(ddof=1)
    market_drawdown = (
        market
        / market.rolling(
            params.market_drawdown_window,
            min_periods=params.market_drawdown_window,
        ).max()
        - 1
    )

    breadth_return = prices / prices.shift(params.breadth_window) - 1
    breadth = breadth_return.gt(0).where(breadth_return.notna()).mean(axis=1)

    dispersion_return = prices / prices.shift(params.dispersion_window) - 1
    dispersion = dispersion_return.std(axis=1, ddof=1, skipna=True)

    return pd.DataFrame(
        {
            "market_momentum": market_momentum,
            "market_vol": market_vol,
            "market_drawdown": market_drawdown,
            "breadth": breadth,
            "dispersion": dispersion,
        },
        index=prices.index,
    )


def _market_model_statistics(
    stock_return: pd.Series,
    market_return: pd.Series,
    window: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute alpha, beta and annualized residual volatility of a rolling market regression."""
    stock_mean = stock_return.rolling(window, min_periods=window).mean()
    market_mean = market_return.rolling(window, min_periods=window).mean()
    stock_var = stock_return.rolling(window, min_periods=window).var(ddof=1)
    market_var = market_return.rolling(window, min_periods=window).var(ddof=1)
    covariance = stock_return.rolling(window, min_periods=window).cov(market_return)
    valid_market_var = market_var.where(market_var > 0)
    beta = covariance / valid_market_var
    alpha = stock_mean - beta * market_mean
    residual_var = (
        stock_var - covariance.pow(2) / valid_market_var
    ).clip(lower=0)
    annualized_idiosyncratic_vol = np.sqrt(
        residual_var * TRADING_DAYS_PER_YEAR
    )
    return alpha, beta, annualized_idiosyncratic_vol


def _stock_features(
    prices: pd.DataFrame,
    market: pd.Series,
    params: KNNParams,
) -> dict[str, pd.DataFrame]:
    """Compute the stock state features for each stock on each trading day."""
    stock_return = prices.pct_change(fill_method=None)
    market_return = market.pct_change(fill_method=None)
    market_momentum = market / market.shift(params.stock_momentum_window) - 1
    stock_momentum = prices / prices.shift(params.stock_momentum_window) - 1
    relative_momentum = stock_momentum.sub(market_momentum, axis=0)
    stock_drawdown = (
        prices
        / prices.rolling(
            params.stock_drawdown_window,
            min_periods=params.stock_drawdown_window,
        ).max()
        - 1
    )

    result: dict[str, pd.DataFrame] = {}
    for ticker in prices.columns:
        alpha, beta, annualized_idio_vol = _market_model_statistics(
            stock_return[ticker], market_return, params.stock_vol_window
        )
        # Explain the day's return with the rolling regression coefficients available the previous day, so that the day's return
        # is not used both in estimating the coefficients and in computing the residual. The beta estimation window is stock_vol_window;
        # residual momentum is the sum of regression residuals over the last stock_momentum_window days.
        expected_return = alpha.shift(1) + beta.shift(1) * market_return
        residual_return = stock_return[ticker] - expected_return
        residual_momentum = residual_return.rolling(
            params.stock_momentum_window,
            min_periods=params.stock_momentum_window,
        ).sum()
        result[ticker] = pd.DataFrame(
            {
                "relative_momentum": relative_momentum[ticker],
                "residual_momentum": residual_momentum,
                "annualized_idiosyncratic_vol": annualized_idio_vol,
                "stock_drawdown": stock_drawdown[ticker],
            },
            index=prices.index,
        )
    return result


def _robust_scale(
    reference: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Standardize reference and target with the median and MAD of reference."""
    median = reference.median(axis=0, skipna=True)
    mad = reference.sub(median, axis=1).abs().median(axis=0, skipna=True)
    scale = 1.4826 * mad

    # When MAD is zero but the variable is not constant, fall back to the sample standard deviation; truly constant dimensions are excluded from the distance.
    fallback = reference.std(axis=0, ddof=1, skipna=True)
    scale = scale.where(scale > 0, fallback)
    usable = scale.index[scale.notna() & (scale > 0) & target.notna()]
    if len(usable) == 0:
        return reference.iloc[:, 0:0], target.iloc[0:0]
    return (
        reference[usable]
        .sub(median[usable], axis=1)
        .div(scale[usable], axis=1)
        .astype("float64"),
        target[usable].sub(median[usable]).div(scale[usable]).astype("float64"),
    )


def _select_spaced_neighbors(
    distances: pd.Series,
    positions: pd.Series,
    n_neighbors: int,
    gap: int,
) -> list[pd.Timestamp]:
    """Select neighbors by ascending distance, requiring any two neighbors to be at least gap trading days apart."""
    selected: list[pd.Timestamp] = []
    selected_positions: list[int] = []
    for date in distances.sort_values(kind="stable").index:
        position = int(positions.loc[date])
        if all(abs(position - other) >= gap for other in selected_positions):
            selected.append(pd.Timestamp(date))
            selected_positions.append(position)
            if len(selected) == n_neighbors:
                break
    return selected


def predict_return_details(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    benchmarks: pd.DataFrame | None,
    as_of: pd.Timestamp,
    horizon: int,
    params: KNNParams,
) -> KNNResult:
    """Predict each stock's return over the next horizon trading days from two-stage historically similar states.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices, may be the full sample; only rows with date ≤ as_of are used. Missing prices are not filled.
    fundamentals : long-format fundamentals table | None
        Not used in the current version; kept to match the repository's common interface.
    benchmarks : DataFrame[date × benchmark] | None
        Index closing levels, may be the full sample; must contain ``params.market``, only rows with date ≤ as_of are used.
    as_of : Timestamp
        Decision date, ∈ prices.index.
    horizon : int
        Prediction horizon H (trading days), ≥ 1; historical states s must satisfy s + H < as_of.
    params : KNNParams
        Two-stage KNN parameters, passed explicitly from ``config.py``.

    Returns
    -------
    KNNResult
        `predicted_return` is the raw horizon-day predicted simple return; `outcome_dispersion` is 1.4826 × MAD of the neighbors'
        forward returns; `uncertainty` is the approximate standard error of the sample-median prediction,
        `outcome_dispersion * sqrt(pi / (2 * K))`; `neighbor_count` is the actual number of neighbors.
        Each Series has the same index as prices.columns; `neighbors` stores the historical dates actually used.
        The prediction is NaN when the current state is unavailable or there is not a single valid neighbor; with only one neighbor a prediction is still returned, but dispersion
        and uncertainty cannot be estimated and stay NaN. When fewer than `min_neighbors` neighbors remain after de-duplication, the existing window is still used for prediction
        and a summary warning is emitted.

    Raises
    ------
    ValueError
        Invalid inputs, parameters or market index; fewer than ``min_neighbors`` valid historical candidates for the market state.
    """
    del fundamentals  # The current version uses only price and market data; the parameter is kept for compatibility with the repository's common interface.
    _validate_inputs(prices, benchmarks, as_of, horizon, params)
    assert benchmarks is not None  # Already checked by _validate_inputs.

    prices = prices.loc[:as_of].copy()
    benchmarks = benchmarks.loc[:as_of]
    market = benchmarks[params.market].reindex(prices.index)
    as_of_position = prices.index.get_loc(as_of)

    market_features = _market_features(prices, market, params)
    stock_features = _stock_features(prices, market, params)

    monthly_dates = _monthly_first_dates(prices.index)
    monthly_positions = pd.Series(
        prices.index.get_indexer(monthly_dates), index=monthly_dates, dtype="int64"
    )
    # Strictly require s + H < as_of; when equal to as_of, standing at the as_of close one cannot yet know a label that resolves after that close.
    eligible_dates = monthly_dates[monthly_positions.to_numpy() + horizon < as_of_position]
    market_reference = market_features.reindex(eligible_dates).dropna(how="any")
    market_target = market_features.loc[as_of]
    market_scaled, target_scaled = _robust_scale(market_reference, market_target)
    if market_scaled.shape[1] == 0 or target_scaled.isna().any():
        raise ValueError(f"market state features are unavailable at as_of={as_of.date()}")
    market_scaled = market_scaled.dropna(how="any")
    if len(market_scaled) < params.min_neighbors:
        raise ValueError(
            f"only {len(market_scaled)} valid market state candidates, fewer than min_neighbors={params.min_neighbors}"
        )

    market_distance = np.sqrt(
        market_scaled.sub(target_scaled, axis=1).pow(2).mean(axis=1)
    ).sort_values(kind="stable")
    market_pool = market_distance.iloc[: params.market_pool_size].index
    position_lookup = pd.Series(
        prices.index.get_indexer(market_pool), index=market_pool, dtype="int64"
    )
    future_return = prices.shift(-horizon).div(prices).sub(1)

    predicted = pd.Series(np.nan, index=prices.columns, dtype="float64", name="predicted_return")
    outcome_dispersion = pd.Series(
        np.nan, index=prices.columns, dtype="float64", name="outcome_dispersion"
    )
    uncertainty = pd.Series(np.nan, index=prices.columns, dtype="float64", name="uncertainty")
    neighbor_count = pd.Series(0, index=prices.columns, dtype="int64", name="neighbor_count")
    neighbors: dict[str, tuple[pd.Timestamp, ...]] = {}
    thin_neighbors: dict[str, int] = {}
    gap = horizon if params.neighbor_gap is None else params.neighbor_gap

    for ticker in prices.columns:
        current_stock_state = stock_features[ticker].loc[as_of]
        full_stock_reference = stock_features[ticker].reindex(market_scaled.index)
        valid_reference = full_stock_reference.dropna(how="any")
        valid_reference = valid_reference.loc[
            future_return.loc[valid_reference.index, ticker].notna()
        ]
        stock_scaled, current_stock_scaled = _robust_scale(
            valid_reference, current_stock_state
        )
        if stock_scaled.shape[1] == 0 or current_stock_scaled.isna().any():
            neighbors[ticker] = ()
            continue

        usable_pool = market_pool.intersection(stock_scaled.dropna(how="any").index, sort=False)
        usable_pool = usable_pool[
            future_return.loc[usable_pool, ticker].notna().to_numpy()
        ]
        if len(usable_pool) == 0:
            neighbors[ticker] = ()
            continue

        stock_distance = np.sqrt(
            stock_scaled.loc[usable_pool]
            .sub(current_stock_scaled, axis=1)
            .pow(2)
            .mean(axis=1)
        )
        combined_distance = np.sqrt(
            params.market_distance_weight * market_distance.loc[usable_pool].pow(2)
            + params.stock_distance_weight * stock_distance.pow(2)
        )
        selected = _select_spaced_neighbors(
            combined_distance,
            position_lookup.loc[usable_pool],
            params.n_neighbors,
            gap,
        )
        neighbors[ticker] = tuple(selected)
        if not selected:
            continue
        if len(selected) < params.min_neighbors:
            thin_neighbors[ticker] = len(selected)

        outcomes = future_return.loc[selected, ticker]
        count = len(selected)
        estimate = float(outcomes.median())
        predicted.loc[ticker] = estimate
        neighbor_count.loc[ticker] = count
        if count >= 2:
            dispersion = float(1.4826 * (outcomes - estimate).abs().median())
            outcome_dispersion.loc[ticker] = dispersion
            uncertainty.loc[ticker] = dispersion * np.sqrt(np.pi / (2.0 * count))

    if thin_neighbors:
        counts = ", ".join(f"{ticker}={count}" for ticker, count in thin_neighbors.items())
        warnings.warn(
            f"as_of={as_of.date()}: after time de-duplication, neighbors are fewer than min_neighbors="
            f"{params.min_neighbors}; still predicting with the existing window: {counts}",
            UserWarning,
            stacklevel=2,
        )

    return KNNResult(
        predicted_return=predicted,
        outcome_dispersion=outcome_dispersion,
        uncertainty=uncertainty,
        neighbor_count=neighbor_count,
        neighbors=neighbors,
    )


def predict_return(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    benchmarks: pd.DataFrame | None,
    as_of: pd.Timestamp,
    horizon: int,
    params: KNNParams,
) -> pd.Series:
    """Return each stock's raw two-stage KNN predicted return over the next horizon trading days.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices, may be the full sample; only rows with date ≤ as_of are used.
    fundamentals : long-format fundamentals table | None
        Not used in the current version.
    benchmarks : DataFrame[date × benchmark] | None
        Index closing levels, must contain ``params.market``; only rows with date ≤ as_of are used.
    as_of : Timestamp
        Decision date, ∈ prices.index.
    horizon : int
        Prediction horizon H (trading days), ≥ 1.
    params : KNNParams
        Two-stage KNN parameters, passed explicitly from ``config.py``.

    Returns
    -------
    Series[ticker]
        Raw horizon-day predicted simple return (decimal, not annualized), same index as prices.columns; NaN for stocks whose current state is unavailable or
        that have no valid neighbor; with fewer neighbors than the minimum it still predicts and emits a summary warning. This raw value is kept by ``scoring`` as predicted_return and
        used by ``weights`` as μ; the cross-sectional standardization used for selection is done by ``scoring.standardize``.
    """
    return predict_return_details(
        prices,
        fundamentals,
        benchmarks,
        as_of,
        horizon,
        params,
    ).predicted_return

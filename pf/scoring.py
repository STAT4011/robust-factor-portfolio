"""scoring — factor values on as_of, two-layer standardization and weighted total score.

Input: truncated prices / fundamentals / benchmarks, as_of, per-factor parameters, weights, clipping multiple n.
Output: score table DataFrame[ticker × {score, predicted_return}].
The caller calls ``factor_values`` (step 1) and then ``combine`` (steps 2–4), passing only enabled factors to ``factor_values``.

1. Factor values: ``factor_values`` calls ``factors.FACTORS`` for each factor in factor_params; benchmarks is forwarded only to
   factors in ``factors.EXTRA_DATA_FACTORS`` (ret).
2. Per-factor standardization: ``FACTOR_STANDARDIZERS[k]``, one function per factor, signature ``g(x: Series[ticker]) -> Series[ticker]``.
3. Common standardization: ``standardize(x, n)``, shared by all factors: MAD winsorization (clipping multiple n) followed by a cross-sectional z-score.
4. Combination: ``combine``, score = Σ_k σ_k · z_k. Factors are already higher-is-better, so direction is not handled; the total score is NaN if any z_k is NaN.
   Also outputs predicted_return: the raw value of the ret factor (i.e. ``knn.predict_return``), without calling knn again.

Design status: all factors use the same standardization (step 3); step 2 is kept as an extension point and is currently identity for all factors (output = input).
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from pf.factors import EXTRA_DATA_FACTORS, FACTORS


def standardize_momentum(x: pd.Series) -> pd.Series:
    """Per-factor standardization of momentum: identity (all factors share ``standardize``).

    Parameters
    ----------
    x : Series[ticker]
        Raw momentum values on as_of.

    Returns
    -------
    Series[ticker]
        Same as x.
    """
    return x


def standardize_downside_vol(x: pd.Series) -> pd.Series:
    """Per-factor standardization of downside_vol: identity (all factors share ``standardize``).

    Parameters
    ----------
    x : Series[ticker]
        Raw downside_vol values on as_of.

    Returns
    -------
    Series[ticker]
        Same as x.
    """
    return x


def standardize_revenue_growth(x: pd.Series) -> pd.Series:
    """Per-factor standardization of revenue_growth: identity (all factors share ``standardize``).

    Parameters
    ----------
    x : Series[ticker]
        Raw revenue_growth values on as_of.

    Returns
    -------
    Series[ticker]
        Same as x.
    """
    return x


def standardize_ret(x: pd.Series) -> pd.Series:
    """Per-factor standardization of ret: identity (all factors share ``standardize``).

    Parameters
    ----------
    x : Series[ticker]
        Raw ret values on as_of.

    Returns
    -------
    Series[ticker]
        Same as x.
    """
    return x


# Factor name → that factor's per-factor standardization function; keys must match factors.FACTORS.
FACTOR_STANDARDIZERS: dict[str, Callable[[pd.Series], pd.Series]] = {
    "momentum": standardize_momentum,
    "downside_vol": standardize_downside_vol,
    "revenue_growth": standardize_revenue_growth,
    "ret": standardize_ret,
}


def standardize(x: pd.Series, n: float) -> pd.Series:
    """Common standardization: first clip at median ± n·MAD, then take a cross-sectional z-score, MAD = median(|x − median(x)|).

    median, MAD, mean and standard deviation (ddof = 1) are all computed only over the non-NaN values of x.

    Parameters
    ----------
    x : Series[ticker]
        One factor on as_of after per-factor standardization; all stocks form one cross-section.
    n : float
        Clipping multiple, > 0; MAD is not multiplied by a constant such as 1.4826.

    Returns
    -------
    Series[ticker]
        Clipped z-score (dimensionless); same index as x.index.
        Stocks where x is NaN stay NaN. No clipping when MAD = 0 (more than half of the values are identical);
        when all available values are identical after clipping (including only one available value), all available stocks are 0; all NaN when x is all NaN.

    Raises
    ------
    ValueError
        n ≤ 0.
    """
    if not n > 0:
        raise ValueError(f"n must be > 0, got {n}")
    med = x.median()
    mad = (x - med).abs().median()
    # When MAD = 0 the bounds collapse to the median, which would squash data that still varies into a constant, so do not clip.
    w = x.clip(med - n * mad, med + n * mad) if mad > 0 else x
    # No cross-sectional variation = the factor carries no information today; record 0 instead of 0/0 = NaN,
    # otherwise a constant factor would make every stock's total score NaN.
    if w.dropna().nunique() <= 1:
        return w * 0.0
    return (w - w.mean()) / w.std(ddof=1)


def factor_values(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame | None,
    as_of: pd.Timestamp,
    factor_params: dict[str, dict[str, Any]],
    *,
    benchmarks: pd.DataFrame | None,
    precomputed: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Raw values of each factor on as_of: call ``FACTORS[k]`` for each factor in factor_params.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices, already truncated.
    fundamentals : long-format fundamentals table | None
        Already truncated; must not be None when a factor that needs it is enabled.
    as_of : Timestamp
        Decision date, ∈ prices.index.
    factor_params : dict[str, dict[str, Any]]
        Factor name → all parameters of that factor (including downside_vol's periods_per_year and ret's horizon merged in from the B class). Keys ⊆ ``FACTORS``.
    benchmarks : DataFrame[date × benchmark] | None
        Index closing levels, already truncated. Forwarded only to factors in ``EXTRA_DATA_FACTORS``.
    precomputed : dict[str, Series[ticker]] | None
        Already computed factor values; keys must belong to factor_params. Lets backtest reuse the predicted_return from the same KNN run.

    Returns
    -------
    DataFrame[ticker × factor]
        Same index as prices.columns, columns in the order of factor_params; NaN as in each factor function.

    Raises
    ------
    ValueError
        factor_params is empty or contains factors not in ``FACTORS``.
    """
    if not factor_params:
        raise ValueError("factor_params must not be empty")
    unknown = set(factor_params) - set(FACTORS)
    if unknown:
        raise ValueError(f"unregistered factors: {sorted(unknown)}")
    precomputed = {} if precomputed is None else precomputed
    unknown_precomputed = set(precomputed) - set(factor_params)
    if unknown_precomputed:
        raise ValueError(f"precomputed contains factors that are not enabled: {sorted(unknown_precomputed)}")
    extra = {"benchmarks": benchmarks}
    columns: dict[str, pd.Series] = {}
    for key, params in factor_params.items():
        if key in precomputed:
            columns[key] = precomputed[key].reindex(prices.columns)
        else:
            columns[key] = FACTORS[key](
                prices,
                fundamentals,
                as_of,
                **params,
                **(extra if key in EXTRA_DATA_FACTORS else {}),
            )
    return pd.DataFrame(columns, index=prices.columns)


def combine(values: pd.DataFrame, weights: dict[str, float], n: float) -> pd.DataFrame:
    """Score table: total score S = Σ_k σ_k · standardize(g_k(x_k), n) (summed over the factors in weights), plus the raw ret value.

    Parameters
    ----------
    values : DataFrame[ticker × factor]
        Raw values of each factor on as_of, columns ⊇ keys of weights.
    weights : dict[str, float]
        Factor name → weight σ_k; the keys determine which factors are enabled, ⊆ ``FACTOR_STANDARDIZERS``.
    n : float
        MAD clipping multiple, > 0, shared by all factors; see ``standardize``.

    Returns
    -------
    DataFrame[ticker × {score, predicted_return}]
        Same index as values.index (named ticker), not sorted.
        score: total score (dimensionless); NaN for stocks where any enabled factor is NaN.
        predicted_return: values["ret"] taken as is (knn's return prediction, not standardized);
        all NaN when values has no ret (not enabled).

    Raises
    ------
    ValueError
        weights is empty, or contains factors missing from values or from ``FACTOR_STANDARDIZERS``; n ≤ 0.
    """
    if not weights:
        raise ValueError("weights must not be empty")
    missing = set(weights) - set(values.columns)
    unknown = set(weights) - set(FACTOR_STANDARDIZERS)
    if missing or unknown:
        raise ValueError(f"missing from values: {sorted(missing)}; no registered standardizer: {sorted(unknown)}")
    z = pd.DataFrame(
        {k: standardize(FACTOR_STANDARDIZERS[k](values[k]), n) for k in weights},
        index=values.index,
    )
    # skipna=False: missing one factor makes the whole stock NaN; weights are not redistributed over the remaining factors.
    score = (z * pd.Series(weights)).sum(axis=1, skipna=False)
    # Take the raw ret value instead of calling knn again: knn is computed only once, in factor_values.
    pred = values["ret"] if "ret" in values.columns else pd.Series(np.nan, index=values.index)
    return pd.DataFrame({"score": score, "predicted_return": pred}, index=values.index).rename_axis("ticker")


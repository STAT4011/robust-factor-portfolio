"""weights — compute position weights, one as_of per call.

Methods are registered in ``WEIGHTERS``, with a common signature::

    w(prices, predicted_return, prediction_uncertainty, selected, as_of, **params) -> Series[ticker]

- prices: DataFrame[date × ticker], already truncated (passing the full sample gives the same result: only rows on and before as_of are used).
- predicted_return: Series[ticker], the predicted_return column of the score table (knn's return prediction over the H-day holding period).
- prediction_uncertainty: Series[ticker], the standard error of the predicted median from the same knn run.
- selected: selected stock symbols.
- Output: index equals ``selected + ["CASH"]``; CASH is a free cash position with zero return and zero volatility.

Unlike other modules, when selected stocks have incomplete data (NaN predicted return, missing prices in the window), this **raises instead of outputting NaN**:
the weights are a joint optimization, so we cannot set only the problematic stock to NaN, and we do not re-optimize over the remaining stocks.

Implemented: robust_mvo (robust mean-variance with an ellipsoidal uncertainty set). Parameters are registered in ``config.WeightsParams``.
"""

from collections.abc import Callable

import cvxpy as cp
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

CASH = "CASH"


def robust_mvo(
    prices: pd.DataFrame,
    predicted_return: pd.Series,
    prediction_uncertainty: pd.Series,
    selected: list[str],
    as_of: pd.Timestamp,
    lookback: int,
    horizon: int,
    kappa: float,
    max_volatility: float,
    lower: float,
    upper: float,
) -> pd.Series:
    """Robust mean-variance: max μᵀw − κ·√(wᵀΩw), with stock and free cash weights summing to 1.

    Σ = horizon × Ledoit–Wolf(the lookback daily returns up to as_of),
    Ω = diag(prediction_uncertainty²). Cash return, prediction uncertainty and volatility are all 0,
    so only stock weights w are optimized, with w_cash = 1 − Σ_i w_i.

    Parameters
    ----------
    prices : DataFrame[date × ticker]
        Close prices. Only the selected columns of as_of and the lookback + 1 rows before it are used.
    predicted_return : Series[ticker]
        μ: predicted simple return over the H-day holding period (decimal, not annualized), on the same horizon basis as Σ; index ⊇ selected.
    prediction_uncertainty : Series[ticker]
        u: KNN's standard error of μ, on the same H-day return basis as μ; Ω = diag(u²), index ⊇ selected.
    selected : list[str]
        Selected stock symbols, non-empty, no duplicates, ⊆ prices.columns.
    as_of : Timestamp
        Decision date, ∈ prices.index.
    lookback : int
        Number of daily returns used to estimate Σ, ≥ 2.
    horizon : int
        Holding period H (trading days), ≥ 1; the daily covariance is multiplied by it to get the H-day covariance.
    kappa : float
        Uncertainty set radius κ, ≥ 0: in the worst case μ deviates by about κ H-day standard deviations; 0 is plain mean-variance.
    max_volatility : float
        Maximum portfolio volatility over the horizon holding period (decimal), > 0; the constraint is wᵀΣw ≤ max_volatility².
    lower, upper : float
        Lower and upper bounds on single-stock weights, must satisfy 0 ≤ lower ≤ upper and lower · len(selected) ≤ 1;
        this method always forbids short selling. The cash weight can vary freely in [0, 1] and is not bound by the stock upper limit.

    Returns
    -------
    Series[ticker]
        Weights (decimal); index equals ``selected + ["CASH"]``. Stock weights lie in [lower, upper] (values within
        1e-8 of a bound are set to the bound), CASH ≥ 0; all weights sum to 1 within solver precision (about 1e-7).

    Raises
    ------
    ValueError
        Parameters out of range or constraints infeasible; selected is empty, has duplicates, contains ``CASH``, or is not in prices / predicted_return / prediction_uncertainty;
        fewer than lookback rows before as_of; μ or uncertainty of a selected stock is invalid, or prices are missing in the window.
    RuntimeError
        The solver did not reach an optimal solution.
    """
    n = len(selected)
    if lookback < 2 or horizon < 1 or kappa < 0 or max_volatility <= 0:
        raise ValueError(
            "lookback ≥ 2, horizon ≥ 1, kappa ≥ 0 and max_volatility > 0 are required, "
            f"got lookback={lookback}, horizon={horizon}, kappa={kappa}, "
            f"max_volatility={max_volatility}"
        )
    if n == 0 or len(set(selected)) != n:
        raise ValueError(f"selected must be non-empty without duplicates, got {selected}")
    if CASH in selected:
        raise ValueError(f"selected must not contain the cash symbol {CASH!r}; cash is added automatically by this function")
    if not (0 <= lower <= upper and n * lower <= 1):
        raise ValueError(
            f"long-only weight constraints are infeasible for {n} stocks plus free cash: "
            f"lower={lower}, upper={upper}"
        )
    missing = sorted(
        set(selected) - set(prices.columns)
        | set(selected) - set(predicted_return.index)
        | set(selected) - set(prediction_uncertainty.index)
    )
    if missing:
        raise ValueError(
            f"missing from prices, predicted_return or prediction_uncertainty: {missing}"
        )
    t = prices.index.get_loc(as_of)
    if t < lookback:
        raise ValueError(f"fewer than lookback={lookback} rows before as_of={as_of.date()}")
    p = prices.iloc[t - lookback : t + 1][selected]
    r = (p / p.shift(1)).iloc[1:] - 1
    mu = predicted_return[selected]
    uncertainty = prediction_uncertainty[selected]
    bad_prices = set(r.columns[r.isna().any()])
    bad_mu = set(mu.index[~np.isfinite(mu.to_numpy())])
    bad_uncertainty = set(
        uncertainty.index[
            ~np.isfinite(uncertainty.to_numpy()) | (uncertainty.to_numpy() < 0)
        ]
    )
    bad = sorted(bad_prices | bad_mu | bad_uncertainty)
    if bad:
        raise ValueError(
            "invalid predicted return / prediction uncertainty or missing prices in the lookback window: "
            f"{bad}"
        )

    sigma = horizon * LedoitWolf().fit(r.to_numpy()).covariance_
    sigma_psd = cp.psd_wrap(sigma)
    w = cp.Variable(n)
    objective = (mu.to_numpy() @ w
                 - kappa * cp.norm(cp.multiply(uncertainty.to_numpy(), w), 2))
    # psd_wrap: the matrix is necessarily positive definite after LW shrinkage, so skip cvxpy's PSD check, which is sensitive to floating-point error.
    constraints = [
        cp.sum(w) <= 1,
        w >= lower,
        w <= upper,
        cp.quad_form(w, sigma_psd) <= max_volatility ** 2,
    ]
    problem = cp.Problem(cp.Maximize(objective), constraints)
    # Fix the solver so results are consistent across platforms.
    problem.solve(solver=cp.CLARABEL)
    if problem.status != cp.OPTIMAL:
        raise RuntimeError(f"solver status {problem.status}")
    # Interior-point solutions satisfy constraints only to about 1e-8: without rounding they would leave fake positions like 5e-10, −1e-9 and negative cash.
    tol = 1e-8
    risky = pd.Series(np.asarray(w.value).reshape(-1), index=selected).clip(lower, upper)
    risky = risky.mask(risky - lower < tol, lower).mask(upper - risky < tol, upper)
    cash = 1.0 - float(risky.sum())
    if cash < -1e-7:
        raise RuntimeError(f"solver returned a negative cash weight {cash}")
    return pd.concat([risky, pd.Series({CASH: cash if cash >= tol else 0.0})])


WEIGHTERS: dict[str, Callable[..., pd.Series]] = {
    "robust_mvo": robust_mvo,
}

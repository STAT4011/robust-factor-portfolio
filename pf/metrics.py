"""metrics — evaluation metrics and saving / loading run records.

- Per-period detail ``period_table``: ``BacktestResult`` → DataFrame[as_of × <metric>.<subject>], with ret / vol / sharpe / mdd for strategy S and each baseline,
  beta / alpha / n_sel / w_cash for S, and the rank of S and ew_sel among random portfolios p_ret / p_sharpe. Pure function.
- Summary ``evaluate``: cross-period statistics from the per-period detail, returns ``dict[str, float]`` keyed by ``metric_keys(cfg)``
  (``<metric>.<subject>.<statistic>``, plus ``n_periods`` and ``n_cash_only``). Pure function.
- Save ``save_run``: the only function in the project that writes to ``outputs/``. Each run writes one json
  ``<runs_dir>/<YYYYMMDD_HHMMSS>_<label>_<4-char random string>.json`` (the random string avoids name clashes when several people use the same label in the same second), with five sections: ``meta`` (run time, git commit, label, note),
  ``run`` (start, end, step), ``config`` (Config, nested as dataclasses), ``metrics``,
  ``periods`` (as_of → last day of the holding period, period return, per-period detail columns except ret.S, entry weights). Portfolio daily returns, baselines and random portfolios are not stored:
  they can be recomputed from weights, prices and seeds (the same price data must be used; floating-point summation order may introduce rounding error). NaN is stored as json ``null``.
- Load ``load_run``: read back one run (Config / RunSpec restored losslessly); ``load_runs``: aggregate all records in the directory into a table,
  one row per run with parameters expanded into columns, for tuning comparisons. No separate summary file is saved.

For all-cash periods (std(r^S) = 0), sharpe.S, dsr and p_sharpe.S are NaN; they are skipped in the summary and counted in ``n_cash_only``.
"""

import json
import math
import secrets
import subprocess
import typing
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, get_type_hints

import numpy as np
import pandas as pd

from pf.backtest import BacktestResult, baseline_keys
from pf.config import LABEL_ILLEGAL, PROJECT_ROOT, Config, RunSpec
from pf.weights import CASH

# Absolute metrics shared by all portfolios in the per-period detail.
_ABSOLUTE = ("ret", "vol", "sharpe", "mdd")


def period_table(result: BacktestResult, cfg: Config) -> pd.DataFrame:
    """Per-period metrics: absolute metrics of strategy S and each deterministic baseline, S's beta / alpha / count / cash, and random portfolio ranks.

    Parameters
    ----------
    result : BacktestResult
        Output of ``backtest.run_backtest``; uses daily, weights, baseline_daily, random_daily, rf_daily.
    cfg : Config
        Uses ``task.periods_per_year`` (A), ``task.holding_days`` (H), ``task.eval`` and ``data.benchmark`` (M).

    Returns
    -------
    DataFrame[as_of × column]
        Same index as ``result.periods.index``; column names are ``<metric>.<subject>``, in order ret.S, vol.S, sharpe.S, mdd.S,
        beta.S, alpha.S, n_sel.S, w_cash.S, p_ret.S, p_sharpe.S, p_ret.ew_sel, p_sharpe.ew_sel,
        then for each baseline b (``backtest.baseline_keys``) ret.b, vol.b, sharpe.b, mdd.b.

        - ret = V_H − 1 (decimal, H-day cumulative, not annualized; ret.S is period_return); vol = std(r)·√A (ddof = 1, annualized);
  sharpe = mean(r − rf) / std(r) · √A (annualized); mdd = max_d (1 − V_d / max_{j≤d} V_j), including V_0 = 1 (decimal, ≥ 0).
        - beta, alpha: OLS with intercept of r^S − rf = a + β (r^M − rf) + ε, alpha = a·A (annualized). All-cash periods are regressed as usual,
  giving beta ≈ 0 and alpha ≈ −mean(rf)·A (only approximate when rf varies day to day).
        - n_sel: number of selected stocks (including those with weight 0); w_cash: CASH weight at entry.
        - p_ret, p_sharpe: number of random portfolios in the period whose ret / sharpe is strictly less than X × 100 / n_random (∈ [0, 100]).

        NaN: if any daily return of a portfolio is NaN, all its metrics for that period and ranks involving it are NaN; if rf is NaN on any day of the holding period,
        sharpe of all portfolios and beta, alpha, p_sharpe are NaN; when std(r^S) = 0 (all cash), sharpe.S and p_sharpe.S are NaN.
    """
    A, H, ev = cfg.task.periods_per_year, cfg.task.holding_days, cfg.task.eval
    keys = baseline_keys(ev)
    market = keys[list(ev.index_baselines).index(cfg.data.benchmark)]
    rows = {}
    for as_of in result.periods.index:
        dates = result.daily.index[result.daily.index > as_of][:H]
        rf = result.rf_daily.reindex(dates)
        r_s = result.daily.loc[dates, as_of]
        portfolios = pd.DataFrame({"S": r_s, **{b: result.baseline_daily[b].loc[dates, as_of] for b in keys}})
        absolute = _absolute(portfolios, rf, A)
        rand = _absolute(result.random_daily[as_of], rf, A)

        w = result.weights.loc[as_of]
        row = {f"{m}.S": absolute.loc["S", m] for m in _ABSOLUTE}
        row["beta.S"], row["alpha.S"] = _beta_alpha(r_s, portfolios[market], rf, A)
        row["n_sel.S"] = int(w.drop(CASH).notna().sum())
        row["w_cash.S"] = float(w[CASH])
        for x in ("S", "ew_sel"):
            for m in ("ret", "sharpe"):
                row[f"p_{m}.{x}"] = _rank(absolute.loc[x, m], rand[m])
        for b in keys:
            row.update({f"{m}.{b}": absolute.loc[b, m] for m in _ABSOLUTE})
        rows[as_of] = row
    columns = [*(f"{m}.S" for m in _ABSOLUTE), "beta.S", "alpha.S", "n_sel.S", "w_cash.S",
               "p_ret.S", "p_sharpe.S", "p_ret.ew_sel", "p_sharpe.ew_sel",
               *(f"{m}.{b}" for b in keys for m in _ABSOLUTE)]
    table = pd.DataFrame.from_dict(rows, orient="index", columns=columns).astype("float64")
    return table.astype({"n_sel.S": "int64"}).rename_axis(index="as_of")


def metric_keys(cfg: Config) -> list[str]:
    """Keys output by ``evaluate``, in a fixed order: count keys first, the rest in the table order of §4.1.

    Parameters
    ----------
    cfg : Config
        Uses ``task.eval`` (baselines and quantiles determine the key names).

    Returns
    -------
    list[str]
        ``n_periods``, ``n_cash_only``, then ``<metric>.<subject>.<statistic>``; 128 in total with default parameters.
    """
    return ["n_periods", "n_cash_only", *(f"{m}.{x}.{st}" for m, x, st in _summary_keys(cfg))]


def evaluate(result: BacktestResult, cfg: Config) -> dict[str, float]:
    """Cross-period summary: mean, median, quantiles, win rate and standard error of each metric from the per-period detail of ``period_table``.

    Parameters
    ----------
    result : BacktestResult
        Output of ``backtest.run_backtest``.
    cfg : Config
        Same as ``period_table``; quantiles are taken from ``task.eval.quantiles``.

    Returns
    -------
    dict[str, float]
        Keys are ``metric_keys(cfg)`` (same order). ``n_periods``: number of periods; ``n_cash_only``: number of all-cash periods (vol.S = 0).
        Other keys ``<metric>.<subject>.<statistic>``:

        - Relative metrics er.b = ret.S − ret.b, dsr.b = sharpe.S − sharpe.b, dmdd.b = mdd.S − mdd.b (positive means the strategy's drawdown is deeper).
        - mean, median: arithmetic mean and median across periods; pXX: ``np.quantile`` (linear);
  win: for er, dsr the number of periods with value > 0 / n, for dmdd with value < 0 / n, for p_ret, p_sharpe with value > 50 / n (ties are not wins);
  se: std (ddof = 1) / √n, computed only when periods do not overlap (each period's end ≤ the next as_of), otherwise NaN. n is the number of periods included.
        - A statistic is NaN if any period is NaN (also NaN with 0 periods); exceptions: sharpe.S.*, dsr.*.*, p_sharpe.S.* skip all-cash periods.
    """
    table = period_table(result, cfg)
    cash_only = table["vol.S"] == 0
    columns = dict(table.items())
    for b in baseline_keys(cfg.task.eval):
        columns[f"er.{b}"] = table["ret.S"] - table[f"ret.{b}"]
        columns[f"dsr.{b}"] = table["sharpe.S"] - table[f"sharpe.{b}"]
        columns[f"dmdd.{b}"] = table["mdd.S"] - table[f"mdd.{b}"]
    end = result.periods["end"]
    non_overlapping = bool((end.to_numpy()[:-1] <= result.periods.index.to_numpy()[1:]).all())

    out: dict[str, float] = {"n_periods": len(table), "n_cash_only": int(cash_only.sum())}
    for m, x, st in _summary_keys(cfg):
        x_values = columns[f"{m}.{x}"]
        # All-cash periods have no Sharpe ratio: these items skip them, other metrics include them as usual.
        if (m, x) in (("sharpe", "S"), ("p_sharpe", "S")) or m == "dsr":
            x_values = x_values[~cash_only]
        out[f"{m}.{x}.{st}"] = _statistic(x_values.to_numpy(dtype="float64"), m, st, non_overlapping)
    return out


def _summary_keys(cfg: Config) -> list[tuple[str, str, str]]:
    """(metric, subject, statistic) of the summary keys, in the table order of §4.1.

    Parameters
    ----------
    cfg : Config
        Uses ``task.eval``.

    Returns
    -------
    list[tuple[str, str, str]]
        Quantile statistic names are ``p`` + percent (0.1 → p10).
    """
    q = [f"p{round(v * 100, 10):g}" for v in cfg.task.eval.quantiles]
    keys = baseline_keys(cfg.task.eval)
    groups = [
        (("ret", "vol", "sharpe", "mdd"), ("S",), ("mean", "median", *q)),
        (("er", "dsr", "dmdd"), keys, ("mean", "median", *q, "win", "se")),
        (("beta", "alpha"), ("S",), ("median", *q)),
        (("p_ret", "p_sharpe"), ("S", "ew_sel"), ("mean", "median", "win")),
        (("n_sel", "w_cash"), ("S",), ("mean",)),
    ]
    return [(m, x, st) for metrics, objects, stats in groups for m in metrics for x in objects for st in stats]


def _statistic(values: np.ndarray, metric: str, stat: str, non_overlapping: bool) -> float:
    """One statistic of a metric's per-period values; NaN if any value is NaN or there are no values.

    Parameters
    ----------
    values : ndarray
        Per-period values included in the statistic.
    metric : str
        Metric name; determines the direction of win.
    stat : str
        ∈ {mean, median, pXX, win, se}.
    non_overlapping : bool
        Whether the periods do not overlap; otherwise se is NaN.

    Returns
    -------
    float
        Value of the statistic.
    """
    n = len(values)
    if n == 0 or np.isnan(values).any():
        return math.nan
    if stat == "mean":
        return float(values.mean())
    if stat == "median":
        return float(np.median(values))
    if stat == "win":
        wins = {"dmdd": values < 0, "p_ret": values > 50, "p_sharpe": values > 50}.get(metric, values > 0)
        return float(wins.sum() / n)
    if stat == "se":
        return float(values.std(ddof=1) / math.sqrt(n)) if non_overlapping and n > 1 else math.nan
    return float(np.quantile(values, float(stat[1:]) / 100, method="linear"))


def _absolute(r: pd.DataFrame, rf: pd.Series, periods_per_year: int) -> pd.DataFrame:
    """ret, vol, sharpe, mdd of each portfolio for one period (formulas in ``period_table``).

    Parameters
    ----------
    r : DataFrame[date × portfolio]
        Portfolio daily simple returns (decimal) over one period's H trading days.
    rf : Series[date]
        Daily risk-free rate of the same period (decimal), same index as r.index.
    periods_per_year : int
        Number of periods per year A, > 0.

    Returns
    -------
    DataFrame[portfolio × {ret, vol, sharpe, mdd}]
        Same index as r.columns. The whole row is NaN if any day of a portfolio is NaN; sharpe is NaN if rf has NaN or std(r) = 0.
    """
    a = r.to_numpy(dtype="float64")
    v = np.vstack([np.ones(a.shape[1]), np.cumprod(1 + a, axis=0)])
    std = a.std(axis=0, ddof=1)
    out = pd.DataFrame({
        "ret": (1 + r).prod(skipna=False).to_numpy() - 1,
        "vol": std * math.sqrt(periods_per_year),
        # The denominator uses the standard deviation of raw returns: exactly 0 in all-cash periods, treated as NaN instead of dividing to inf.
        "sharpe": np.where(std > 0, (a - rf.to_numpy()[:, None]).mean(axis=0) / np.where(std > 0, std, 1.0), np.nan)
                  * math.sqrt(periods_per_year),
        "mdd": (1 - v / np.maximum.accumulate(v, axis=0)).max(axis=0),
    }, index=r.columns)
    out.loc[np.isnan(a).any(axis=0)] = np.nan
    return out


def _beta_alpha(r: pd.Series, r_market: pd.Series, rf: pd.Series, periods_per_year: int) -> tuple[float, float]:
    """OLS with intercept of r − rf = a + β (r_market − rf) + ε; returns (β, a·A).

    Parameters
    ----------
    r, r_market : Series[date]
        Daily returns of the strategy and the market index over the same period (decimal).
    rf : Series[date]
        Daily risk-free rate of the same period (decimal).
    periods_per_year : int
        Number of periods per year A, > 0.

    Returns
    -------
    tuple[float, float]
        beta and annualized alpha; both NaN if any input has NaN.
    """
    y = (r - rf).to_numpy(dtype="float64")
    x = (r_market - rf).to_numpy(dtype="float64")
    if np.isnan(y).any() or np.isnan(x).any():
        return math.nan, math.nan
    (a, beta), *_ = np.linalg.lstsq(np.column_stack([np.ones_like(x), x]), y, rcond=None)
    return float(beta), float(a * periods_per_year)


def _rank(value: float, random: pd.Series) -> float:
    """Number of random portfolios strictly below value × 100 / number of portfolios.

    Parameters
    ----------
    value : float
        Metric value of the portfolio being compared.
    random : Series[draw]
        The same metric for each random portfolio in the same period.

    Returns
    -------
    float
        ∈ [0, 100]; NaN if value or any random portfolio is NaN.
    """
    if math.isnan(value) or random.isna().any():
        return math.nan
    return float(100 * (random < value).sum() / len(random))


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One run read back by ``load_run``."""

    # Run record file.
    path: Path
    # time: YYYY-MM-DDTHH:MM:SS (local time); git_commit: commit hash or "unknown"; label, note: original text.
    meta: dict[str, str]
    run: RunSpec
    config: Config
    # Keys are metric names at save time; null is read back as NaN.
    metrics: dict[str, float]
    # DataFrame[as_of × {end, period_return, <metric>.<subject>…}]: BacktestResult.periods followed by the period_table columns except ret.S
    # (ret.S is period_return); null is read back as NaN, columns absent from the record (old records) do not appear.
    periods: pd.DataFrame
    # DataFrame[as_of × ticker]: entry weights (decimal); columns include only stocks selected at least once (alphabetical), NaN when not selected.
    weights: pd.DataFrame


def save_run(
    cfg: Config,
    run: RunSpec,
    result: BacktestResult,
    metrics: dict[str, float],
    runs_dir: Path,
) -> Path:
    """Write one run's parameters, metrics and per-period weights to a json; return the file path.

    Parameters
    ----------
    cfg : Config
        Strategy parameters of this run.
    run : RunSpec
        Rolling rule, label and note of this run; ``label`` is filtered for illegal characters (``config.LABEL_ILLEGAL``) and inserted into the file name.
    result : BacktestResult
        Output of ``backtest.run_backtest``; stores ``periods``, the ``period_table(result, cfg)`` columns except ret.S,
        and ``weights`` (each period stores the selected stocks and CASH).
    metrics : dict[str, float]
        Output of ``evaluate(result, cfg)``; NaN is stored as null.
    runs_dir : Path
        Record directory (``config.RUNS_DIR`` for real runs); created automatically if missing.

    Returns
    -------
    Path
        Newly written file ``runs_dir/<YYYYMMDD_HHMMSS>_<label>_<4-char random string>.json``; the random string is 4 lowercase hex characters.

    Raises
    ------
    ValueError
        ``run.label`` is empty after filtering illegal characters and stripping leading/trailing dots and underscores (when the label was changed to empty after constructing ``RunSpec``).
    FileExistsError
        The file name already exists (same second, same label and same random string); existing records are not overwritten.
    """
    now = datetime.now()
    label = LABEL_ILLEGAL.sub("_", run.label).strip("._")
    if not label:
        raise ValueError(f"RunSpec.label must not be empty: {run.label!r}")
    # The random string avoids name clashes between run records with the same label in the same second.
    stem = f"{now.strftime('%Y%m%d_%H%M%S')}_{label}_{secrets.token_hex(2)}"

    detail = period_table(result, cfg).drop(columns="ret.S")
    periods = {}
    # iterrows would infer a row of dates and NaN returns as datetime, turning NaN into NaT; keep each column's original dtype.
    for as_of, end, period_return in result.periods[["end", "period_return"]].itertuples(name=None):
        w = result.weights.loc[as_of].dropna()
        periods[as_of.date().isoformat()] = {
            "end": end.date().isoformat(),
            "period_return": _nan_to_none(float(period_return)),
            **{k: _nan_to_none(v if isinstance(v, int) else float(v)) for k, v in detail.loc[as_of].to_dict().items()},
            "weights": {t: float(v) for t, v in w.items()},
        }
    run_data = asdict(run)
    record = {
        "meta": {"time": now.isoformat(timespec="seconds"), "git_commit": _git_commit(),
                 "label": run_data.pop("label"), "note": run_data.pop("note")},
        "run": run_data,
        "config": asdict(cfg),
        "metrics": {k: _nan_to_none(v) for k, v in metrics.items()},
        "periods": periods,
    }

    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{stem}.json"
    # Validate as standard JSON first, so a serialization failure does not leave a corrupted run record.
    serialized = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    with open(path, "x", encoding="utf-8") as f:
        f.write(serialized)
    return path


def load_run(path: Path) -> RunRecord:
    """Read back one run written by ``save_run``; Config and RunSpec are restored losslessly as dataclasses.

    Parameters
    ----------
    path : Path
        Run record file.

    Returns
    -------
    RunRecord
        See its field descriptions. Parameters added to config.py after saving take their current defaults.

    Notes
    -----
    If the record contains parameters that no longer exist in the current config.py, the dataclass constructor raises TypeError.
    """
    with open(path, encoding="utf-8") as f:
        record = json.load(f)

    as_of = pd.DatetimeIndex(list(record["periods"]), name="as_of")
    rows = record["periods"].values()
    detail = list(dict.fromkeys(k for p in rows for k in p if k not in ("end", "period_return", "weights")))
    periods = pd.DataFrame(
        {"end": pd.DatetimeIndex([p["end"] for p in rows]),
         "period_return": [_none_to_nan(p["period_return"]) for p in rows],
         **{k: [_none_to_nan(p.get(k)) for p in rows] for k in detail}},
        index=as_of,
    )
    weights = pd.DataFrame([p["weights"] for p in rows], index=as_of, dtype="float64")
    stock_columns = sorted(c for c in weights.columns if c != CASH)
    columns = stock_columns + ([CASH] if CASH in weights.columns else [])
    weights = weights.reindex(columns=columns).rename_axis(columns="ticker")
    return RunRecord(
        path=path,
        meta=record["meta"],
        run=_from_dict(RunSpec, {**record["run"], "label": record["meta"]["label"],
                                "note": record["meta"]["note"]}),
        config=_from_dict(Config, record["config"]),
        metrics={k: _none_to_nan(v) for k, v in record["metrics"].items()},
        periods=periods,
        weights=weights,
    )


def load_runs(runs_dir: Path) -> pd.DataFrame:
    """Aggregate all run records under ``runs_dir``, one row per run, for tuning comparisons.

    Parameters
    ----------
    runs_dir : Path
        Record directory (``config.RUNS_DIR`` for real runs).

    Returns
    -------
    DataFrame[run × column]
        index is the record file name (without .json, named ``run``), ascending by file name, i.e. run time. columns are
        ``meta.*``, ``run.*``, ``metrics.*`` (in record order), ``config.*`` (nested parameters expanded with dots,
        e.g. ``config.factors.momentum.lookback``, ``config.factor_weights.ret``); no per-period detail.
        Parameters a run does not have (e.g. weights of disabled factors, parameters registered later) or metrics (old records) are NaN; null metrics are NaN.
        Empty table if the directory does not exist or is empty.
    """
    paths = sorted(runs_dir.glob("*.json")) if runs_dir.is_dir() else []
    records = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        r.pop("periods")
        r["metrics"] = {k: _none_to_nan(v) for k, v in r["metrics"].items()}
        records.append(r)
    table = pd.json_normalize(records).set_axis(pd.Index([p.stem for p in paths], name="run"))
    order = {"meta": 0, "run": 1, "metrics": 2, "config": 3}
    # Stable sort: keep record order within each section.
    cols = sorted(table.columns, key=lambda c: order.get(c.split(".")[0], 4))
    return table[cols]


def _from_dict(cls: type, d: dict[str, Any]) -> Any:
    """Restore nested dataclasses according to the current parameter classes.

    Parameters
    ----------
    cls : type
        dataclass type from config.py.
    d : dict[str, Any]
        Field names and values at save time; missing fields use cls defaults.

    Returns
    -------
    Any
        cls instance, with nested sections also restored as the corresponding dataclasses; tuple fields (arrays in json) restored as tuples.
    """
    hints = get_type_hints(cls)
    return cls(**{k: _restore_field(hints.get(k), v) for k, v in d.items()})


def _restore_field(hint: Any, v: Any) -> Any:
    """Restore one saved value according to the field's type annotation.

    Parameters
    ----------
    hint : Any
        Type annotation of the field; None if the field no longer exists (the dataclass constructor raises).
    v : Any
        Value read back from json.

    Returns
    -------
    Any
        dataclass sections are restored as instances; tuple fields are restored from json arrays as tuples; everything else as is.
    """
    if is_dataclass(hint):
        return _from_dict(hint, v)
    # json has no tuples, so tuple fields come back as lists; without restoring them the loaded Config would not equal the original.
    return tuple(v) if typing.get_origin(hint) is tuple and isinstance(v, list) else v


def _nan_to_none(x: Any) -> Any:
    """Convert a numeric NaN to a JSON-representable null.

    Parameters
    ----------
    x : Any
        Metric or period return scalar.

    Returns
    -------
    Any
        None for NaN, other values unchanged.
    """
    return None if isinstance(x, float) and math.isnan(x) else x


def _none_to_nan(x: Any) -> Any:
    """Restore a JSON null to a numeric NaN.

    Parameters
    ----------
    x : Any
        Metric or period return scalar read back from a record.

    Returns
    -------
    Any
        NaN for None, other values unchanged.
    """
    return math.nan if x is None else x


def _git_commit() -> str:
    """Read the commit hash of the current repository.

    Returns
    -------
    str
        Full commit hash; unknown if git is unavailable or not in a repository.
    """
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, encoding="utf-8", check=True)
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"

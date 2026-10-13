"""Parameter defaults and path constants.

DataSpec and TaskSpec are the fixed data source and task setup; RunSpec describes a single backtest.
Config gathers the strategy parameters; factor weights and the fields of each parameter section can be adjusted in notebooks.
Configuration uses dataclasses, which metrics can save as JSON and restore.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data_wsy"
# Run record directory: metrics.save_run writes one json per run, metrics.load_runs aggregates from here.
RUNS_DIR: Path = PROJECT_ROOT / "outputs" / "runs"
# Characters replaced by "_" in run labels: characters not allowed in Windows file names, plus path separators and whitespace.
LABEL_ILLEGAL: re.Pattern[str] = re.compile(r'[:*?"<>|/\\\s]+')


@dataclass(frozen=True, slots=True)
class DataSpec:
    """A Data source: which files to read, which price column to use, which index to compare against."""

    # Price file under data_wsy/, currently only "prices_raw.csv": from 2011-01-03, 31 stocks (this is the candidate pool; pf/ keeps no separate list).
    # Unbalanced panel: MPC, NOW, ABBV, AAOI, ONDS, TEM, CRWV listed later and are NaN before listing.
    # When a balanced panel is needed (e.g. covariance estimation), the caller runs dropna on the truncated data.
    price_file: str = "prices_raw.csv"

    # Price column; determines what "return" means throughout the pipeline. Both columns are split-adjusted, so there are no false gaps on split dates;
    # they differ only in dividends:
    #   "close"      Close price, not dividend-adjusted. Return = price change; the price drop on ex-dividend dates counts as a loss.
    #                The brief states that dividends are not counted in returns, and scoring uses this definition, so it is the default.
    #   "adj_close"  Adjusted close, additionally dividend-adjusted (assuming dividends are reinvested). Return = total return,
    #                usable for robustness checks.
    #                It is adjusted backwards from the latest day, so historical values change with every data update; only returns are meaningful,
    #                and the price level is not a real traded price.
    price_col: str = "close"

    # Quarterly revenue, including the announcement date report_date; used by the revenue growth factor.
    # Uses "revenue_adjusted.csv", corrected via SEC cross-validation ("revenue.csv" holds the uncorrected raw values).
    # Known issues remain (GE reporting break around its spin-offs, XOM missing the 2026-06 period, etc.); interpret revenue_growth results with this in mind.
    revenue_file: str = "revenue_adjusted.csv"
    # Closing levels of several indices: ^GSPC, ^IXIC, ^NDX, ^DJI, ^RUT, ^VIX (VIX is a volatility level, not a price index),
    # plus the risk-free rate ^IRX (13-week T-bill yield, in annualized percent points, also not a price index).
    benchmark_file: str = "benchmark.csv"

    # Benchmark index, ∈ index_name in benchmark_file; defaults to the S&P 500.
    benchmark: str = "^GSPC"


@dataclass(frozen=True, slots=True)
class EvalSpec:
    """B Evaluation setup (``metrics`` and the baseline portfolios of ``backtest``): all runs must use the same setup to be comparable."""

    # Risk-free rate: column name in benchmarks, values in annualized percent (4.25 means 4.25%).
    # Daily risk-free rate rf_d = effective value / 100 / A; on missing days use the rate of the most recent earlier date with a value.
    risk_free: str = "^IRX"
    # Index baselines: price indices in benchmarks (cannot be ^VIX or ^IRX); must include DataSpec.benchmark (used for beta).
    index_baselines: tuple[str, ...] = ("^GSPC",)
    # Reference stocks: must be in the candidate pool; for reference only, not used in tuning or version selection.
    reference_tickers: tuple[str, ...] = ("NVDA",)
    # mom baseline: the mom_k stocks with the highest momentum(lookback=mom_lookback, skip=mom_skip), equally weighted;
    # does not follow cfg.factors.momentum, so strategy tuning does not change the baseline. mom_k ≥ 1, mom_lookback > mom_skip ≥ 0.
    mom_k: int = 3
    mom_lookback: int = 252
    mom_skip: int = 21
    # Random portfolios: each period, draw k stocks without replacement from those with a price that day (k = number of stocks the strategy selected), equally weighted, n_random portfolios in total, ≥ 1.
    n_random: int = 1000
    # Random seed (integer): each as_of draws separately with default_rng([random_seed, YYYYMMDD]).
    random_seed: int = 2026
    # Summary quantiles, 0 < first < second < 1; key names are generated from the quantiles (0.1 → p10).
    quantiles: tuple[float, float] = (0.1, 0.9)


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """B Task constraints / evaluation setup: determined by the assignment brief and shared by all runs so results are comparable."""

    # Decision date of the final portfolio, YYYY-MM-DD. Decide after that day's close and enter at the close price.
    # Stored as a string for lossless round-tripping with json; used once as as_of only in run.ipynb, functions in pf/ do not read it.
    decision_date: str = "2026-10-12"

    # Holding period H (trading days), ≥ 1. The evaluation period 2026-10-13 → 2026-12-07 has 39 trading days
    # (market closed 11-26 for Thanksgiving); each backtest period is held just as long, consistent with the evaluation period.
    # Also the horizon for prediction and risk: knn predicts H-day future returns (the horizon of ret),
    # and robust_mvo scales the daily covariance to H days (horizon). Both reuse this field; there is no separate parameter.
    holding_days: int = 39

    # Number of periods per year A, > 0: multiply daily quantities by √A (volatility) or A (returns) to annualize. Shared by factors and metrics.
    periods_per_year: int = 252

    # Evaluation setup: risk-free rate, baseline portfolios, random portfolios and summary quantiles.
    eval: EvalSpec = field(default_factory=EvalSpec)


@dataclass(slots=True)
class RunSpec:
    """C This run: backtest range and run record info. No field has a default; every run must fill them in (step is usually holding_days); an empty label raises ValueError."""

    # —— Rolling rule (backtest.as_of_dates) ——
    # Backtest start, YYYY-MM-DD: the first as_of is the first trading day ≥ start.
    # Leave room for the factor and covariance windows (about 253 trading days) and revenue yoy history, otherwise no stocks can be selected early on and robust_mvo raises.
    start: str
    # Backtest end, YYYY-MM-DD: the last day of every holding period is ≤ end; only as_of dates followed by H more trading days are kept.
    end: str
    # Number of trading days between adjacent as_of dates, ≥ 1. When = H (holding_days), periods are joined end to end into one portfolio daily return series;
    # when < H, periods overlap and there are only per-period returns, no portfolio daily return series.
    step: int

    # —— Run record (metrics.save_run) ——
    # Label (required): inserted into the record file name <YYYYMMDD_HHMMSS>_<label>_<4-char random string>.json; briefly describes what this run tried (e.g. "mom_only");
    # illegal file name characters : * ? " < > | / \ and whitespace are replaced by "_", then leading/trailing dots and underscores are stripped; the result must not be empty.
    label: str
    # Note: stored only in the record, not in the file name; may be an empty string.
    note: str

    def __post_init__(self) -> None:
        # Check at construction time so we don't run a whole backtest only to fail in save_run.
        if not LABEL_ILLEGAL.sub("_", self.label).strip("._"):
            raise ValueError(f"RunSpec.label must not be empty, nor consist only of illegal file name characters, whitespace, dots and underscores: {self.label!r}")


@dataclass(slots=True)
class MomentumParams:
    """E1 momentum factor parameters: P[t−skip] / P[t−lookback] − 1. Constraint lookback > skip ≥ 0."""

    # Lookback in trading days; about one year.
    lookback: int = 252
    # Number of most recent trading days to skip, ≥ 0 (0 means no skip); about one month.
    skip: int = 21


@dataclass(slots=True)
class DownsideVolParams:
    """E1 downside_vol factor parameters: −√(A · mean(min(r − tau, 0)²)), where A is the B-class periods_per_year."""

    # Number of daily returns, ≥ 1; about one year.
    window: int = 252
    # Daily return threshold (decimal); days above the threshold count as 0.
    tau: float = 0.0


@dataclass(slots=True)
class RevenueGrowthParams:
    """E1 revenue_growth factor parameters: revenue growth of the latest effective fiscal quarter versus the base period."""

    # ∈ {"yoy", "qoq"}: the base period is the fiscal quarter 12 months / 3 months earlier (aligned by the month of period_end).
    method: str = "yoy"


@dataclass(slots=True)
class KNNParams:
    """E1 two-stage KNN parameters; the prediction horizon reuses the B-class holding_days.

    Defaults are cfg_008 selected on 2026-10-02: 31 stocks, Development 2015–2018,
    Validation 2019–2022; short windows, 15 neighbors, market / stock distance weights 1:2.
    """

    # Size of the stage-one market candidate pool and the number of final stage-two neighbors.
    market: str = "^GSPC"
    market_pool_size: int = 40
    n_neighbors: int = 15
    # Below this count it still predicts but emits a summary UserWarning; this parameter does not change neighbors or predictions.
    min_neighbors: int = 8
    # Minimum number of trading days between neighbors; None means reuse holding_days.
    neighbor_gap: int | None = None

    # Stage-one market state windows (trading days).
    market_momentum_window: int = 21
    market_vol_window: int = 21
    market_drawdown_window: int = 63
    breadth_window: int = 21
    dispersion_window: int = 21

    # Stage-two stock state windows (trading days).
    stock_momentum_window: int = 21
    stock_vol_window: int = 21
    stock_drawdown_window: int = 63

    # Distance weights of the market block and the stock block; as long as the ratio is the same, the neighbor ranking is the same.
    market_distance_weight: float = 1.0
    stock_distance_weight: float = 2.0


@dataclass(slots=True)
class RetParams(KNNParams):
    """E1 ret factor parameters: inherits all two-stage KNN parameters.

    ``horizon`` reuses the B-class ``holding_days``; it is merged in by the caller and not registered again here.
    """


@dataclass(slots=True)
class FactorParams:
    """E1 per-factor parameters: field name = registered name in ``factors.FACTORS``, value = that factor's parameters."""

    momentum: MomentumParams = field(default_factory=MomentumParams)
    downside_vol: DownsideVolParams = field(default_factory=DownsideVolParams)
    revenue_growth: RevenueGrowthParams = field(default_factory=RevenueGrowthParams)
    ret: RetParams = field(default_factory=RetParams)


@dataclass(slots=True)
class StandardizeParams:
    """E2 standardization parameters: all factors share MAD winsorization + cross-sectional z-score (``scoring.standardize``)."""

    # Clipping multiple n, > 0: clip to median ± n·MAD, MAD = median(|x − median|), not multiplied by 1.4826.
    # n = 5 is roughly ±3.4σ under normality.
    n: float = 5.0


@dataclass(slots=True)
class ClusterParams:
    """E3 clustering parameters (``cluster.hierarchical``): agglomerative hierarchical clustering, distance d = √(2(1 − ρ)), outputs the cluster column of the score table.

    Selection (``SelectionParams``) uses this label to cap the number of stocks per cluster.
    """

    # Number of daily returns for the correlation, ≥ 3; about one year. The estimation error of the correlation is about (1 − ρ²)/√T,
    # about 0.1 at T = 60, making clusters very unstable. 120–250 is recommended; do not go below 120.
    window: int = 250

    # True: first run an OLS with intercept on the market index mkt and cluster on residuals (removing common market moves); False: use daily returns directly.
    remove_market: bool = True
    # Market proxy, a price index ∈ benchmarks (^GSPC, ^IXIC, ^NDX, ^DJI, ^RUT; ^VIX cannot be used).
    mkt: str = "^GSPC"

    # Linkage rule (distance between two clusters):
    #   "single"    The closest pair; suffers from chaining (A like B, B like C, so A and C get strung together); not recommended.
    #   "complete"  The farthest pair; compact clusters, and when cut at min_corr any two stocks in a cluster have ρ ≥ min_corr.
    #   "average"   Average over all cross-cluster pairs; cutting at min_corr only guarantees it on average.
    #   "ward"      Smallest increase in within-cluster variance; more even cluster sizes, the threshold has no direct meaning.
    method: str = "complete"

    # Tree cut: exactly one of min_corr and n_clusters is non-None (set the other to None), otherwise an error is raised.
    # min_corr ∈ [−1, 1]: cut at height h = √(2(1 − min_corr)); under complete this means "same risk only if residual correlation ≥ min_corr".
    min_corr: float | None = 0.1
    # Number of clusters, ≥ 1: cut directly into at most this many clusters; independent of the pool's structure, it may split essentially identical stocks.
    n_clusters: int | None = None


@dataclass(slots=True)
class ClusterCapParams:
    """E3 cluster_cap selection parameters: greedy selection by total score descending, capped per cluster, stopping at a score gap once min_n stocks are selected."""

    # Maximum number of stocks per cluster, ≥ 1. Works together with the cfg.cluster cut threshold: loosen it when the threshold is loose and clusters are large; 1 is enough when the threshold is tight and clusters are small.
    max_per_cluster: int = 2
    # Minimum number of stocks, 1 ≤ min_n ≤ max_n: score gaps are not checked until min_n stocks are selected (the cluster cap always applies).
    min_n: int = 4
    # Maximum number of stocks, ≥ 1.
    max_n: int = 6
    # Maximum allowed score gap (in total-score units, dimensionless), ≥ 0: stop selecting once a candidate scores more than this below the previous selected stock.
    # The total score is the weighted sum of the factor z-scores.
    max_gap: float = 100.0


@dataclass(slots=True)
class SelectionParams:
    """E3 selection: registered name of the rule and the parameters of each rule; field names (except method) = registered names in ``selection.SELECTORS``."""

    # ∈ selection.SELECTORS; currently only "cluster_cap".
    method: str = "cluster_cap"
    cluster_cap: ClusterCapParams = field(default_factory=ClusterCapParams)


@dataclass(slots=True)
class RobustMVOParams:
    """E4 robust_mvo parameters: max μᵀw − κ·√(wᵀΩw), with stock and free cash weights summing to 1.

    μ is knn's H-day predicted return; Σ = H × Ledoit–Wolf(daily returns); Ω = diag(u²),
    where u is the standard error of the predicted median from the same knn run, on the same H-day return basis as μ.
    CASH has return, prediction uncertainty and volatility fixed at 0; any shortfall of stock weights below 1 automatically becomes cash.
    H (horizon) reuses the B-class holding_days; it is merged in by the caller and not registered here.
    Fixed choices with no parameters: covariance is always Ledoit–Wolf; the uncertainty set always uses KNN's diagonal prediction error variance.
    """

    # —— Estimation ——
    # Number of daily returns for estimating Σ, ≥ 2; about one year. Shorter tracks current conditions more closely but is noisier;
    # LW shrinkage strengthens automatically as the sample shrinks. 126–504 is recommended.
    lookback: int = 252

    # —— Preferences ——
    # Uncertainty set radius κ, ≥ 0: penalizes unreliable predictions by the standard error of μ from KNN; 0 is plain mean-variance.
    kappa: float = 0.5
    # Maximum portfolio volatility allowed over the H-day holding period (decimal), > 0; the constraint is wᵀΣw ≤ max_volatility².
    # E.g. 0.10 means the portfolio's volatility over the holding period must not exceed 10%. This is a risk budget, not part of the objective.
    max_volatility: float = 0.10

    # —— Constraints ——
    # Lower and upper bounds on single-stock weights (decimal). Must satisfy 0 ≤ lower ≤ upper and lower × number of stocks ≤ 1, otherwise an error is raised.
    # Always long-only; the cash weight can vary freely in [0, 1] and is not bound by the stock upper limit.
    lower: float = 0.0
    upper: float = 0.3


@dataclass(slots=True)
class WeightsParams:
    """E4 weights: registered name of the method and the parameters of each method; field names (except method) = registered names in ``weights.WEIGHTERS``."""

    # ∈ weights.WEIGHTERS; currently only "robust_mvo".
    method: str = "robust_mvo"
    robust_mvo: RobustMVOParams = field(default_factory=RobustMVOParams)


@dataclass(frozen=True, slots=True)
class Config:
    """All parameters of a strategy (A, B, D, E); sections cannot be replaced wholesale, only fields within the D and E sections can be changed."""

    data: DataSpec = field(default_factory=DataSpec)
    task: TaskSpec = field(default_factory=TaskSpec)

    # D factor weights: factor name → weight σ_k, total score S = Σ σ_k · z_k. The keys determine which factors are enabled and must be ∈ factors.FACTORS.
    # To turn off a factor, delete its key; do not set its weight to 0 (0 × NaN = NaN, so a factor with missing values would still make the total score NaN).
    factor_weights: dict[str, float] = field(default_factory=lambda: {
        "momentum": 1.0,        # σ1
        "downside_vol": 1.0,    # σ2
        "revenue_growth": 1.0,  # σ3
        "ret": 1.0,             # σ4
    })

    # E1 factor parameters; cfg.factors.<factor name>.<parameter> can be changed, cfg.factors itself cannot be replaced.
    factors: FactorParams = field(default_factory=FactorParams)

    # E2 standardization parameters; cfg.standardize.n can be changed.
    standardize: StandardizeParams = field(default_factory=StandardizeParams)

    # E3 clustering; cfg.cluster.<parameter> can be changed. Adds the cluster column to the score table so selection can cap stocks per cluster.
    cluster: ClusterParams = field(default_factory=ClusterParams)

    # E3 selection; cfg.selection.<rule name>.<parameter> can be changed, cfg.selection itself cannot be replaced.
    selection: SelectionParams = field(default_factory=SelectionParams)

    # E4 weights; cfg.weights.<method name>.<parameter> can be changed, cfg.weights itself cannot be replaced.
    weights: WeightsParams = field(default_factory=WeightsParams)

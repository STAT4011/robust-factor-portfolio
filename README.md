# Multi-factor stock selection and portfolio backtesting

Course project: select stocks by combining momentum, downside volatility, revenue growth and KNN return prediction, control concentration with clustering, then allocate weights with robust mean–variance optimization.

## Project structure

```text
data_wsy/       price, revenue and index data, plus the fetch scripts
pf/             factor computation, stock selection, portfolio optimization and backtesting
notebooks/      interactive runs, validation and results display
outputs/runs/   saved parameters, backtest metrics and per-period holdings (JSON)
pyproject.toml  Python version and dependency configuration
uv.lock         dependency lock file
```

Main flow of `pf/`: `loader` reads data → `factors` (calling `knn`) computes factors → `scoring` standardizes and scores → `cluster` clusters → `selection` selects stocks → `weights` allocates weights.
`backtest` handles single-period decisions and rolling backtests, `baselines` builds the baseline portfolios, and `metrics` computes metrics and saves/loads run records; parameters are centralized in `config.py`.

## Quick start

Requires Python 3.12 and uv. From the project root, run:

```bash
uv sync
uv run jupyter lab
```

1. Open `notebooks/run.ipynb` and use the project's `.venv` kernel.
2. Set the backtest range, step, label and note in section C, and adjust strategy parameters in sections D and E.
3. Run all cells in order to see the backtest results and the final portfolio. Run records are saved to `outputs/runs/`, and the report cell re-executes and saves `report.ipynb`.

By default it reads `prices_raw.csv`, `revenue_adjusted.csv` and `benchmark.csv` from `data_wsy/`, all of which are included in the project.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| [run.ipynb](notebooks/run.ipynb) | Main entry point: set parameters, run rolling backtests, save results, compare runs and generate the final portfolio. |
| [explore.ipynb](notebooks/explore.ipynb) | Step-by-step exploration of a single day: inspect factors, the clustering dendrogram, automatic or manual stock selection, and portfolio weights. |
| [knn_test.ipynb](notebooks/knn_test.ipynb) | Standalone evaluation and tuning of the two-stage KNN, comparing predictive power, coverage and neighbor stability. |
| [debug.ipynb](notebooks/debug.ipynb) | Formula and numerical validation: hand-computed comparisons, causality, missing values and edge-case checks. |
| [report.ipynb](notebooks/report.ipynb) | Reads a saved run record and plots returns, risk, benchmark comparisons and holdings charts; change `RUN_FILE` to switch records. |

## Frozen parameters

The final strategy parameters are frozen from the validation run [`20261013_171155_val-mad_1cb7.json`](outputs/runs/20261013_171155_val-mad_1cb7.json) (label `val-mad`, git commit `a167275`; backtest 2023-01-01 → 2024-12-31, step 39, 12 periods).
Values marked **bold** differ from the defaults in `pf/config.py`; all others are the defaults.

Data and task (A, B): `prices_raw.csv` (`close`), `revenue_adjusted.csv`, `benchmark.csv`, benchmark `^GSPC`; decision date 2026-10-12, holding period 39 trading days, 252 periods per year.

| Section | Parameter | Value |
| --- | --- | --- |
| D factor weights | `momentum` / `downside_vol` / `revenue_growth` / `ret` | **0.4 / 0.3 / 0.4 / 0.3** |
| E1 momentum | `lookback` / `skip` | **189** / 21 |
| E1 downside_vol | `window` / `tau` | 252 / 0.0 |
| E1 revenue_growth | `method` | `yoy` |
| E1 ret (KNN) | `market` / `market_pool_size` / `n_neighbors` / `min_neighbors` / `neighbor_gap` | `^GSPC` / 40 / 15 / 8 / `None` (= holding period) |
| E1 ret (KNN) | market windows: momentum / vol / drawdown / breadth / dispersion | 21 / 21 / 63 / 21 / 21 |
| E1 ret (KNN) | stock windows: momentum / vol / drawdown | 21 / 21 / 63 |
| E1 ret (KNN) | `market_distance_weight` / `stock_distance_weight` | 1.0 / 2.0 |
| E2 standardize | `n` (MAD clip multiple) | 5.0 |
| E3 cluster | `window` / `remove_market` / `mkt` | 250 / `True` / `^GSPC` |
| E3 cluster | `method` / `min_corr` / `n_clusters` | `complete` / 0.1 / `None` |
| E3 selection | `method` | `cluster_cap` |
| E3 cluster_cap | `max_per_cluster` / `min_n` / `max_n` / `max_gap` | 2 / 4 / 6 / 100.0 |
| E4 weights | `method` | `robust_mvo` |
| E4 robust_mvo | `lookback` / `kappa` / `max_volatility` | **504** / **1** / **0.15** |
| E4 robust_mvo | `lower` / `upper` | 0.0 / 0.3 |

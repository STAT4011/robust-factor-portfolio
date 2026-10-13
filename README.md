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

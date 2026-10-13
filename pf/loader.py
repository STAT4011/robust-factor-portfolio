"""loader — read ``data_wsy/`` and convert to the internal standard format; the only module in the project that reads ``data_wsy/``.

Outputs (all full sample, not truncated)
----
- Prices (``load_prices``): DataFrame[date × ticker], the column used is given by the ``price_col`` parameter;
  late-listed stocks (e.g. ``CRWV``) are NaN before listing.
- Fundamentals (``load_fundamentals``): long-format fundamentals table, currently revenue only.
- Benchmark indices (``load_benchmarks``): DataFrame[date × benchmark], closing levels.

Rows with empty values in prices and indices are treated as "no record" (when the data source is scraped intraday on a trading day, that day leaves empty rows).

The revenue data still has known issues (GE reporting break around its spin-offs, XOM missing the 2026-06 period, etc.).
``load_fundamentals`` does not use the data source's fiscal quarter labels; it identifies fiscal quarters by ``period_end``.

Only reading, cleaning and format conversion; no truncation and no factor computation.
"""

import pandas as pd

from pf.config import DATA_DIR

# Price files allowed to be read; see the comments on config.DataSpec.price_file for each file's range.
PRICE_FILES: tuple[str, ...] = ("prices_raw.csv",)
# Columns allowed as prices; see the comments on config.DataSpec.price_col for their meaning.
PRICE_COLS: tuple[str, ...] = ("close", "adj_close")


def load_prices(price_file: str, price_col: str) -> pd.DataFrame:
    """Read the long-format price table in ``data_wsy/`` and convert it to wide format.

    Parameters
    ----------
    price_file : str
        File name under ``data_wsy/``, ∈ ``PRICE_FILES``.
    price_col : str
        Column used as price, ∈ ``PRICE_COLS``.

    Returns
    -------
    DataFrame[date × ticker]
        Daily prices (USD), full sample, not truncated. index is trading days (ascending, unique, named ``date``),
        columns are stock symbols (alphabetical, named ``ticker``). NaN when a stock has no record or an empty value on a trading day,
        e.g. ``CRWV`` before its listing on 2025-03-28; dates on which all stocks are empty do not appear in the index.

    Raises
    ------
    ValueError
        ``price_file`` is not in ``PRICE_FILES``, ``price_col`` is not in ``PRICE_COLS``,
        or the same (date, ticker) appears in multiple rows.
    """
    if price_file not in PRICE_FILES:
        raise ValueError(f"price_file must be one of {PRICE_FILES}, got {price_file!r}")
    if price_col not in PRICE_COLS:
        raise ValueError(f"price_col must be one of {PRICE_COLS}, got {price_col!r}")

    long = pd.read_csv(
        DATA_DIR / price_file,
        encoding="utf-8-sig",
        usecols=["date", "ticker", price_col],
        dtype={"ticker": "str", price_col: "float64"},
    )
    long["date"] = pd.to_datetime(long["date"], format="%Y-%m-%d")
    # A day scraped intraday has only volume and no close; drop empty rows first, otherwise the index ends with an all-NaN "trading day".
    long = long.dropna(subset=[price_col])
    # pivot raises ValueError directly on duplicate (date, ticker), instead of silently picking one.
    return long.pivot(index="date", columns="ticker", values=price_col).sort_index()


def load_fundamentals(revenue_file: str) -> pd.DataFrame:
    """Read the quarterly revenue in ``data_wsy/`` and convert it to a long-format fundamentals table.

    Parameters
    ----------
    revenue_file : str
        File name under ``data_wsy/``; columns must include ``ticker, period_end, report_date, revenue``.

    Returns
    -------
    long-format fundamentals table
        Columns ``ticker, ann_date, period, value``; each row is a stock's revenue for a fiscal quarter (USD, single quarter).
        ``ann_date`` is ``report_date``; ``period`` is the month of the month-end nearest to ``period_end``,
        formatted ``"YYYY-MM"`` (e.g. JNJ 2021-01-03 → ``"2020-12"``). ``value`` is NaN when revenue is missing,
        ``ann_date`` is NaT when the announcement date is missing; all these rows are kept. Sorted by ``ticker, ann_date``.

    Raises
    ------
    ValueError
        The same (ticker, period) appears in multiple rows.
    """
    raw = pd.read_csv(
        DATA_DIR / revenue_file,
        encoding="utf-8-sig",
        usecols=["ticker", "period_end", "report_date", "revenue"],
        dtype={"ticker": "str", "revenue": "float64"},
    )
    # Do not use the data source's fiscal_year / fiscal_quarter: they are inferred by the data script from ordering,
    # and repeat or skip at year boundaries of 52/53-week fiscal years. The period_end of 52/53-week fiscal years drifts a few days around month-end;
    # after taking the nearest month-end, the same quarter of each year falls in the same month, so year-over-year aligns by a 12-month difference.
    period_end = pd.to_datetime(raw["period_end"], format="%Y-%m-%d")
    period = ((period_end + pd.Timedelta(days=15)).dt.to_period("M") - 1).astype("str")
    out = pd.DataFrame({
        "ticker": raw["ticker"],
        "ann_date": pd.to_datetime(raw["report_date"], format="%Y-%m-%d"),
        "period": period,
        "value": raw["revenue"],
    })
    dup = out.duplicated(subset=["ticker", "period"], keep=False)
    if dup.any():
        raise ValueError(f"duplicate (ticker, period):\n{out.loc[dup, ['ticker', 'period']]}")
    return out.sort_values(["ticker", "ann_date"], ignore_index=True)


def load_benchmarks(benchmark_file: str) -> pd.DataFrame:
    """Read the long-format index closing level table in ``data_wsy/`` and convert it to wide format.

    Parameters
    ----------
    benchmark_file : str
        File name under ``data_wsy/``; columns must include ``date, index_name, close``.

    Returns
    -------
    DataFrame[date × benchmark]
        Daily closing levels (``^VIX`` is a volatility level, not a price index). index is dates (ascending, unique, named ``date``),
        columns are index names (alphabetical, named ``benchmark``). NaN when an index has no record or an empty value on a date;
        dates on which all indices are empty do not appear in the index. The index is not guaranteed to equal the price index
        (``^VIX`` in the data source also has values on a few market holidays). Which index to use is given by ``DataSpec.benchmark``; the caller takes the corresponding column.

    Raises
    ------
    ValueError
        The same (date, index_name) appears in multiple rows.
    """
    long = pd.read_csv(
        DATA_DIR / benchmark_file,
        encoding="utf-8-sig",
        usecols=["date", "index_name", "close"],
        dtype={"index_name": "str", "close": "float64"},
    )
    long["date"] = pd.to_datetime(long["date"], format="%Y-%m-%d")
    long = long.dropna(subset=["close"])
    wide = long.pivot(index="date", columns="index_name", values="close").sort_index()
    return wide.rename_axis(columns="benchmark")


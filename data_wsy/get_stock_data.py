#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAT4011 Part 1 — Data acquisition script
=========================================

Fetches three kinds of data in one go and saves them as CSV:

  1. prices_raw.csv   Historical prices (Yahoo Finance / yfinance) — 31 stocks
                      Columns: date, ticker, close, adj_close, volume, dividend
  2. revenue.csv      Quarterly revenue (BusinessQuant) + filing date (SEC EDGAR)
                      Columns: ticker, period_end, report_date, revenue, revenue_ttm
                      ⚠️ This table does **not include** fiscal_year / fiscal_quarter, and holds raw BQ values
                         **with known errors**. Corrections and fiscal quarter labels are done by merge_revenue.py;
                         use revenue_adjusted.csv for backtests.
  3. benchmark.csv    Market indices (Yahoo Finance / yfinance)
                      Columns: date, index_name, close

  ⚠️ This script only "fetches the data". Revenue corrections (single line item / Q4 check / fiscal quarter labels) live in
     merge_revenue.py, which is the single entry point for the revenue part.

Design notes
------------
* **Revenue from BusinessQuant + SEC**:
  - **Single-quarter revenue** comes from BusinessQuant (free 40 requests/day, batches of 5 → 7 requests for all 31 stocks).
    It already converts SEC "cumulative figures" into single quarters, a step that is very error-prone to do yourself
    (fiscal years not matching calendar years, tag changes midway, 52/53-week fiscal years, multiple records for the same period end...).
    Its Q4 can still be wrong, though, so merge_revenue.py re-checks it against SEC.
  - **Filing dates** come from the SEC EDGAR submissions endpoint (free, unlimited),
    matched to BQ report periods by `date == reportDate` (verified: the vast majority match exactly;
    52/53-week fiscal years may differ by a day or two, so downstream pairs them by the snapped month).
* **Panel alignment is not done here**: this script only "obtains the raw data";
  alignment (pivot + dropna) is left to the downstream analysis scripts.
* **No time trimming**: the revenue table outputs the longest history BQ provides (from about 2009);
  the backtest window is chosen by the analysis scripts. Start years differ across companies (due to restructurings / late listings).

Usage
-----
    python get_stock_data.py                       # fetch everything
    python get_stock_data.py --only prices         # prices only
    python get_stock_data.py --only revenue        # revenue only
    python get_stock_data.py --only benchmark      # indices only
    python get_stock_data.py --tickers AAPL,MSFT   # only the given stocks
    python get_stock_data.py --start 2015-01-01 --end 2026-09-30
    python get_stock_data.py --revenue-source legacy   # fall back to the old SEC reconstruction logic

Dependencies: yfinance, pandas (both pull in the other packages needed)
      HTTP requests use the standard library urllib, so requests need not be installed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime
from typing import Any, Iterable

import pandas as pd

# =============================================================================
# Configuration — everything you may need to change is here
# =============================================================================

# ---- Known limitations of the revenue data (must read) ----------------------
# revenue.csv now takes single-quarter revenue from BusinessQuant; data quality has been checked against actual filings.
# Still note:
#   * **XOM has no revenue data**: after the 2026 holding-company restructuring, BQ points XOM to a new entity
#     with only recent data. Price data is unaffected.
#   * **8 companies have less than 15 years of history** (due to restructurings or late listings; changing data source would not help):
#       GOOGL from 2014-09 (Alphabet restructuring in 2015)
#       AVGO  from 2017-04 (Avago acquired Broadcom in 2016)
#       LIN   from 2017-06 (merger in 2018)
#       ABBV  from 2012-03 (spun off from Abbott in 2013)
#       AAOI  from 2013-03, ONDS from 2016-06, TEM from 2023-06, CRWV from 2024-03
#   * **About 2.5% of report_date values are missing**: BQ's data comes from a predecessor entity (e.g. Google Inc.
#     before the Alphabet restructuring) and does not match filings under the current CIK. Mostly early periods.
#   * **Q4 may be too small for some companies**: BQ derives Q4 as "full year minus 9 months"; when the 10-K and
#     10-Q definitions differ this leaves a residual (seen: BRK-B 2024Q4 = 5.72B, normally about 90B).
#     If an abnormally small value appears at the end of a reporting period, check it manually.

# ---- Time range (fixed dates for reproducibility) ---------------------------
START = "2011-01-01"          # Start of prices and indices. The revenue table is not trimmed; output up to the data source's limit
END = None                    # None = today

# ---- Candidate stocks (31) --------------------------------------------------
TICKERS: list[str] = [
    # Technology
    "GOOGL", "AVGO", "MSFT", "INTC", "NOW", "NVDA", "AAPL", "AMZN",
    # Energy
    "XOM", "MPC", "EOG",
    # Materials
    "NEM", "LIN",
    # Consumer
    "WMT", "MCD", "KO",
    # Healthcare
    "LLY", "JNJ", "ABBV",
    # Industrials & defense
    "GE", "CAT", "LMT",
    # Financials
    "JPM", "BRK-B", "V", "BAC",
    # Small & mid caps
    "CRWV", "AAOI", "TEM", "MSTR", "ONDS",
]

# ---- Market indices ---------------------------------------------------------
# In the form (Yahoo symbol, index_name written to the csv).
# The "^" before the symbol cannot be omitted.
#
# ⚠️ Note that ^VIX has different units from the other indices: it is quoted as a percentage (e.g. 18.5 means annualized volatility
#    of 18.5%), whereas ^GSPC is in points (e.g. 5800). Both are in the same table, so
#    after pivoting, **do not apply pct_change to the whole table**, or VIX will yield
#    "the rate of change of volatility", which is meaningless. Handle each index_name separately in analysis.
INDICES: list[tuple[str, str]] = [
    ("^GSPC", "^GSPC"),   # S&P 500
    ("^IXIC", "^IXIC"),   # Nasdaq Composite
    ("^NDX",  "^NDX"),    # Nasdaq 100
    ("^DJI",  "^DJI"),    # Dow Jones Industrial Average
    ("^RUT",  "^RUT"),    # Russell 2000 (small caps)
    ("^VIX",  "^VIX"),    # Volatility index (see the warning above)
    ("^IRX",  "^IRX"),    # 13-week T-bill yield (annualized percent, risk-free rate; also must not use pct_change)
]

# ---- Fiscal year end month --------------------------------------------------
# Used to assign report periods to fiscal years. **Do not rely on statistical inference** — three heuristics were tried and all failed:
#   * Counting up to 4 sequentially: one missing period shifts everything; misses boundaries of 52/53-week fiscal years
#   * Most frequent month: wrong for 24 of 27 companies in testing (end dates drift into adjacent months)
#   * Largest gap between adjacent periods: also unreliable in testing (JNJ's gap distribution is {2:4, 3:59, 4:4, 6:1};
#     the fiscal year boundary is not the largest gap, and there is also a 6-month missing period)
#
# So it is **given explicitly** instead. Source: SEC's fiscalYearEnd field (e.g. "12-31").
# Companies not listed default to December (calendar fiscal year), and the inferred result is printed for checking.
FY_END_MONTH: dict[str, int] = {
    "AAPL": 9,
    "MSFT": 6,
    "WMT": 1,
    "AVGO": 10,
    "V": 9,
}

# Fiscal year end month for the remaining companies (anything not listed is treated as December)
FY_END_MONTH_DEFAULT = 12

# ---- SEC EDGAR settings -----------------------------------------------------
# SEC requires contact information in the User-Agent, otherwise it returns 403.
SEC_CONTACT = "shuiyuntian6@gmail.com"
SEC_UA = f"STAT4011-Part1/1.0 ({SEC_CONTACT})"
SEC_SLEEP = 0.15              # Interval between requests (seconds). SEC rate limit is about 10 requests/second

# ---- BusinessQuant settings (revenue data source) ---------------------------
# Free quota is **40 requests/day**, so requests must be batched (at most 5 tickers each).
# Single-quarter revenue + TTM revenue are fetched in the same request, using no extra quota.
#
# Why use BusinessQuant instead of reconstructing single quarters from SEC XBRL ourselves:
#   SEC only gives cumulative figures (Q1, H1, 9M, full year); single quarters come from subtraction, with many edge cases
#   (fiscal years not matching calendar years, companies changing tags midway, 52/53-week fiscal years, multiple records for the same period end...).
#   BusinessQuant already does this step, and its values have been checked item by item against actual filings.
BQ_API_KEY = os.environ.get("BQ_API_KEY", "7505d20215ff5975bf07c93f5e46eacd")
BQ_BASE = "https://data.businessquant.com"
BQ_BATCH = 5                  # ticker limit; exceeding it returns 400
BQ_SLUGS = ["revenue", "revenue-ttm"]   # single quarter + trailing four quarters

# Symbols not covered by BUSINESSQUANT (revenue dimension missing, prices still fine)
# XOM: after the 2026 holding-company restructuring, BQ points XOM to a new entity with only recent data;
#      the old entity has a full filing history at SEC but no usable revenue series.
BQ_KNOWN_MISSING = {"XOM"}

# ---- [Deprecated] SEC XBRL revenue tag priority -----------------------------
# The logic below (_collect_revenue_facts / build_quarterly_revenue) has been
# superseded by the BusinessQuant approach. Kept so we can fall back if BQ is unavailable.
# Known defect: reconstructing single quarters from cumulative figures does not converge on edge cases, causing some report periods
# to hold full-year cumulative figures (205 report periods were wrong in testing). **Do not use for production data.**
REVENUE_TAGS: list[str] = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
    "RevenuesNetOfInterestExpense",            # common for banks
    "InterestAndDividendIncomeOperating",
    "RegulatedAndUnregulatedOperatingRevenue",
    # Industry-specific tags below. Placed later: they only take effect when the generic tags have no data.
    # E.g. Newmont (NEM) reported revenue as RevenueMineralSales before 2017;
    #     without it, its 2011–2017 revenue would be lost entirely.
    "RevenueMineralSales",                     # mining
    "RevenueFromRelatedParties",
    "ElectricUtilityRevenue",                  # utilities
    "OilAndGasRevenue",
    "HealthCareOrganizationRevenue",
    "RealEstateRevenueNet",
]

# ---- Auto-discover additional revenue tags ----------------------------------
# Besides the fixed list above, all us-gaap tags whose names match the patterns below are also candidates
# (see _collect_revenue_facts). Purpose: handle industry/company-specific revenue tags,
# so that "only recognizing the list" does not lose whole stretches of history.
# These auto-included tags rank **after** the fixed list (lowest priority).
REVENUE_TAG_AUTODISCOVER = True

# ---- ticker → CIK manual overrides ------------------------------------------
# Usually not needed: the script uses SEC's official ticker mapping.
# But when a company undergoes a **holding-company restructuring**, SEC points the ticker to the newly created entity,
# which only has filings after the restructuring, while the historical data stays under the old entity.
# E.g. ExxonMobil (XOM) restructured in 2026; SEC points XOM to the new entity CIK 0002115436
#     (data only from 2025 on), while the 15-year history is under the old entity CIK 0000034088.
# This specifies "which CIK to prefer"; the script automatically determines whose data is more complete.
CIK_OVERRIDES: dict[str, list[str]] = {
    "XOM": ["0000034088", "0002115436"],   # old entity first, new entity as fallback
}

# ---- EDGAR data start year (mandatory XBRL filing began in 2009) ------------
EDGAR_MIN_YEAR = 2009

# ---- yfinance repair option -------------------------------------------------
# repair=True calls scikit-learn to fix Yahoo's split/price errors for cleaner data,
# but requires scikit-learn to be installed. If it is missing, the script automatically downgrades to repair=False
# and notes it in the log (data is still usable, just without automatic repair).
USE_REPAIR = True

# ---- Default output directory = this script's directory ---------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Logging: output to both the console and a file
# =============================================================================

logger = logging.getLogger("stat4011")


class _Tee:
    """Write whatever goes to stdout to a file as well, so fetch_log.txt is kept."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    log_path = os.path.join(outdir, "fetch_log.txt")
    fh = open(log_path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
    return log_path


def log(msg: str = "") -> None:
    logger.info(msg)


# =============================================================================
# General utilities
# =============================================================================

def parse_date(s: str | None) -> date | None:
    return datetime.strptime(s, "%Y-%m-%d").date() if s else None


def norm_ticker(t: str) -> str:
    """Normalize a ticker for comparison against SEC's mapping table.

    BRK-B / BRK.B / BRK/B must all match the same record.
    """
    return t.upper().replace(".", "").replace("-", "").replace("/", "").strip()


def sec_get(url: str, retries: int = 3) -> Any | None:
    """SEC GET with retries; returns the parsed JSON, or None on failure.

    Implemented with the standard library urllib to avoid an extra HTTP dependency.
    """
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            time.sleep(SEC_SLEEP)
            return payload
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None                      # the company has no such resource
            if exc.code in (429, 503):
                time.sleep(2 * (attempt + 1))    # rate limited, back off and retry
                continue
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            if attempt == retries - 1:
                return None
            time.sleep(2)
    return None


def sec_get_text(url: str) -> str | None:
    """Download plain text/JSON and return it as is; return None on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            text = resp.read().decode("utf-8")
        time.sleep(SEC_SLEEP)
        return text
    except Exception:                                # noqa: BLE001
        return None


# =============================================================================
# Part 1: Prices (yfinance)
# =============================================================================

def _resolve_repair() -> bool:
    """Check whether scikit-learn is available, to decide yfinance's repair switch.

    yfinance's repair=True depends on scikit-learn. Without it **every** ticker
    raises ModuleNotFoundError, which looks like "everything failed" and is easily mistaken for wrong stock symbols.
    So probe once here; if missing, downgrade automatically and state the reason in the log.
    """
    if not USE_REPAIR:
        return False
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


def _drop_partial_tail(long: pd.DataFrame, tickers: list[str],
                       kind: str = "prices") -> pd.DataFrame:
    """Drop "incomplete" trading days at the end.

    Why this is needed: when yfinance runs **during market hours**, it mixes the day's **live quotes** into
    the historical series. The observed symptom is that on the last day only some symbols have values, and the prices are round numbers
    (e.g. MSFT=498.00, JPM=340.00), which are not real closes. Such incomplete rows make
    downstream logic that relies on "all symbols having data on a given day" (correlation matrix, clustering, cross-sectional factors)
    turn entirely NaN on that day, which in turn means no stocks can be selected on the decision date.

    Approach: looking back from the last day, as long as the number of symbols with data is less than "the expected number for that day",
    drop that whole row, until a complete trading day is reached.

    "The expected number for that day" is estimated by **the coverage of the second-to-last complete day**, so it is robust to normal gaps
    such as IPOs and delistings: it only drops the last few days where coverage "suddenly drops at the end".
    """
    if long.empty:
        return long

    key_col = "index_name" if kind == "benchmark" else "ticker"
    counts = long.groupby("date")[key_col].nunique().sort_index()
    if len(counts) < 2:
        return long

    expected = int(counts.iloc[-2])          # use the second-to-last day as the reference
    keep_until = counts.index[-1]
    dropped = []
    for dt in reversed(counts.index):
        if int(counts.loc[dt]) >= expected:
            keep_until = dt
            break
        dropped.append((dt, int(counts.loc[dt])))
        keep_until = dt

    if dropped:
        log(f"  dropping {len(dropped)} incomplete trading day(s) at the end"
            f" (expected {expected} entries that day, actually fewer):")
        for dt, n in reversed(dropped):
            log(f"    {dt.date()}  only {n} / {expected} entries")

    return long[long["date"] <= keep_until].reset_index(drop=True)


def fetch_prices(
    tickers: list[str], start: str, end: str | None
) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch daily prices; returns (long DataFrame, list of failures)."""
    import yfinance as yf

    repair = _resolve_repair()
    if USE_REPAIR and not repair:
        log("  ⚠ scikit-learn not detected; repair has been turned off automatically (prices will no longer be auto-repaired for splits/anomalies)")
        log("    To enable it: pip install scikit-learn, or uv add scikit-learn")

    frames: list[pd.DataFrame] = []
    failures: list[dict] = []

    for i, sym in enumerate(tickers, 1):
        try:
            df = yf.download(
                sym,
                start=start,
                end=end,
                interval="1d",
                auto_adjust=False,   # key: keep both Close and Adj Close
                actions=True,        # key: bring back a Dividends column
                repair=repair,       # see _resolve_repair() above
                progress=False,
                threads=False,
            )
        except Exception as exc:                      # noqa: BLE001
            failures.append({"kind": "prices", "ticker": sym, "reason": repr(exc)})
            log(f"[prices {i:>2}/{len(tickers)}] {sym:<7} error: {exc}")
            continue

        if df is None or df.empty:
            failures.append({"kind": "prices", "ticker": sym, "reason": "empty result"})
            log(f"[prices {i:>2}/{len(tickers)}] {sym:<7} failed: empty data returned")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if "Close" not in df.columns:
            failures.append({"kind": "prices", "ticker": sym, "reason": "no Close column"})
            log(f"[prices {i:>2}/{len(tickers)}] {sym:<7} failed: no Close column")
            continue

        idx = df.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)

        out = pd.DataFrame({
            "date":       pd.to_datetime(idx).normalize(),
            "ticker":     sym,
            "close":      df["Close"].to_numpy(),
            # A few symbols have no Adj Close (e.g. BRK-B, which pays no dividends); fall back to Close to keep the column structure consistent
            "adj_close":  (df["Adj Close"].to_numpy() if "Adj Close" in df.columns
                           else df["Close"].to_numpy()),
            "volume":     (df["Volume"].to_numpy() if "Volume" in df.columns else pd.NA),
            # dividends come from actions=True; the whole column is 0 for non-dividend payers
            "dividend":   (df["Dividends"].to_numpy() if "Dividends" in df.columns
                           else 0.0),
        })
        n_div = int((out["dividend"] > 0).sum())
        frames.append(out)
        log(f"[prices {i:>2}/{len(tickers)}] {sym:<7} OK {len(out):>5} rows "
            f"({out['date'].iloc[0].date()} → {out['date'].iloc[-1].date()}) "
            f"dividends {n_div} times")

    if not frames:
        return pd.DataFrame(
            columns=["date", "ticker", "close", "adj_close", "volume", "dividend"]
        ), failures

    long = pd.concat(frames, ignore_index=True)
    long = long.sort_values(["date", "ticker"]).reset_index(drop=True)
    # When run during market hours yfinance mixes in the day's live quotes; drop incomplete trading days at the end
    long = _drop_partial_tail(long, tickers, "prices")
    return long, failures


# =============================================================================
# Part 2: Revenue (SEC EDGAR XBRL)
# =============================================================================

def load_cik_map(cache_dir: str) -> dict[str, str]:
    """ticker -> CIK (10-digit zero-padded string).

    Uses SEC's official mapping table. Cached locally on the first run to avoid downloading every time.
    """
    cache = os.path.join(cache_dir, "_sec_ticker_map.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            raw = json.load(f)
    else:
        text = sec_get_text("https://www.sec.gov/files/company_tickers.json")
        if text is None:
            raise RuntimeError("unable to download SEC ticker→CIK mapping table")
        raw = json.loads(text)
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(raw, f)

    out: dict[str, str] = {}
    for item in raw.values():
        ticker = str(item.get("ticker", ""))
        cik = str(item.get("cik_str", "")).zfill(10)
        if ticker and cik:
            out[norm_ticker(ticker)] = cik
    return out


def _ordered_revenue_tags(us_gaap: dict) -> list[str]:
    """Return the priority order of revenue tags (fixed list first, auto-discovered after).

    Auto-discovery catches industry-specific revenue tags (mining, utilities, healthcare, etc.),
    so history is not lost because "it's not on the list". But auto-included tags have the lowest priority
    and only take effect when the company has no data under the generic tags.
    """
    ordered = [t for t in REVENUE_TAGS if t in us_gaap]
    if not REVENUE_TAG_AUTODISCOVER:
        return ordered
    extra = sorted(
        t for t in us_gaap
        if t not in ordered
        and ("revenue" in t.lower() or "sales" in t.lower())
        and not any(k in t.lower() for k in
                    ("deferred", "unrealized", "cost", "proceeds", "tax"))
    )
    return ordered + extra


def _tag_rank(tag: str) -> int:
    """Give tag priority according to the fixed list; tags outside the list (auto-discovered) go last, stably sorted by name."""
    try:
        return REVENUE_TAGS.index(tag)
    except ValueError:
        return len(REVENUE_TAGS)


def _collect_revenue_facts(facts: dict, start_year: int) -> list[dict]:
    """Extract all usable revenue period records from companyfacts.

    Returns [{start, end, val, filed, form, tag, tag_rank}, ...].

    Key design: **merge across tags instead of using just one tag.**

    Reason: many companies changed revenue tags midway (e.g. after the ASC 606 accounting standard change,
    from `SalesRevenueNet` to `RevenueFromContractWithCustomerExcludingAssessedTax`).
    Taking only "the first tag with data" would lose the other half of the history — showing up as some companies
    having only a handful of quarters.

    Priority rule: when a report period has multiple tags, **take the highest-priority one (smallest tag_rank)**;
    among equal priority, take the earliest filed (= the originally announced value).
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    if not us_gaap:
        return []

    # Tag priority: fixed list first, then auto-discovered tags after it, sorted by name.
    ordered = _ordered_revenue_tags(us_gaap)
    rank_of = {tag: i for i, tag in enumerate(ordered)}
    if not rank_of:
        return []

    records: dict[tuple[str | None, str], dict] = {}

    for tag, rank in rank_of.items():
        node = us_gaap.get(tag)
        if not node:
            continue
        for unit, items in node.get("units", {}).items():
            # USD amounts only; some tags also have units such as shares
            if unit != "USD":
                continue
            for it in items:
                end = it.get("end")
                if not end:
                    continue
                if int(end[:4]) < start_year:
                    continue
                key = (it.get("start"), end)
                prev = records.get(key)
                if prev is None:
                    records[key] = {
                        "start": it.get("start"),
                        "end": end,
                        "val": it.get("val"),
                        "filed": it.get("filed"),
                        "form": it.get("form"),
                        "tag": tag,
                        "tag_rank": rank,
                    }
                elif rank < prev["tag_rank"]:
                    # Higher-priority tag: replace directly
                    records[key] = {
                        "start": it.get("start"),
                        "end": end,
                        "val": it.get("val"),
                        "filed": it.get("filed"),
                        "form": it.get("form"),
                        "tag": tag,
                        "tag_rank": rank,
                    }
                elif rank == prev["tag_rank"]:
                    # Duplicate filing of the same tag: keep the earliest announced
                    if str(it.get("filed", "9999")) < str(prev.get("filed", "9999")):
                        records[key]["val"] = it.get("val")
                        records[key]["filed"] = it.get("filed")
                        records[key]["form"] = it.get("form")

    return list(records.values())


def _value_by_end(recs: Iterable[dict]) -> dict[str, dict]:
    """Group records by period end date, keeping for each end the one record "most like a single quarter".

    Purpose: tag statistics in the log (showing which revenue tags each company actually used).
    The actual single-quarter reconstruction is done in build_quarterly_revenue().

    Selection criterion: shortest period first (closer to a single quarter is better); for equal length take the earliest filed
    (point-in-time: what investors saw at the time was the originally announced value, not a restatement years later).
    """
    best: dict[str, dict] = {}
    for r in recs:
        if not r.get("start") or r.get("val") is None:
            continue
        try:
            d0 = datetime.strptime(r["start"], "%Y-%m-%d").date()
            d1 = datetime.strptime(r["end"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        days = (d1 - d0).days
        if days <= 0:
            continue
        r = dict(r, days=days)
        cur = best.get(r["end"])
        if cur is None:
            best[r["end"]] = r
            continue
        key_new = (r["days"], str(r.get("filed", "9999")), r["tag_rank"])
        key_cur = (cur["days"], str(cur.get("filed", "9999")), cur["tag_rank"])
        if key_new < key_cur:
            best[r["end"]] = r
    return best


def _candidate_tags(groups: dict[tuple[str, str], list[dict]]) -> list[str]:
    """Give candidate revenue tags in order of "number of covered periods, most to least".

    When a company changed tags midway (e.g. from SalesRevenueNet to
    RevenueFromContractWithCustomerExcludingAssessedTax after ASC 606), the two stretches of history are
    covered by different tags. So we cannot use only the one with the most periods; they must be concatenated in order.
    """
    tally: dict[str, set] = {}
    for (start, tag), g in groups.items():
        tally.setdefault(tag, set()).update(x["end"] for x in g)
    return sorted(tally, key=lambda t: (-len(tally[t]), _tag_rank(t), t))


def build_quarterly_revenue(recs: list[dict]) -> list[dict]:
    """Convert records mixing single quarters / year-to-date (YTD) / annual into **single-quarter** revenue.

    The correct approach has two steps and cannot be merged into one (an early version coupled the two steps, so
    each year's Q4 slot was taken by the full-year cumulative figure, with 205 report periods wrong across all companies):

    Step 1: group by **(start date, tag)**.
        All figures in one filing share the same start date; e.g. a company's filings for one fiscal year are
            [1/1→3/31] Q1, [1/1→6/30] Q1+Q2, [1/1→9/30] Q1+Q2+Q3, [1/1→12/31] full year
        all with the same start. So grouping by start separates "single quarter + cumulative figures"
        from "four separate single-quarter filings".

        **Must also group by tag**, not just by start:
          (a) Different tags of the same company have different definitions; subtracting across tags is meaningless. E.g. GE 2015Q1:
              `Revenues` is 29.4B while `SalesRevenueServicesNet` is only 7.1B, a 4x difference.
          (b) Under the same start there may be two series at once: a cumulative series (90/180/270/365 days)
              and standalone single quarters (e.g. 91 days). Mixing them and sorting by length gives absurd differences.
              Example: MPC 2023 Q4 was once computed as 0.11B because of this; the correct value is about 36B.

    Step 2: within each group sort by period length ascending, and subtract adjacent values to get single quarters:
            Q1 = value₁
            Q2 = value₂ − value₁
            Q3 = value₃ − value₂
            Q4 = value₄ − value₃
        If the group also has real single-quarter records (filed directly in a 10-Q), the single-quarter original value is preferred for that period end.

    For 10-Qs where "each quarter is filed separately and values do not include each other" (e.g. Apple),
    subtraction yields 0 or negative numbers, which are discarded automatically and do not affect the result.
    """
    # ---- Step 1: group by (start, tag) -----------------------------------------
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in recs:
        if not r.get("start") or r.get("val") is None:
            continue
        try:
            d0 = datetime.strptime(r["start"], "%Y-%m-%d").date()
            d1 = datetime.strptime(r["end"], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        days = (d1 - d0).days
        if not (20 <= days <= 400):
            continue
        groups.setdefault((r["start"], r["tag"]), []).append(dict(r, days=days))

    # ---- Pick the best-quality series as the backbone, then **back-fill** earlier quarters from other tags ---
    # Only fill with earlier quarters; never overwrite ranges the backbone already has (to avoid mixing definitions).
    series_by_tag: dict[str, list[dict]] = {}
    for tag in _candidate_tags(groups):
        s = _series_for_tag(groups, tag)
        if s:
            series_by_tag[tag] = s
    if not series_by_tag:
        return []

    # Backbone choice: most periods first; ties broken by tag priority.
    # This way, when a company changes tags midway, the longest-covered stretch becomes the backbone.
    main_tag = min(series_by_tag,
                   key=lambda t: (-len(series_by_tag[t]), _tag_rank(t), t))
    out = list(series_by_tag[main_tag])
    start_of_main = out[0]["end"]

    # Back-fill: attach earlier quarters before the backbone
    patch: list[dict] = []
    for tag in _candidate_tags(groups):
        if tag == main_tag or tag not in series_by_tag:
            continue
        patch += [r for r in series_by_tag[tag] if r["end"] < start_of_main]
    if patch:
        # De-duplicate (one record per period end), and restrict the filled stretch to "one contiguous run"
        # reaching from the earliest up to the backbone start; if broken in the middle, discard the earlier part
        by_end = {r["end"]: r for r in sorted(patch, key=lambda r: r["end"])}
        seq = [by_end[e] for e in sorted(by_end)]
        keep: list[dict] = []
        for r in reversed(seq):
            if keep:
                d0 = datetime.strptime(r["end"], "%Y-%m-%d").date()
                d1 = datetime.strptime(keep[0]["end"], "%Y-%m-%d").date()
                if (d1 - d0).days > 120:
                    break
            keep.insert(0, r)
        out = keep + out

    out.sort(key=lambda r: r["end"])
    return out


def _series_for_tag(groups: dict[tuple[str, str], list[dict]],
                    tag: str) -> list[dict]:
    """Build the company's quarterly series from a **single tag** only.

    Why it must stay within a single tag: different tags have entirely different definitions, so subtracting across tags is meaningless.
    Take Marathon Petroleum (MPC) 2023Q4: the same period end has both
        RevenueFromContractWithCustomer...  full-year cumulative 148.379B
        SalesTypeLeaseInterestIncome        91 days 0.114B (lease interest income)
    The latter is a perfectly valid 91-day single-quarter record; if mixed with other tags and picked by preference,
    it would be chosen by mistake, leaving that quarter at only 0.3% of the true value.

    For the same reason, when a company changes tags midway, the series must be **stitched segment by segment** in the order of _candidate_tags,
    rather than mixing two tags in one series.
    """
    quarters: dict[str, list[dict]] = {}

    def add(end, val, filed, qlen, derived):
        """qlen must be **the length of the quarter itself** (for derived values, this segment minus the previous one),
        not the absolute length of the cumulative period."""
        if val is None or val <= 0:
            return
        quarters.setdefault(end, []).append(
            {"start": None, "end": end, "val": val, "filed": str(filed),
             "tag": tag, "days": qlen, "derived": derived})

    for (start, gtag), group in groups.items():
        if gtag != tag:
            continue
        # Duplicate records with the same period length (multiple filings): keep only the earliest filed
        uniq: dict[int, dict] = {}
        for g in group:
            cur = uniq.get(g["days"])
            if cur is None or str(g["filed"]) < str(cur["filed"]):
                uniq[g["days"]] = g
        seq = [uniq[d] for d in sorted(uniq)]          # ascending by period length

        if len(seq) == 1:
            # Only one report for this start: use it only if it really is a single quarter,
            # otherwise (e.g. only an annual report) it cannot be reconstructed; skip.
            g = seq[0]
            if g["days"] <= 150:
                add(g["end"], g["val"], g["filed"], g["days"], False)
            continue

        prev = None
        for g in seq:
            if prev is None:
                add(g["end"], g["val"], g["filed"], g["days"], False)
            else:
                add(g["end"], g["val"] - prev["val"], g["filed"],
                    g["days"] - prev["days"], True)
            prev = g

    result = []
    for end, cand in quarters.items():
        cand.sort(key=lambda c: (c["derived"], c["filed"]))
        result.append(cand[0])
    result.sort(key=lambda r: r["end"])
    return result


def _fiscal_periods(
    quarter_recs: list[dict],
    fiscal_year_end: str | None = None,
    fy_end_month: int | None = None,
) -> dict[str, tuple[int, int]]:
    """Infer the (fiscal_year, fiscal_quarter) for each period end date.

    The fiscal year end month is determined by the following priority (**no statistical inference**; see the comment on FY_END_MONTH for why):
      1. fy_end_month passed explicitly by the caller (from the FY_END_MONTH table)
      2. EDGAR's fiscal_year_end field (e.g. "12-31")
      3. FY_END_MONTH_DEFAULT (December)

    Assignment rule:
      fiscal year = end month <= fiscal year end month ? that year : previous year
      but an end date within 7 days after the fiscal year end month (e.g. JNJ's 2012-01-01,
      where a 52/53-week fiscal year drifts end of December into early January) still belongs to the previous fiscal year.
      fiscal quarter = numbered Q1..Q4 in chronological order within the same fiscal year.
    """
    ends = sorted({r["end"] for r in quarter_recs})
    if not ends:
        return {}

    # ---- Determine the fiscal year end month -------------------------------
    fy_month: int | None = fy_end_month
    if fy_month is None and fiscal_year_end:
        try:
            m = int(str(fiscal_year_end)[:2])
            if 1 <= m <= 12:
                fy_month = m
        except (ValueError, TypeError):
            fy_month = None
    if fy_month is None:
        fy_month = FY_END_MONTH_DEFAULT

    # ---- Assign fiscal years by date --------------------------------------
    # The goal is not "matching each company's official fiscal year label" (that is unsolvable: Apple calls the year ending 2024-09
    # FY2024, Walmart calls the year ending 2026-01 FY2026; conventions simply differ), but rather:
    #   (a) each fiscal year has **exactly 4 quarters** (otherwise the yoy base period is misaligned)
    #   (b) the order is correct
    #
    # Approach: **subtract fy_month months** from each end date and use the calendar year of the result as the fiscal year label.
    # Equivalent to "re-splitting years at the fiscal year end month"; satisfies both goals:
    #   JNJ (Dec): 2012-01-01 → 2011-01-01 → FY2011; 2012-12-30 → FY2012  ✓
    #   MSFT (Jun): 2026-06-30 → 2025-12-30 → FY2025; 2026-03-31 → FY2025  ✓
    #   WMT (Jan): 2026-01-31 → 2025-01-31 → FY2025; 2026-04-30 → FY2025  ✓
    # Note: the labels are **an internally consistent convention of this project** and may differ by 1 from the fiscal year numbers in company annual reports
    # (e.g. the period WMT itself calls FY2026 is labeled FY2025 here). This does not affect yoy/qoq
    # computations, because the base period is taken as "4 periods earlier under the same convention".
    def fiscal_year_of(end: str) -> int:
        y, mo = int(end[:4]), int(end[5:7])
        shifted = y * 12 + (mo - 1) - fy_month      # shift back fy_month months
        return shifted // 12

    by_fy: dict[int, list[str]] = {}
    for e in ends:
        by_fy.setdefault(fiscal_year_of(e), []).append(e)

    # ---- Number Q1..Q4 chronologically within each fiscal year -------------
    mapping: dict[str, tuple[int, int]] = {}
    for fy, es in by_fy.items():
        for i, e in enumerate(sorted(es), 1):
            mapping[e] = (fy, i if i <= 4 else 4)
    return mapping


# =============================================================================
# Part 2b: Revenue (BusinessQuant, single quarter + TTM) ★ currently used
# =============================================================================
#
# Data pipeline:
#   BusinessQuant  http://data.businessquant.com/historic  ->  single-quarter revenue + TTM revenue
#   SEC EDGAR      https://data.sec.gov/submissions/CIK... ->  filingDate (filing date)
#   The two are joined exactly on **report period end date** — checked company by company, dates match exactly
#   (including AAPL, WMT, KO, etc. with 52/53-week fiscal years).
#
# Quota: free 40 requests/day, so batching is required (5 per request). 28 stocks = 6 requests.
#       Raw JSON is cached locally, so repeated runs use no further quota.


def _cached_get_json(url: str, cache_path: str, ua: str,
                     timeout: int = 180) -> Any | None:
    """GET JSON with a disk cache. No request is sent at all on a cache hit."""
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 20:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:200]
        except Exception:                                  # noqa: BLE001
            pass
        log(f"    HTTP {exc.code}: {body}")
        return None
    except Exception as exc:                               # noqa: BLE001
        log(f"    request failed: {exc}")
        return None
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return data


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def resolve_ciks(tickers: list[str], outdir: str) -> tuple[dict[str, str], list[dict]]:
    """Find for each ticker the CIK with "the most suitable filing history".

    Note: SEC's ticker→CIK mapping alone is not enough. After a holding-company restructuring,
    SEC points the ticker to the **newly created entity**, which only has filings after the restructuring,
    while the history stays under the old entity (e.g. ExxonMobil's 2026 restructuring).
    So this checks whether the candidate CIK's tickers field really contains the ticker,
    and falls back to the old entity specified in CIK_OVERRIDES if not.
    """
    log("  fetching SEC ticker→CIK mapping table ...")
    try:
        cik_map = load_cik_map(outdir)
        log(f"  mapping table has {len(cik_map)} entries")
    except Exception as exc:                               # noqa: BLE001
        log(f"  ⚠ unable to fetch CIK mapping table: {exc} (will rely only on CIK_OVERRIDES)")
        cik_map = {}

    resolved: dict[str, str] = {}
    failures: list[dict] = []
    cache = os.path.join(outdir, "_cache")
    os.makedirs(cache, exist_ok=True)

    for sym in tickers:
        cands: list[str] = []
        m = cik_map.get(norm_ticker(sym))
        if m:
            cands.append(m)
        for c in CIK_OVERRIDES.get(sym, []):
            if c not in cands:
                cands.append(c)
        if not cands:
            failures.append({"kind": "revenue", "ticker": sym,
                             "reason": "no CIK in SEC map and no override"})
            continue

        chosen = None
        for c in cands:
            cp = os.path.join(cache, f"sec_sub_{c}.json")
            d = _cached_get_json(
                f"https://data.sec.gov/submissions/CIK{c}.json", cp, SEC_UA)
            time.sleep(SEC_SLEEP)
            if not d:
                continue
            listed = [str(x).upper() for x in (d.get("tickers") or [])]
            if sym.upper() in listed or len(cands) == 1:
                chosen = c
                break
            if chosen is None:
                chosen = c            # fallback: note it for now and keep looking for a better match
        if chosen:
            resolved[sym] = chosen

    return resolved, failures


def fetch_sec_filing_dates(cik: str, outdir: str) -> dict[str, str]:
    """Get {report period end date: filing date} for this CIK (10-K / 10-Q only).

    Covers recent and older archive files (SEC splits old filings into
    CIK##########-submissions-00N.json).
    """
    cache = os.path.join(outdir, "_cache")
    d = _cached_get_json(f"https://data.sec.gov/submissions/CIK{cik}.json",
                         os.path.join(cache, f"sec_sub_{cik}.json"), SEC_UA)
    if not d:
        return {}

    blocks = [d.get("filings", {}).get("recent", {})]
    for f in d.get("filings", {}).get("files", []):
        name = f.get("name")
        if not name:
            continue
        extra = _cached_get_json(
            f"https://data.sec.gov/submissions/{name}",
            os.path.join(cache, f"sec_{name}"), SEC_UA)
        time.sleep(SEC_SLEEP)
        if extra:
            blocks.append(extra)

    out: dict[str, str] = {}
    for blk in blocks:
        for form, fd, rd in zip(blk.get("form", []),
                                blk.get("filingDate", []),
                                blk.get("reportDate", []) or []):
            if form in ("10-K", "10-Q") and rd and fd:
                # If a report period was restated, take the later filing date (consistent with BQ showing the latest value)
                if rd not in out or fd > out[rd]:
                    out[rd] = fd
    return out


def fetch_revenue_bq(
    tickers: list[str], start: str, outdir: str
) -> tuple[pd.DataFrame, list[dict]]:
    """Get single-quarter + TTM revenue from BusinessQuant, then join with SEC filing dates.

    ⚠️ This function **no longer outputs** `fiscal_year` / `fiscal_quarter`. Reason: fiscal quarter assignment must
    first snap `period_end` to month-end (quarter ends of 52/53-week fiscal years drift into adjacent months), and the
    sequential numbering this function used early on was found to produce **duplicate labels** (AVGO 2020Q4, JNJ 2011/2016/2022
    each had two Q4 rows, 4 groups in total). The correct labels are computed by `merge_revenue.py` and written to
    `revenue_adjusted.csv`, which also has a uniqueness self-check.

    So this table only "fetches the data"; the fiscal quarter convention is always taken from the downstream `revenue_adjusted.csv`.
    """
    cols = ["ticker", "period_end", "report_date", "revenue", "revenue_ttm"]
    failures: list[dict] = []
    cache = os.path.join(outdir, "_cache")
    os.makedirs(cache, exist_ok=True)

    # ------------------------------------------------------------------ 1. BQ
    wanted = [t for t in tickers if t not in BQ_KNOWN_MISSING]
    for t in tickers:
        if t in BQ_KNOWN_MISSING:
            failures.append({"kind": "revenue", "ticker": t,
                             "reason": "BusinessQuant has no revenue data (company restructuring, "
                                       "ticker points to a new entity with only recent data)"})
            log(f"  ⚠ {t} is in BQ_KNOWN_MISSING, skipping (revenue dimension missing)")

    log(f"  requesting {'+'.join(BQ_SLUGS)} from BusinessQuant, "
        f"{BQ_BATCH} per batch, {len(list(_chunks(wanted, BQ_BATCH)))} batches in total")
    log(f"  free quota 40 requests/day; local cache hits use no quota")

    slug_param = ",".join(BQ_SLUGS)
    rows_all: list[dict] = []
    used_requests = 0

    for bi, batch in enumerate(_chunks(wanted, BQ_BATCH), 1):
        cp = os.path.join(cache, f"bq_{'_'.join(batch)}.json")
        fresh = not (os.path.exists(cp) and os.path.getsize(cp) > 20)
        url = (f"{BQ_BASE}/historic?slug={slug_param}"
               f"&ticker={','.join(batch)}&period=max&api_key={BQ_API_KEY}")
        d = _cached_get_json(url, cp, "STAT4011-Part1/1.0")
        if fresh:
            used_requests += 1
            time.sleep(0.5)
        if d is None:
            failures.append({"kind": "revenue", "ticker": "/".join(batch),
                             "reason": "BusinessQuant request failed"})
            log(f"[revenue bq {bi}] {'/'.join(batch):<40} failed")
            continue
        rows = d if isinstance(d, list) else d.get("data", [])
        rows_all += rows
        got = sorted({r["ticker"] for r in rows})
        log(f"[revenue bq {bi}] {'/'.join(batch):<40} "
            f"{'network' if fresh else 'cache'} -> {len(rows):>3} rows, {len(got)} stocks")

    log(f"  BusinessQuant quota used this run: {used_requests} requests"
        f" (0 means everything hit the cache)")

    if not rows_all:
        return pd.DataFrame(columns=cols), failures

    # ------------------------------------------------- 2. Split into single-quarter / TTM tables
    q: dict[tuple[str, str], float] = {}
    ttm: dict[tuple[str, str], float] = {}
    for r in rows_all:
        key = (r["ticker"], r["date"])
        metric = str(r.get("metric", ""))
        if "TTM" in metric.upper():
            ttm[key] = r["value"]
        else:
            q[key] = r["value"]

    # ------------------------------------------------- 3. Join with SEC filing dates
    log("  fetching SEC filing dates (free, uses no BQ quota) ...")
    ciks, cik_fail = resolve_ciks(sorted({t for t, _ in q}), outdir)
    failures += cik_fail

    filing: dict[str, dict[str, str]] = {}
    for sym, cik in ciks.items():
        filing[sym] = fetch_sec_filing_dates(cik, outdir)
    log(f"  fetched filing records for {len(filing)} stocks")

    # ------------------------------------------------- 4. Assemble output
    rows: list[dict] = []
    matched = unmatched = 0
    for (sym, period_end), val in q.items():
        rd = filing.get(sym, {}).get(period_end)
        if rd:
            matched += 1
        else:
            unmatched += 1
        rows.append({
            "ticker": sym,
            "fiscal_year": int(period_end[:4]),
            "fiscal_quarter": 0,          # filled in later by _fiscal_periods
            "period_end": period_end,
            "report_date": rd,            # may be None (2.5% of cases)
            "revenue": val,
            "revenue_ttm": ttm.get((sym, period_end)),
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols), failures

    # Fiscal quarter assignment is **not done here** — see this function's docstring.
    # The correct fiscal_year / fiscal_quarter are computed by merge_revenue.py.
    df = df[cols].sort_values(["ticker", "period_end"]).reset_index(drop=True)
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")
    df["revenue_ttm"] = pd.to_numeric(df["revenue_ttm"], errors="coerce")

    # ------------------------------------------------- 5. Coverage report
    log(f"\n  ---- Revenue data coverage (raw BQ length, not trimmed to a time window) ----")
    cov = (df.groupby("ticker")["period_end"]
             .agg(n_quarters="count", first="min", last="max"))
    log(f"  {'ticker':<8}{'quarters':>9}{'first period':>14}{'last period':>14}")
    for t, r in cov.sort_values("first").iterrows():
        log(f"  {t:<8}{r['n_quarters']:>7}{r['first']:>14}{r['last']:>14}")

    log(f"\n  filing date join: {matched} quarters matched, {unmatched} unmatched"
        f" ({matched / max(matched + unmatched, 1) * 100:.1f}% coverage)")
    if unmatched:
        log("  Unmatched ones are mostly early periods: BQ's data comes from a predecessor entity (e.g. Google Inc. before the Alphabet restructuring)"
            " and does not match filings under the current CIK. Can be ignored for backtesting.")

    no_data = [t for t in tickers if t not in set(df["ticker"])]
    if no_data:
        log(f"  ⚠ symbols without revenue data: {no_data}")

    return df, failures


# =============================================================================
# Part 2c: [Deprecated] home-built SEC XBRL revenue reconstruction
# =============================================================================
# Kept so we can fall back if BusinessQuant is unavailable.
# Known defect: reconstructing single quarters from cumulative figures does not converge on edge cases (205 report periods were found
# to treat full-year cumulative figures as single quarters). **Do not use for production data.**

def fetch_revenue_legacy(
    tickers: list[str], start: str, outdir: str
) -> tuple[pd.DataFrame, list[dict]]:
    """[Deprecated] Fetch quarterly revenue; returns (long DataFrame, list of failures)."""
    failures: list[dict] = []

    log("  fetching SEC ticker→CIK mapping table ...")
    try:
        cik_map = load_cik_map(outdir)
        log(f"  mapping table has {len(cik_map)} entries")
    except Exception as exc:                          # noqa: BLE001
        log(f"  ✗ unable to fetch CIK mapping table: {exc}")
        return pd.DataFrame(
            columns=["ticker", "fiscal_year", "fiscal_quarter",
                     "period_end", "report_date", "revenue"]
        ), [{"kind": "revenue", "ticker": "*",
             "reason": f"CIK map failed: {exc}"}]

    start_year = max(int(start[:4]), EDGAR_MIN_YEAR)
    rows: list[dict] = []

    for i, sym in enumerate(tickers, 1):
        # ---- Decide which CIK to use ----------------------------------------------
        # Normally use the CIK in SEC's mapping table. But after a holding-company restructuring,
        # SEC points the ticker to the new entity, which only has filings after the restructuring.
        # So try every candidate CIK here and pick the one with the most quarterly data.
        override = CIK_OVERRIDES.get(sym)
        if override:
            candidates = list(override)
        else:
            cik = cik_map.get(norm_ticker(sym))
            candidates = [cik] if cik else []

        if not candidates:
            failures.append({"kind": "revenue", "ticker": sym,
                             "reason": "no CIK found in SEC mapping"})
            log(f"[revenue {i:>2}/{len(tickers)}] {sym:<7} failed: not found in SEC mapping table")
            continue

        facts = None
        best_facts, best_cik, best_nq = None, None, -1
        for cand in candidates:
            f = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cand}.json")
            if not f:
                continue
            recs_try = _collect_revenue_facts(f, start_year)
            nq = len(build_quarterly_revenue(recs_try)) if recs_try else 0
            if nq > best_nq:
                best_facts, best_cik, best_nq = f, cand, nq
        if best_facts is None:
            failures.append({"kind": "revenue", "ticker": sym,
                             "reason": f"companyfacts unavailable (CIK {candidates})"})
            log(f"[revenue {i:>2}/{len(tickers)}] {sym:<7} failed: could not get companyfacts")
            continue

        facts = best_facts
        if override and best_cik != cik_map.get(norm_ticker(sym)):
            log(f"           ↳ {sym} has multiple filing entities, using {best_cik} "
                f"({best_nq} quarters; the entity in SEC's mapping table has less data)")

        recs = _collect_revenue_facts(facts, start_year)
        if not recs:
            failures.append({"kind": "revenue", "ticker": sym,
                             "reason": "no revenue tag matched"})
            log(f"[revenue {i:>2}/{len(tickers)}] {sym:<7} failed: no matching revenue tag")
            continue

        # Single quarters only
        quarter_recs = build_quarterly_revenue(recs)
        if not quarter_recs:
            failures.append({"kind": "revenue", "ticker": sym,
                             "reason": "no single-quarter values could be derived"})
            log(f"[revenue {i:>2}/{len(tickers)}] {sym:<7} failed: unable to reconstruct single-quarter revenue")
            continue

        fye = (facts.get("fiscalYearEnd") or {}).get("value") \
            if isinstance(facts.get("fiscalYearEnd"), dict) else None
        fmap = _fiscal_periods(quarter_recs, fye)

        # The same (fiscal year, fiscal quarter) may appear several times due to restatements -> keep the one with the earliest report_date
        dedup: dict[tuple[int, int], dict] = {}
        for r in quarter_recs:
            fy, fq = fmap.get(r["end"], (int(r["end"][:4]), 0))
            key = (fy, fq)
            prev = dedup.get(key)
            if prev is None or str(r["filed"]) < str(prev["filed"]):
                dedup[key] = {**r, "fy": fy, "fq": fq}

        for r in dedup.values():
            rows.append({
                "ticker":         sym,
                "fiscal_year":    r["fy"],
                "fiscal_quarter": r["fq"],
                "period_end":     r["end"],
                "report_date":    r["filed"],
                "revenue":        r["val"],
            })

        # Count which tags were actually used (there may be more than one after merging across tags)
        used = {}
        for r in dedup.values():
            used[r["tag"]] = used.get(r["tag"], 0) + 1
        tag_summary = ", ".join(
            f"{t.replace('RevenueFromContractWithCustomer', 'RevFromContract')[:44]}×{n}"
            for t, n in sorted(used.items(), key=lambda kv: -kv[1])
        )

        log(f"[revenue {i:>2}/{len(tickers)}] {sym:<7} OK {len(dedup):>3} quarters "
            f"(tags {tag_summary})")

    if not rows:
        return pd.DataFrame(
            columns=["ticker", "fiscal_year", "fiscal_quarter",
                     "period_end", "report_date", "revenue"]
        ), failures

    df = pd.DataFrame(rows).sort_values(["ticker", "period_end"]).reset_index(drop=True)

    # ---- Data coverage summary ---------------------------------------------
    # The number of quarters available varies a lot between companies, for two reasons of entirely different nature:
    #   (a) normal: the company listed/was founded late, or changed filing entity after M&A or restructuring
    #   (b) abnormal: the revenue tag was not matched -> add it to REVENUE_TAGS
    # Clearly short ones are listed here to quickly judge which category they fall into.
    log("\n  ---- Revenue data coverage per stock ----")
    cov = (df.groupby("ticker")["period_end"]
             .agg(n_quarters="count", first="min", last="max"))
    typical = int(cov["n_quarters"].median())
    log(f"  median number of quarters: {typical}")
    short = cov[cov["n_quarters"] < typical * 0.85]
    if len(short):
        log(f"  The following stocks have noticeably few quarters; please check the cause manually (late listing? merger/restructuring? tag not matched?):")
        for t, r in short.iterrows():
            log(f"    {t:<7} {r['n_quarters']:>3} quarters  "
                f"({r['first']} .. {r['last']})")
    else:
        log("  All stocks have consistent quarterly coverage, no abnormal gaps")

    return df, failures


# =============================================================================
# Part 3: Market indices (yfinance)
# =============================================================================

def fetch_benchmark(
    indices: list[tuple[str, str]], start: str, end: str | None
) -> tuple[pd.DataFrame, list[dict]]:
    """Fetch index closes; returns (long DataFrame, list of failures)."""
    import yfinance as yf

    frames: list[pd.DataFrame] = []
    failures: list[dict] = []

    for i, (sym, name) in enumerate(indices, 1):
        try:
            df = yf.download(sym, start=start, end=end, interval="1d",
                             auto_adjust=False, progress=False, threads=False)
        except Exception as exc:                      # noqa: BLE001
            failures.append({"kind": "benchmark", "ticker": sym, "reason": repr(exc)})
            log(f"[benchmark {i:>2}/{len(indices)}] {sym:<7} error: {exc}")
            continue

        if df is None or df.empty:
            failures.append({"kind": "benchmark", "ticker": sym,
                             "reason": "empty result"})
            log(f"[benchmark {i:>2}/{len(indices)}] {sym:<7} failed: empty data returned")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        idx = df.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)

        frames.append(pd.DataFrame({
            "date":       pd.to_datetime(idx).normalize(),
            "index_name": name,
            "close":      df["Close"].to_numpy(),
        }))
        log(f"[benchmark {i:>2}/{len(indices)}] {sym:<7} OK "
            f"{len(df):>5} rows ({df.index[0].date()} → {df.index[-1].date()})")

    if not frames:
        return pd.DataFrame(columns=["date", "index_name", "close"]), failures

    long = pd.concat(frames, ignore_index=True)
    # Sort by index_name first (i.e. "list all dates of one index, then the next"),
    # consistent with the row order of benchmark_sample.csv.
    long = long.sort_values(["index_name", "date"]).reset_index(drop=True)
    # Likewise drop incomplete trading days at the end (running during market hours mixes in the day's live quotes)
    long = _drop_partial_tail(long, [], "benchmark")
    return long, failures


# =============================================================================
# Main flow
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="STAT4011 Part 1 data acquisition: prices / revenue / market indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start", default=START,
                   help=f"start date YYYY-MM-DD (default {START})")
    p.add_argument("--end", default=END,
                   help="end date YYYY-MM-DD (default today)")
    p.add_argument("--only", choices=["prices", "revenue", "benchmark", "all"],
                   default="all", help="fetch only one kind of data (default all)")
    p.add_argument("--tickers", default=None,
                   help="comma-separated ticker list, overriding the default 28")
    p.add_argument("--revenue-source", choices=["bq", "legacy"], default="bq",
                   dest="revenue_source",
                   help="revenue data source: bq=BusinessQuant (default, recommended); "
                        "legacy=deprecated home-built SEC XBRL reconstruction")
    p.add_argument("--outdir", default=_SCRIPT_DIR,
                   help="output directory (default: this script's directory)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    outdir = os.path.abspath(args.outdir)
    log_path = setup_logging(outdir)

    tickers = ([t.strip().upper() for t in args.tickers.split(",") if t.strip()]
               if args.tickers else TICKERS)

    log("=" * 74)
    log("STAT4011 Part 1 — Data acquisition")
    log("=" * 74)
    log(f"Time range : {args.start} → {args.end or 'today'}")
    log(f"Stocks     : {len(tickers)}")
    log(f"Indices    : {len(INDICES)}")
    log(f"Output dir : {outdir}")
    log(f"Run log    : {log_path}")
    log("=" * 74)

    all_failures: list[dict] = []
    exit_code = 0

    # ---- 1. Prices -----------------------------------------------------------
    if args.only in ("prices", "all"):
        log("\n[1/3] Prices (Yahoo Finance)")
        df, fails = fetch_prices(tickers, args.start, args.end)
        all_failures += fails
        if df.empty:
            log("  ✗ no price data fetched")
            exit_code = 1
        else:
            path = os.path.join(outdir, "prices_raw.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            log(f"  → written to {path}  ({len(df)} rows, "
                f"{os.path.getsize(path)/1024:.1f} KB)")

    # ---- 2. Revenue ----------------------------------------------------------
    if args.only in ("revenue", "all"):
        if args.revenue_source == "bq":
            log("\n[2/3] Revenue (BusinessQuant single quarter + TTM, filing dates from SEC)")
            log(f"  batched fetching, {BQ_BATCH} per batch; free quota 40 requests/day")
            df, fails = fetch_revenue_bq(tickers, args.start, outdir)
        else:
            log("\n[2/3] Revenue ([Deprecated] home-built SEC EDGAR XBRL reconstruction)")
            log("  ⚠ this path is known to put full-year cumulative figures into some report periods; for fallback debugging only")
            df, fails = fetch_revenue_legacy(tickers, args.start, outdir)
        all_failures += fails
        if df.empty:
            log("  ✗ no revenue data fetched")
            exit_code = 1
        else:
            path = os.path.join(outdir, "revenue.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            log(f"  → written to {path}  ({len(df)} rows, "
                f"{os.path.getsize(path)/1024:.1f} KB)")

    # ---- 3. Indices ----------------------------------------------------------
    if args.only in ("benchmark", "all"):
        log("\n[3/3] Market indices (Yahoo Finance)")
        log("  Note: ^VIX has different units from the other indices; handle it separately in analysis")
        df, fails = fetch_benchmark(INDICES, args.start, args.end)
        all_failures += fails
        if df.empty:
            log("  ✗ no index data fetched")
            exit_code = 1
        else:
            path = os.path.join(outdir, "benchmark.csv")
            df.to_csv(path, index=False, encoding="utf-8-sig")
            log(f"  → written to {path}  ({len(df)} rows, "
                f"{os.path.getsize(path)/1024:.1f} KB)")

    # ---- Summary -------------------------------------------------------------
    log("\n" + "=" * 74)
    if all_failures:
        log(f"⚠ {len(all_failures)} item(s) could not be fetched:")
        for f in all_failures:
            log(f"    [{f['kind']}] {f['ticker']}: {f['reason']}")
        log("  These items are missing in the CSV (not filled with any fake data).")
        log("  Common causes: the symbol did not file the corresponding XBRL tag, or it listed after the start date.")
        exit_code = 1
    else:
        log("✓ All data fetched successfully, no failures")
    log("=" * 74)
    log(f"Full log: {log_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())

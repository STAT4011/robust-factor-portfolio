# data_wsy — Data acquisition notes

This directory provides all the raw data acquisition code and data files needed for STAT4011 Part 1.

> This file is **the README for this directory only**; it does not change the `README.md` at the repository root.

---

## 1. File list

### Committed to git

| File | Description |
|---|---|
| `get_stock_data.py` | Data acquisition (prices / revenue / market indices) |
| `merge_revenue.py` | **Single entry point for revenue correction**: fetch SEC → choose line item → derive Q4 → recompute fiscal quarter labels → compare with BQ → output → self-check |
| `prices_raw.csv` | Daily prices of 31 stocks |
| `benchmark.csv` | 6 US market indices + risk-free rate `^IRX` |
| `revenue.csv` | Quarterly revenue (**raw BusinessQuant values, with known errors**) |
| **`revenue_adjusted.csv`** | **Corrected revenue — use this one for backtests** |
| `revenue_merge_report.csv` | Chosen line item per stock + per-period replacement details + Q4 comparison + sanity checks |
| `_sec_ticker_map.json` | SEC's official ticker→CIK mapping table (874 KB, committed with the code to avoid re-downloading every time) |
| `README_wsy.md` | This README |
| `.gitignore` | Ignores caches and intermediate artifacts |

> **The whole project has only two Python entry points**: `get_stock_data.py` (fetching) and
> `merge_revenue.py` (revenue correction). All of the revenue methodology lives in the latter,
> which runs its own checks at the end (label uniqueness / non-positive revenue / key regression values).

> ⚠️ **There are two revenue files; don't use the wrong one**: `revenue.csv` holds BusinessQuant's **raw values**,
> known to have substantive errors for GE / BRK-B / EOG / NEM / MPC, and **Q4 was never checked**
> (BRK-B 2024Q4 is only 6% of the actual value, and EOG's Q4 is even negative);
> **`revenue_adjusted.csv` is the corrected version; use it for backtests.**
> See the section "Revenue correction" for the correction process.

> `revenue_adjusted.csv` is **not trimmed to a time window**; it outputs up to the data source's limit
> (`period_month` actually covers **2007-09 → 2026-07**). The backtest window is chosen by the analysis scripts.

### Not committed (see `.gitignore`)

| Path | Description |
|---|---|
| `_cache/` | Raw API responses from BQ and SEC. **Do not delete** — rerunning after deletion requires re-downloading (the BQ part uses 6 requests of quota, the SEC part is about 200 files) |
| `fetch_log.txt` | Log overwritten on every run |
| `__pycache__/` | Python bytecode |

> BQ responses in `_cache/` **use none of BusinessQuant's 40 requests/day quota** on a hit,
> so rerunning the script repeatedly while changing analysis code does not waste quota.

---

## 2. Dependencies

The script needs only two third-party packages:

```
yfinance
pandas
```

**The repository's existing `pyproject.toml` already has `pandas`; only `yfinance` needs to be added.**

Add it to the `[project] dependencies` list in `pyproject.toml`:

```toml
dependencies = [
    "pandas",
    "numpy",
    "scipy",
    "scikit-learn",
    "cvxpy",
    "matplotlib",
    "jupyter",
    "yfinance",     # ← new: market data acquisition
]
```

Then sync the environment:

```bash
uv lock && uv sync
```

> Note: HTTP requests use the Python standard library `urllib`; **`requests` is not needed**.
> `yfinance` automatically brings in the indirect dependencies it needs, such as `numpy` / `requests` / `beautifulsoup4`,
> so they need not be declared by hand.

### 2.1 `scikit-learn` and the `repair` option

`yfinance`'s `repair=True` uses `scikit-learn` to automatically fix Yahoo's split/price errors,
**but it is not a default dependency of yfinance**. Without it, yfinance raises
`ModuleNotFoundError: No module named 'sklearn'` for **every** ticker, and the download function **returns an empty DataFrame**,
which on the surface looks like "all the stock symbols are wrong".

The script handles this: at startup it probes for `scikit-learn`, and if missing it **automatically downgrades to `repair=False`**
and says so in the log. The data is still usable, just without automatic repair.

**Installing it is recommended** (your `pyproject.toml` already lists `scikit-learn`):

```bash
uv sync            # if the dependency is already in pyproject.toml
# or
pip install scikit-learn
```

To turn `repair` off completely, set `USE_REPAIR` at the top of the script to `False`.

---

## 3. Running

```bash
python get_stock_data.py                      # fetch everything (recommended)
python get_stock_data.py --only prices        # prices only
python get_stock_data.py --only revenue       # revenue only
python get_stock_data.py --only benchmark     # indices only
python get_stock_data.py --tickers AAPL,MSFT  # only the given stocks
python get_stock_data.py --start 2015-01-01 --end 2026-09-30
python get_stock_data.py --outdir /some/path  # set the output directory
```

**Exit code**: returns `1` if any item failed to fetch, `0` if everything succeeded (convenient for CI).

---

## 4. Table schemas

### 4.1 `prices_raw.csv`

| Column | Type | Description |
|---|---|---|
| `date` | date | Trading day |
| `ticker` | str | Stock symbol (Yahoo format; Berkshire is `BRK-B`) |
| `close` | float | Close price: **split-adjusted but not dividend-adjusted** — use this column for analysis |
| `adj_close` | float | Adjusted close (splits + dividends reinvested) — for robustness checks |
| `volume` | float | Volume |
| `dividend` | float | Cash dividend per share, recorded on the **ex-dividend date** row; `0` when there is no dividend |

> ⚠️ **Wording correction (2026-09-29)**: this file previously described `close` as "raw close (unadjusted)",
> which was **wrong**. `close` is in fact **already split-adjusted** — in testing, AVGO's
> 10:1 split on 2024-07-15 moved only +0.80% that day, GOOGL's 20:1 split on 2022-07-18 −2.46%,
> and WMT's 3:1 split on 2024-02-26 +1.85%, all without jumps.
> It **lacks only the dividend adjustment** (`close < adj_close`, with the gap widening as dividends accumulate).
>
> This property is exactly what this project needs: **splits must be adjusted** (otherwise returns would show a false
> −90% cliff), **dividends need not be adjusted** (the course states that dividends are not counted in portfolio returns).
> So using `close` rather than `adj_close` is correct; see the reasoning below.

> **Why analysis should use `close` rather than `adj_close`**: the course states that dividends are not counted in portfolio returns.
> With the non-dividend-adjusted `close`, the share price naturally drops on the ex-dividend date, so dividends are automatically excluded;
> using `adj_close` amounts to counting dividends in.

### 4.2 `revenue.csv` (upstream fetch table, with known errors)

| Column | Type | Description |
|---|---|---|
| `ticker` | str | Stock symbol |
| `period_end` | date | **Report period end date** (may be a 52/53-week fiscal year date, e.g. `2025-06-28`) |
| `report_date` | date | **Filing date** — the date the filing was submitted to SEC, i.e. the first date investors could see it |
| `revenue` | float | Revenue for the quarter (USD), **single-quarter value** (raw BusinessQuant value) |
| `revenue_ttm` | float | Sum of revenue over the trailing four quarters (trailing twelve months) |

> ⚠️ **This table has only 5 columns, with no `fiscal_year` / `fiscal_quarter`.**
> An earlier version generated fiscal quarters here by sequential numbering, which was found to produce **duplicate labels**
> (AVGO 2020Q4, JNJ 2011/2016/2022 each had two Q4 rows, 4 groups in total),
> because quarter ends of 52/53-week fiscal years drift into adjacent months. Correct fiscal quarter labels require first snapping `period_end`
> to month-end, which `merge_revenue.py` does and writes to `revenue_adjusted.csv`
> (which also has a uniqueness self-check). **So the fiscal quarter convention is always taken from downstream.**

> ⚠️ **This table holds raw BusinessQuant values with known substantive errors** (Q4 derivation residuals,
> wrong line items for some companies); **do not use it directly for quantitative analysis**.

> **`report_date` is crucial for backtesting (point-in-time).** Only use records with
> `report_date <= decision date`, otherwise look-ahead bias is introduced.
> The measured median filing lag is **32 days** (range 18–60 days) — about a month after quarter end
> before investors can see that quarter's revenue.

> **This table is not trimmed to a time window**; it outputs the longest history the data source provides (from about 2009),
> and the backtest window is chosen by the analysis scripts. Start years differ across companies; see "Known limitations" below.

#### Data sources and pipeline

| Field | Source |
|---|---|
| `revenue`, `revenue_ttm`, `period_end` | **BusinessQuant** `historic` endpoint (already reconstructed single-quarter / TTM values) |
| `report_date` | **SEC EDGAR** submissions endpoint (free official filing dates) |
| `fiscal_year`, `fiscal_quarter` | Inferred sequentially from `period_end` (without assuming "Q1 = 3/31") |

The two sources are joined exactly on **`period_end`** — checked company by company, BusinessQuant's
`date` and SEC's `reportDate` are **identical** (including AAPL, WMT, KO with 52/53-week
fiscal years), so no fuzzy matching is needed.

**Value check**: MSFT's four quarters of 2024 = 61.86 / 64.73 / 65.59 / 69.63 B,
matching the actual filings digit for digit; in 198 spot checks the deviation between `revenue_ttm` and the sum of the latest four single quarters
was always below 5%, showing that the two fields are consistent.

#### BusinessQuant quota

The free quota is **40 requests/day**, with **at most 5 tickers per request**:
- 31 stocks → **7 requests** (`⌈31/5⌉`)
- Raw responses are cached to `_cache/bq_*.json`, so **repeated runs use no further quota**
- Single quarter and TTM are fetched in the same request, using no extra quota

#### ⚠️ Correct use of `fiscal_year` / `fiscal_quarter`

**These two columns are for identification only; do not use them to compute yoy/qoq.** Three reasons:

1. **Companies' fiscal year numbering conventions simply differ, so no single rule can match all companies at once**:
   Apple calls the fiscal year ending 2024-09 FY2024, Microsoft calls the one ending 2026-06 FY2026,
   and Walmart calls the one ending 2026-01 FY2026. Any uniform rule will be off by 1 for some companies.
   (The script uses the convention "shift the end date back by the fiscal year end month", which is internally consistent but may differ
   by 1 from the numbering in company annual reports.)
2. **52/53-week fiscal years make end dates drift between adjacent months** (JNJ's Q4 sometimes falls at the end of December,
   sometimes in early January of the next year), so inferring fiscal quarter boundaries from dates inevitably fails in edge cases.
3. **The data source itself has gaps**, which can make the number of quarters in a fiscal year differ from 4 (e.g. JNJ is missing
   2021Q4 and 2022Q1, so that year does not have 4 quarters), in which case "the same quarter of the previous fiscal year" does not exist.

**The correct approach is shifting in time** (robust to any fiscal year scheme, and no fiscal year labels needed):

```python
df = pd.read_csv("revenue.csv").sort_values(["ticker", "period_end"])
g = df.groupby("ticker")

df["rev_yoy"] = df["revenue"] / g["revenue"].shift(4) - 1   # year-over-year: 4 periods earlier
df["rev_qoq"] = df["revenue"] / g["revenue"].shift(1) - 1   # quarter-over-quarter: 1 period earlier
```

This assumes each stock's period series is **essentially contiguous**. The script prints each stock's number of quarters, so contiguity can be
checked directly. **If there are gaps, `shift(4)` does not get "the same period last year"** — mark the gaps,
or use a more robust base period match (find the period whose report date is closest to 365 days earlier).

> In testing: the yoy base period gap from `shift(4)` falls around one year for JNJ/MSFT/WMT/AVGO/V/AAPL,
> with no obvious misalignment. But **JNJ has a full-year gap around 2022**, and results there
> need manual checking. Each company's quarterly contiguity can be seen in the coverage table printed by the script.

#### Revenue correction: a single SEC line item per company + Q4 check (important)

**BusinessQuant's revenue is correct for most companies, but has substantive errors for a few, and Q4 was never
verified.** `revenue.csv` cannot be used directly; it must go through `merge_revenue.py` for correction,
and backtests should use **`revenue_adjusted.csv`**.

**Two defects of the earlier approach (fixed 2026-09-29)**

1. **Comparing without separating line items.** The same company may have several "revenue" line items on SEC with very different definitions.
   E.g. BRK-B:

   | Line item | FY2024 | Meaning |
   |---|---|---|
   | `Revenues` | 371.4B | Total revenue (the definition in the filings) |
   | `RevenueFromContractWithCustomerExcludingAssessedTax` | 249.7B | Revenue from customer contracts only |

   The old approach compared the **narrow definition** against BQ's **total revenue**, a gap of over 30%, so it judged the correct value as wrong and
   replaced it with the narrow definition, making the series jump back and forth between two definitions (year-over-year would bounce between −30% and +40%).

2. **Q4 was never checked.** The old approach only compared 60–120-day single-quarter figures, and the 10-K **does not report
   Q4 separately**, so Q4 was always a blind spot. In testing, BQ's Q4 was wrong in many cases:

   | Symbol | Period | BQ value | Actual |
   |---|---|---|---|
   | BRK-B | 2024-12 | 5.72B | **94.92B** |
   | EOG | 2011-12 | −0.46B (negative) | about 3.0B |
   | V | 2016-09 | 0.55B | **4.26B** |
   | GE | December of each year 2013–2016 | 0.09–0.36x the adjacent quarters | — |

**The current approach (four steps)**

Step 1: **Choose a single SEC line item per company and never switch within the series.**
Selection: on **non-Q4** quarters, the line item with the highest agreement rate with BusinessQuant; when agreement rates tie,
take the one "whose scale is closest to BQ". The criterion uses **the lower confidence bound of the agreement rate**
(`rate − 1.5·√(rate(1−rate)/n)`) rather than the raw agreement rate — otherwise line items that are "100% on just 2 periods"
would win (LMT with 2 periods and MPC with 3 periods both once wrongly won).
If a period only has values under other line items → **leave that period blank; do not fill it from another line item**.

Step 2: **Generate the SEC single-quarter series from this line item**, including Q4.
* Q1–Q3: 3-month figures of 60–120 days in the 10-Q
* Q4 = 10-K full year (350–380 days) − the 9-month cumulative of Q3 in the same year (260–280 days),
  both must have **the same start date** (ensuring the same fiscal year) and **the same line item**
* When a period has several records (company restatements), take the one with the **earliest** `filed` — that is the figure
  investors saw at the time, which is exactly what this project's "look-ahead protection by announcement date" requires.
  This matters especially for companies with spin-offs: GE 2023Q3 was filed as 17.346B on 2023-10-24,
  and restated to 9.302B on 2024-10-22 due to the Vernova spin-off; only the earliest one matches the historical definition.

Step 3: **Compare with BQ, including Q4.** When the deviation > 20%, take the SEC value;
if SEC also cannot compute it or the value is non-positive, **leave it blank** (`revenue_source` marked `dropped`) —
a blank only means that period has no factor, while a wrong value goes straight into the ranking.

Step 4: **Sanity checks.** Periods with `revenue ≤ 0`, or a ratio to the median of the two quarters before and after < 0.5 / > 2,
are marked `suspect` (see the `data_flag` column) and written to the report for manual review.
Symbols with median revenue below USD 1 million skip the ratio check (early ONDS revenue was only tens of thousands of dollars,
so ratios are meaningless).

**Measured results**

| Metric | Old version | New version |
|---|---|---|
| BRK-B 2024Q4 | 5.72B ❌ | **94.916B** ✓ (matches the figure given in feedback) |
| V FY2016 Q4 | 0.554B ❌ | **4.261B** ✓ |
| EOG 2011Q4 | −0.46B ❌ | blank / corrected |
| GE 2023Q4 | 9.456B ❌ | **19.423B** ✓ |
| XOM revenue | **entirely missing** ❌ | **37 periods** (from 2017-03, old SEC entity CIK) ✓ |
| `(ticker,fy,q)` duplicates | 4 groups ❌ | **0** ✓ |
| Line item definition | two line items mixed ❌ | **a single line item per stock** ✓ |

**New columns**

| Column | Values | Meaning |
|---|---|---|
| `period_month` | `YYYY-MM` | `period_end` snapped to month-end. **This is the reliable key for quarter alignment** |
| `fiscal_year` / `fiscal_quarter` | — | Computed directly from `period_month` and the fiscal year end month; unique across the table |
| `revenue_source` | `BQ` / `SEC` / `dropped` | Which source the period's revenue comes from; `dropped` means left blank |
| `report_date_source` | `BQ` / `SEC` / empty | Which source the filing date comes from |
| `revenue_ttm_source` | `BQ` / `SEC` / `mixed` / empty | Nature of the period's TTM |
| `data_flag` | empty / `suspect:…` / `sec_unavailable` | Sanity check flag |

##### Why `period_month` must be snapped to month-end

Quarter ends of 52/53-week fiscal years fall a few days around month-end, so judging the fiscal quarter directly from `period_end` goes wrong:

| period_end | Actually belongs to | Note |
|---|---|---|
| JNJ `2016-01-01` (Saturday) | `2015-12` | FY2015 Q4 |
| AAPL `2024-09-28` (Saturday) | `2024-09` | FY2024 Q4 |
| INTC `2008-12-27` (Saturday) | `2008-12` | FY2008 Q4 |

Snapping rule (**do not use "which calendar month-end is closer"**; it fails on dates like 12-27,
putting INTC's Q4 in November, which then collides with the real Q1 as a duplicate label):

```
day >= 20  ->  this month
day <= 7   ->  previous month (drifted into the start of the month)
otherwise  ->  this month
```

The fiscal year end month applies the same rule to SEC's `fiscalYearEnd` (whose values may be specific dates such as `1227`,
`0103` rather than month-ends), and **is preferably inferred from the data** (quarter-end months recur).

##### ⚠️ `revenue_ttm` is "mixed-definition" for some periods

TTM is recomputed by the script as `sum of the latest 4 single quarters`. But **replaced periods are often scattered through the series**,
so TTM mixes two definitions:

```
BRK-B  ....................................S.S.SSS.SSS.SSSSSS.SSS.SSS.SSS.SS
GE     SSS.....SSS.............SSS..............S..........SSSSSSSSS.........
```

`revenue_ttm_source == 'mixed'` means that period's TTM mixes definitions; **use with caution**.
If you want to use TTM as a factor, for these stocks compute it yourself from the single-quarter `revenue` instead.
(The single-quarter `revenue` column does not have this problem — it has already been handled as "one definition for the whole series".)

**Usage**

```bash
python merge_revenue.py                                  # everything
python merge_revenue.py --tickers GE,BRK-B               # only the given ones
python merge_revenue.py --tol 0.10                       # stricter replacement threshold
```

When finished it prints the **self-check** results (label uniqueness / fiscal_quarter range / non-positive revenue /
key regression values). If any item fails, the exit code is 1.

**`revenue_merge_report.csv`** has a summary row for each stock (`period_end == "== SUMMARY =="`),
with the chosen line item, agreement rate, number of replaced/filled/blanked periods, `fiscalYearEnd` and CIK;
the other rows are per-period replacement details and Q4 comparison results.

#### Known limitations

1. **XOM has no revenue data.** After ExxonMobil's 2026 holding-company restructuring, BusinessQuant
   points `XOM` to a new entity with only recent data; the old entity has a full filing history at SEC,
   but no usable revenue series. **Price data is unaffected.** This symbol is listed in the script's
   `BQ_KNOWN_MISSING`.

2. **8 companies have less than 15 years of history** (due to restructurings or late listings; changing data source would not help):

   | ticker | Earliest period | Reason |
   |---|---|---|
   | ABBV | 2012-03 | Spun off from Abbott in 2013 |
   | AAOI | 2013-03 | Listed later |
   | GOOGL | 2014-09 | Alphabet restructuring in 2015; previously Google Inc. |
   | ONDS | 2016-06 | Listed later |
   | AVGO | 2017-04 | Avago acquired Broadcom in 2016 |
   | LIN | 2017-06 | Praxair–Linde merger in 2018 |
   | TEM | 2023-06 | Listed later |
   | CRWV | 2024-03 | Listed later |

   The other 19 stocks have full coverage (from 2009–2011, about 70 quarters).

   > **Impact on backtesting**: if "all symbols must have data" is required, the common window can only start in 2024-03;
   > for a longer window, you must trade off the number of symbols against the length of time.

3. **About 2.5% of `report_date` values are missing** (1 missing out of 211 rows in testing). The reason is that BQ's data comes from
   a predecessor entity (e.g. Google Inc. before the Alphabet restructuring) and does not match filings under the current CIK;
   mostly early periods.

4. **Q4 may be too small for some companies in some years.** BusinessQuant derives Q4 as "full year minus
   9-month cumulative"; when the 10-K and 10-Q revenue definitions differ this leaves a residual
   (seen: BRK-B 2024Q4 = 5.72B, normally around 90B).
   If an abnormally small value appears at the end of a reporting period, check it against the original 10-K.

5. **`BQ_API_KEY` is written at the top of the script** (it can also be overridden with the environment variable `BQ_API_KEY`).
   If the repository is made public, change it to read only from the environment variable.

### 4.3 `benchmark.csv`

| Column | Type | Description |
|---|---|---|
| `date` | date | Trading day |
| `index_name` | str | Index symbol |
| `close` | float | Closing level |

Indices included:

| `index_name` | Index |
|---|---|
| `^GSPC` | S&P 500 |
| `^IXIC` | Nasdaq Composite |
| `^NDX` | Nasdaq 100 |
| `^DJI` | Dow Jones Industrial Average |
| `^RUT` | Russell 2000 (small caps) |
| `^VIX` | Volatility index |
| `^IRX` | 13-week US T-bill yield (annualized percent, risk-free rate; also must not use `pct_change`) |

> ⚠️ **`^VIX` has different units from the other indices.** VIX is itself quoted as a percentage
> (e.g. `18.5` means annualized volatility of 18.5%), whereas `^GSPC` is in points (e.g. `5800`).
> Both are in the same table, so **after `pivot`, do not apply `pct_change()` to the whole table**,
> or VIX will yield "the rate of change of volatility", which has no statistical meaning. Handle each `index_name` separately in analysis.

---

## 5. Data sources and time range

| Data | Source | Coverage |
|---|---|---|
| Prices | Yahoo Finance (`yfinance`) | 2011-01-01 to present by default |
| Market indices | Yahoo Finance (`yfinance`) | Same as above |
| Revenue (single quarter + TTM) | **BusinessQuant** `historic` endpoint | From about 2009 (**not trimmed**) |
| Filing dates | **SEC EDGAR** submissions endpoint | Joined with BQ report periods |

**The time range of prices and indices is fixed** (default `2011-01-01`) for reproducibility:
anyone who clones and runs it gets the same data range. For a rolling window, pass `--start` explicitly.

**The revenue table is not trimmed**; it outputs up to the data source's limit (from about 2009), and the backtest window is chosen by the analysis scripts.

> **Why revenue does not use `yfinance`**: its financial statements cover only **4–5 years** and **do not provide filing dates**,
> so `report_date` cannot be constructed, and point-in-time backtesting is impossible.
>
> **Why revenue is not reconstructed ourselves from SEC XBRL**: SEC only gives cumulative figures (Q1, H1, 9M, full year);
> single quarters must come from subtraction, with many edge cases (fiscal years not matching calendar years, companies changing tags midway,
> 52/53-week fiscal years, multiple records with different definitions for the same period end). In testing, our own reconstruction once had
> **205 report periods treating full-year cumulative figures as single quarters**. BusinessQuant has done this step,
> and its values have been checked item by item against actual filings. The old home-built reconstruction code is kept in
> `fetch_revenue_legacy()` and can be used as a fallback with `--revenue-source legacy`.

---

## 6. Known limitations and data handling conventions

### 6.1 Revenue data handling rules

**Current path (BusinessQuant)**: the data source does most of the cleaning; the script does only three things:

1. **Batched fetching**: at most 5 tickers per request (more returns 400); single quarter and TTM are fetched in the same
   request.
2. **Join with filing dates**: exact match on `period_end` (i.e. BQ's `date`, SEC's `reportDate`)
   as the key, with no fuzzy matching. 97.5% of quarters were verified to match.
3. **Infer fiscal quarter numbers**: infer `fiscal_quarter` from the order of each company's actual `period_end`,
   **without assuming "Q1 = 3/31"** (some companies' fiscal years do not end in December, e.g. WMT ends in January,
   AAPL in September).

**If it can't be fetched, leave it blank; never fill with fake data.**

#### Appendix: [Deprecated] the old home-built SEC XBRL reconstruction path

`fetch_revenue_legacy()` (enabled with `--revenue-source legacy`) implemented the following rules.
**This path is known to put full-year cumulative figures into some report periods; do not use it for production data**; it is kept only for fallback debugging:

1. **Tag priority + auto-discovery**: different companies and years may use different XBRL revenue tags.
   The script first uses a fixed priority list, then automatically includes all tags whose names contain revenue/sales as a fallback
   (excluding non-revenue line items such as deferred / unrealized / cost / proceeds / tax).
   > This step is necessary: Newmont (NEM) reported with `RevenueMineralSales` before 2017;
   > without auto-inclusion, its 2011–2017 revenue would be lost entirely (38 → 62 quarters in testing).
2. **Merging across tags**: many companies changed tags around ASC 606 (effective 2018);
   for each report period take the highest-priority one.
3. **De-duplicating the same period**: company restatements leave multiple records; keep the one with the earliest `filed`.
4. **Single-quarter reconstruction**: periods of 60–120 days are treated as single quarters and used directly; 130–400 days are treated as cumulative
   and the previous segment is subtracted. **This step is the source of the known defect.**
5. **Fiscal years not matching calendar years**: inferred from the order of actual `period_end`.
6. **Choosing among multiple filing entities**: when a company undergoes a holding-company restructuring, query both the old and new CIK and take the one with more complete data.
   The mapping is written in `CIK_OVERRIDES` at the top of the script.
   > ExxonMobil (XOM) is such a case: SEC now points `XOM` to `CIK 0002115436`
   > ("ExxonMobil Holdings Corp", filing only since 2026-07), while the history is under the old entity
   > `CIK 0000034088` ("EXXON MOBIL CORP").

### 6.2 Other limitations

- **Price data may be shorter than 15 years**: late-listed symbols (e.g. `CRWV`, listed in 2025) naturally only have post-listing data.
- **Dividend effect without dividend adjustment**: `close` drops on ex-dividend dates, and the `dividend` column can be used for checking;
  for high-dividend symbols (e.g. `MCD`, `XOM`) the gap between `close` and `adj_close` is noticeably larger than for low-dividend symbols.
  (Note that `close` is **already split-adjusted**; don't treat it as the raw quote.)
- **Panel alignment is not done here**: this script only outputs raw data in long format. Before computing correlation matrices or covariance estimates,
  first `pivot` into wide format and `dropna` to get the common window where all stocks have data
  (because IPOs and trading halts cause gaps).

### 6.3 Trading halts and missing data

When `yfinance` encounters a trading halt it **does not add rows; the rows are simply missing**. Four cases need to be distinguished:

| Case | Symptom | Handling |
|---|---|---|
| Market-wide holiday | No stock has that date | Normal, nothing to do |
| Single-stock halt | Only that stock is missing that date | Produces `NaN` on alignment |
| Before an IPO | The stock's series starts on its listing date | Same as above |
| **Run during market hours** | **On the last day only some symbols have values, and prices are round numbers** | **Dropped automatically by the script** (see 6.4) |

**Do not use forward filling (`ffill`)** for halt days — it creates a run of fake "0% returns",
artificially lowering volatility. The correct approach is to drop the date during alignment.

### 6.4 Incomplete trading days at the end are dropped automatically

**Why this step is needed**: when `yfinance` runs **during market hours**, it mixes the day's **live quotes**
into the historical series. The observed symptom is that on the last day only about half the symbols have values, and the prices are round numbers
(e.g. `MSFT=498.00`, `JPM=340.00`) — those are not real closes.

**Harm**: if downstream relies on "all symbols having data on a given day" (correlation matrix, hierarchical clustering, cross-sectional factors),
that day becomes an entire row of `NaN`, which in turn means **no stocks can be selected on the decision date**.

**What the script does**: before writing the CSV, check backwards from the last day; as long as the number of symbols with values is less than
"the coverage of the second-to-last complete day", drop that whole row, until a complete trading day is reached.
The log prints which days were actually dropped.

> In testing (`_drop_partial_tail`):
> - Run during market hours (that day incomplete): the incomplete last day is dropped, and the last day falls back to the previous complete day ✓
> - Run after the close (series complete): no rows dropped; gaps in the middle of the series are **kept, not filled** ✓

**Note the difference**: the script only drops incomplete trading days **at the end**, and **does not handle gaps in the middle**.
Gaps in the middle of the series (e.g. a day with only 14/28 stocks) are real missing data and need to be handled with `dropna` at the alignment stage
— that is the analysis scripts' job; the fetch script does not fill them on its own.

---

## 7. Run time and notes

- **Prices**: 31 stocks × about 1 second ≈ **30–60 seconds**
- **Revenue**: 31 SEC requests + 1 mapping table ≈ **20–40 seconds** (the first run also downloads the mapping table).
  The BusinessQuant part goes in batches of 5, 7 requests in total, using the free quota
- **Indices**: 7 × about 1 second ≈ **10 seconds**
- **Revenue correction** (`merge_revenue.py`): 1 companyfacts request per stock (SEC is free
  and unlimited); about **1 minute** the first time, then served from the `_cache/` cache

**SEC requires contact information in the User-Agent**, otherwise it returns 403. The contact information is in the
`SEC_CONTACT` constant at the top of the script; edit it directly if you need to change it.

**Yahoo may rate-limit (HTTP 429)**: the script uses serial requests (`threads=False`) with intervals.
If it still fails, just rerun — the parts that succeeded are fetched again, which does not affect correctness.

---

## 8. Where to find the failure list

**Two places**, no need to look for other files:

1. **Console**: at the end of the run, all failed items and their reasons are printed
2. **`fetch_log.txt`**: the full record of the same run, including run parameters, per-stock results with timing, and failure reasons

Common causes of failures:

| Cause | Meaning |
|---|---|
| `no CIK found in SEC mapping` | The ticker cannot be found in SEC's mapping table (symbol spelling issue) |
| `companyfacts unavailable` | SEC did not return XBRL data for the company |
| `no revenue tag matched` | None of the revenue tags the script tried has data |
| `no single-quarter values could be derived` | Only annual or YTD data, so single quarters cannot be reconstructed |
| `empty result` | Yahoo returned empty data (halted too long or wrong symbol) |

**Important**: failed items are **missing** in the CSV; the script does not fill them with any estimates or fake data.

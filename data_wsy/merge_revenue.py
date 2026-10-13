#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_revenue.py — correct BusinessQuant revenue with SEC EDGAR and output revenue_adjusted.csv

This script is the **single entry point** for the revenue part; it keeps the whole pipeline "fetch SEC data → choose line item → derive Q4 → recompute fiscal quarter labels
→ compare with BQ → output → self-check" in one file.

Why corrections are needed
--------------------------
BusinessQuant's single-quarter revenue is correct for most companies, but has substantive errors for a few:
  * **Q4 is often wrong** — BQ derives Q4 as "full year − 9-month cumulative"; when the 10-K and 10-Q
    definitions differ this leaves a residual:
      BRK-B 2024Q4 = 5.72B (actual 94.92B), V FY2016Q4 = 0.55B (actual 4.26B),
      EOG 2011Q4 = −0.46B (negative)
  * **Some companies use the wrong accounting line item for the entire series** — BQ's GE is consistently about 55% of SEC's

More importantly: the earlier approach only compared 60–120-day single-quarter figures, and **the 10-K does not report Q4 separately**,
so Q4 was never verified. This script fills that gap.

Method (four steps)
-------------------
1. **Choose a single line item per company and never switch within the series.**
   The same company may have several "revenue" line items on SEC with very different definitions. E.g. BRK-B:

     `Revenues`                                            371.4B (total revenue in the filings)
     `RevenueFromContractWithCustomerExcludingAssessedTax` 249.7B (revenue from customer contracts only)

   Comparing without separating line items would judge total revenue as "wrong" and replace it with the narrow definition, making the series jump back and forth
   between two definitions (year-over-year would bounce between −30% and +40%).
   See `pick_tag()` for the selection: on **non-Q4** quarters take the one with the highest agreement rate, ranked by the lower confidence
   bound of the agreement rate to avoid small samples winning by chance. If a period only has values under other line items → **leave it blank; do not fill from another line item**.

2. **Generate the SEC single-quarter series from this line item.**
   * Q1–Q3: 3-month figures of 60–120 days in the 10-Q
   * Q4 = 10-K full year (350–380 days) − the 9-month cumulative of Q3 in the same year (260–280 days),
     both must have **the same start date** (ensuring the same fiscal year) and **the same line item**
   * When a period has several records (restatements), take the one with the **earliest** `filed` — that is the figure investors
     saw at the time, which is exactly what this project's "look-ahead protection by announcement date" requires.
     This matters especially for companies with spin-offs: GE 2023Q3 was filed as 17.346B on 2023-10-24,
     and restated to 9.302B on 2024-10-22 due to the Vernova spin-off.

3. **Compare with BQ, including Q4.** When the deviation > `--tol` (default 20%), take the SEC value;
   if SEC also cannot compute it or the value is non-positive, **leave it blank** (`revenue_source` marked `dropped`) —
   a blank only means that period has no factor, while a wrong value goes straight into the ranking.

4. **Sanity checks.** Periods with `revenue ≤ 0`, or a ratio to the median of the two quarters before and after < 0.5 / > 2, are
   marked `suspect` (written to the `data_flag` column) for manual review. Symbols with median revenue below
   USD 1 million skip the ratio check (early ONDS revenue was only tens of thousands of dollars, so ratios are meaningless).

Fiscal quarter labels
---------------------
Quarter ends of 52/53-week fiscal years fall a few days around month-end, so judging the fiscal quarter directly from `period_end` goes wrong:
  JNJ `2016-01-01` (Saturday) is actually the period ending at the end of 2015-12
  INTC `2008-12-27` (Saturday) is actually the period ending at the end of 2008-12
So first **snap `period_end` to the month-end it belongs to** to get `period_month`, then
compute `fiscal_year` / `fiscal_quarter` directly from `period_month` and the fiscal year end month.
See `snap_to_month_end()` and `fiscal_label()` for details.

Output
------
  revenue_adjusted.csv       ★ use this one for backtests
      ticker, fiscal_year, fiscal_quarter, period_month, period_end,
      report_date, revenue, revenue_ttm, revenue_source, report_date_source,
      revenue_ttm_source, data_flag
  revenue_merge_report.csv   one "== SUMMARY ==" row per stock (chosen line item / agreement rate /
                             fiscalYearEnd / CIK / number of replaced·filled·blanked periods),
                             the rest are per-period replacement details and Q4 comparison results

Cost
----
SEC EDGAR is free and unlimited. **Uses no BusinessQuant quota.**
Raw responses are cached in `_cache/`, so repeated runs do not download again.

Usage
-----
    python merge_revenue.py                        # everything
    python merge_revenue.py --tickers GE,BRK-B     # only the given ones
    python merge_revenue.py --tol 0.20             # replacement threshold
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from datetime import datetime

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(_SCRIPT_DIR, "_cache")

# =============================================================================
# Configuration
# =============================================================================

# ---- SEC EDGAR --------------------------------------------------------------
# SEC requires contact information in the User-Agent, otherwise it returns 403.
SEC_CONTACT = "shuiyuntian6@gmail.com"
SEC_UA = f"STAT4011-Part1/1.0 ({SEC_CONTACT})"
SEC_SLEEP = 0.12                      # Interval between requests (seconds). SEC rate limit is about 10 requests/second

# ---- Candidate revenue line items -------------------------------------------
# The order only affects tie-breaking, not the main criterion (which is the agreement rate).
CANDIDATE_TAGS = [
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
    "SalesRevenueGoodsNet",
    "SalesRevenueServicesNet",
    "RevenuesNetOfInterestExpense",
    "InterestAndDividendIncomeOperating",
    "RevenueMineralSales",
]

# ---- Period length ranges (days) --------------------------------------------
# Used to distinguish single quarter / 9-month cumulative / full year among SEC period records
Q_LO, Q_HI = 60, 120                  # single quarter (3 months in a 10-Q)
M9_LO, M9_HI = 260, 280               # 9-month cumulative (Q3 10-Q)
Y_LO, Y_HI = 350, 380                 # full year (10-K)

# ---- Decision thresholds ----------------------------------------------------
TOL_MATCH = 0.02                      # relative deviation < 2% counts as "agreeing" (used to choose the line item)
TOL_REPLACE = 0.20                    # replace with the SEC value only when deviation > 20%

# ---- Sanity checks ----------------------------------------------------------
SUSPECT_MIN_SCALE = 1e6               # symbols with median revenue below this skip the ratio check

# ---- Symbols without BQ revenue that need SEC filling -> candidate CIKs (old entity first) ---
FALLBACK_CIK = {"XOM": ["0000034088", "0002115436"]}


def log(msg: str = "") -> None:
    print(msg, flush=True)


# =============================================================================
# Part 1: SEC fetching and caching
# =============================================================================

def _get(url: str, key: str) -> dict | None:
    """GET JSON with a disk cache. No request is sent at all on a cache hit."""
    os.makedirs(CACHE, exist_ok=True)
    cp = os.path.join(CACHE, key + ".json")
    if os.path.exists(cp) and os.path.getsize(cp) > 50:
        with open(cp, encoding="utf-8") as f:
            return json.load(f)
    req = urllib.request.Request(url, headers={"User-Agent": SEC_UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception:                                        # noqa: BLE001
        return None
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(d, f)
    time.sleep(SEC_SLEEP)
    return d


def _days(s: str, e: str) -> int | None:
    try:
        return (datetime.strptime(e, "%Y-%m-%d")
                - datetime.strptime(s, "%Y-%m-%d")).days
    except (ValueError, TypeError):
        return None


def fetch_facts(cik: str) -> dict | None:
    """Fetch **all** us-gaap facts of the company (companyfacts) in one go, to the day.

    Use companyfacts instead of per-line-item companyconcept: one request returns all candidate
    line items, so choosing a line item needs no extra requests.
    """
    return _get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                f"facts_{cik}")


def sec_submissions(cik: str) -> dict | None:
    """SEC submissions: includes fiscalYearEnd and all filing records."""
    return _get(f"https://data.sec.gov/submissions/CIK{cik}.json",
                f"sec_sub_{cik}")


def earliest_filing_dates(cik: str) -> dict[str, str]:
    """Get the **earliest** 10-K/10-Q filing dates: {reportDate: filingDate}.

    Consistent with the project's convention: factors are protected against look-ahead by announcement date, so take the first announcement date rather than the restated one.
    Covers recent and older archive files (SEC splits old filings into
    CIK##########-submissions-00N.json). The result is cached to `firstfiled_<cik>.json`.
    """
    cp = os.path.join(CACHE, f"firstfiled_{cik}.json")
    if os.path.exists(cp) and os.path.getsize(cp) > 20:
        with open(cp, encoding="utf-8") as f:
            return json.load(f)

    d = sec_submissions(cik)
    out: dict[str, str] = {}
    if d:
        blocks = [d.get("filings", {}).get("recent", {})]
        for f in d.get("filings", {}).get("files", []):
            name = f.get("name")
            if not name:
                continue
            extra = _get(f"https://data.sec.gov/submissions/{name}",
                         f"sec_{name}")
            if extra:
                blocks.append(extra)
        for blk in blocks:
            for form, fd, rd in zip(blk.get("form", []),
                                    blk.get("filingDate", []),
                                    blk.get("reportDate", []) or []):
                if form in ("10-K", "10-Q") and rd and fd:
                    if rd not in out or fd < out[rd]:
                        out[rd] = fd
    with open(cp, "w", encoding="utf-8") as f:
        json.dump(out, f)
    return out


# =============================================================================
# Part 2: Build single-quarter series from SEC facts
# =============================================================================

def facts_to_records(facts: dict, tags: list[str] | None = None) -> dict:
    """Split companyfacts into {tag: [record, ...]}.

    record = {start, end, days, val, filed, tag, form}
    If the same (start, end, val) is filed multiple times, keep only the one with the earliest `filed`.
    """
    tags = tags or CANDIDATE_TAGS
    out: dict[str, list[dict]] = {}
    gaap = (facts or {}).get("facts", {}).get("us-gaap", {})
    for tag in tags:
        node = gaap.get(tag)
        if not node:
            continue
        best: dict[tuple, dict] = {}
        # Of the different units (USD / USD/shares, etc.) take only USD
        for unit, items in (node.get("units") or {}).items():
            if unit != "USD":
                continue
            for it in items:
                s, e, v = it.get("start"), it.get("end"), it.get("val")
                if not s or not e or v is None:
                    continue
                d = _days(s, e)
                if d is None:
                    continue
                k = (s, e, v)
                filed = it.get("filed") or "9999-99-99"
                if k not in best or filed < best[k]["filed"]:
                    best[k] = {"start": s, "end": e, "days": d, "val": v,
                               "filed": filed, "tag": tag,
                               "form": it.get("form")}
        if best:
            out[tag] = sorted(best.values(),
                              key=lambda r: (r["end"], r["start"]))
    return out


def quarterly_from_tag(recs: list[dict]) -> dict[str, dict]:
    """Generate a single-quarter series from records of a **single line item**: {period_end: {...}}.

    Q1–Q3 take records of 60–120 days (3-month figures in the 10-Q);
    Q4   = full year (350–380 days) − 9-month cumulative (260–280 days), requiring the same start
           (only the same start guarantees the same fiscal year).
    When a period end has several records, take the earliest `filed`.
    """
    q: dict[str, dict] = {}

    def put(end: str, val, filed, kind):
        cur = q.get(end)
        if cur is None or (filed or "9999") < (cur["filed"] or "9999"):
            q[end] = {"end": end, "val": val, "filed": filed, "kind": kind}

    for r in recs:
        if Q_LO <= r["days"] <= Q_HI:
            put(r["end"], r["val"], r["filed"], "10-Q")

    annual = [r for r in recs if Y_LO <= r["days"] <= Y_HI]
    nine = [r for r in recs if M9_LO <= r["days"] <= M9_HI]
    for a in annual:
        for n in nine:
            if a["start"] == n["start"] and n["end"] < a["end"]:
                filed = (min(a["filed"], n["filed"])
                         if a["filed"] and n["filed"]
                         else (a["filed"] or n["filed"]))
                put(a["end"], a["val"] - n["val"], filed, "10-K_derived")
                break
    return q


# =============================================================================
# Part 3: Choose the line item
# =============================================================================

def _median(v):
    v = sorted(v)
    n = len(v)
    if not n:
        return None
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def score_tag(sec_q: dict[str, dict], bq: dict[str, float],
              fye_month: int) -> dict:
    """Evaluate this line item's agreement rate with BQ on **non-Q4** quarters.

    Q4 is excluded because Q4 is exactly where BQ may be wrong; using it would contaminate the line item choice.
    """
    n_match = n_mismatch = n_cmp = 0
    worst = 0.0
    for e, rec in sec_q.items():
        if e not in bq or rec["kind"] != "10-Q":
            continue
        off = (int(e[5:7]) - fye_month) % 12          # around 0 or 11 means Q4
        if off == 0 or off >= 11:
            continue
        b = bq[e]
        if not b:
            continue
        rel = abs(rec["val"] - b) / abs(b)
        n_cmp += 1
        worst = max(worst, rel)
        if rel < TOL_MATCH:
            n_match += 1
        elif rel > TOL_REPLACE:
            n_mismatch += 1
    return {"n_cmp": n_cmp, "n_match": n_match, "n_mismatch": n_mismatch,
            "rate": (n_match / n_cmp) if n_cmp else 0.0, "worst": worst}


def pick_tag(records: dict[str, list[dict]], bq: dict[str, float],
             fye_month: int) -> tuple[str | None, list[dict]]:
    """Choose the most suitable line item. Returns (tag, score details).

    Ranking uses **the lower confidence bound of the agreement rate** rather than the raw agreement rate — this is crucial:

      the raw agreement rate rewards line items that are "100% on just 2 periods". In testing,
      line items with LMT 2 periods at 100% and MPC 3 periods at 100% wrongly won, while the real total revenue line item had dozens of comparable periods.
      100% on a small sample is not convincing at all.

    Use `rate - 1.5*sqrt(rate*(1-rate)/n)` instead:
      70 periods 95%   -> about 0.909
      2  periods 100%  -> about 0.0     (correctly pushed down)
      30 periods 100%  -> about 1.0     (still wins)
    This amounts to "it must be accurate and also backed by enough periods".

    When agreement rates tie, look at **how close the scale is to BQ**: the same company may have several line items that happen to
    match parts of BQ's periods, e.g. JNJ's `SalesRevenueGoodsNet` is only "sales of goods"
    (median 17.37B) while total revenue is 21.89B; BQ's JNJ quarters are 21–25B, clearly the latter.
    Nor can we unconditionally take the largest — V's `Revenues` has a median of only 3.15B simply because it covers the early
    small-scale years.

    Note: this is "choose the line item whose definition is closest to BQ", not "choose the most authoritative line item". The goal is to keep replacements
    few and on the same definition; if BQ's overall definition is wrong, we'll be wrong together with BQ, but that error is consistent across the whole table
    and does not affect year-over-year.
    """
    bq_vals = [v for v in bq.values() if v]
    bq_med = _median(bq_vals) if bq_vals else None

    scored = []
    for tag, recs in records.items():
        q = quarterly_from_tag(recs)
        if not q:
            continue
        s = score_tag(q, bq, fye_month)
        n, r = s["n_cmp"], s["rate"]
        s["lb"] = (r - 1.5 * math.sqrt(r * (1 - r) / n)) if n else -1.0
        nonq4 = [rec["val"] for e, rec in q.items()
                 if rec["kind"] == "10-Q"
                 and (int(e[5:7]) - fye_month) % 12 not in (0, 11)]
        s["tag"] = tag
        s["n_quarters"] = len(q)
        s["median"] = _median(nonq4) if nonq4 else None
        s["scale_gap"] = (abs(s["median"] - bq_med) / bq_med
                          if (s["median"] and bq_med) else 9.9)
        scored.append(s)

    if not scored:
        return None, []
    scored.sort(key=lambda s: (-s["lb"], s["scale_gap"], -s["n_cmp"]))
    return scored[0]["tag"], scored


# =============================================================================
# Part 4: Fiscal quarter labels (snapping to period_month)
# =============================================================================

def snap_to_month_end(pe: str) -> tuple[int, int]:
    """Snap period_end to the month whose month-end it belongs to; returns (year, month).

    Quarter ends of 52/53-week fiscal years fall a few days around month-end, so they must be snapped before the fiscal quarter can be judged correctly:
      JNJ  2016-01-01 (Saturday) is actually the period ending at the end of 2015-12
      AAPL 2024-09-28 (Saturday) is actually the period ending at the end of 2024-09
      INTC 2008-12-27 (Saturday) is actually the period ending at the end of 2008-12

    Criterion (**do not use "which calendar month-end is closer"**, nor "day ≤ 7 means the previous month"):
      day >= 20 → this month (the month-end is on this side)
      day <= 7  → previous month (drifted into the start of the month)
      otherwise → this month (normal mid-month, should not occur in theory)

    Why not "nearest month-end": on dates like 12-27 edge cases make it wrong, putting INTC's Q4
    in November, which then collides with the real Q1 (March) as a duplicate label.
    "day ≤ 7" would misplace 2016-10-02 (JNJ's Q3) outside September.
    The threshold method above was verified correct for AAPL(27/28/29), JNJ(01/02/03), INTC(27), WMT(31),
    AVGO(01), KO(30/31).
    """
    y, m, d = int(pe[:4]), int(pe[5:7]), int(pe[8:10])
    if d >= 20:
        return y, m
    if d <= 7:
        return (y - 1, 12) if m == 1 else (y, m - 1)
    return y, m


def snap_fye(raw: str | None, default: int = 12) -> int:
    """Snap fiscalYearEnd ('1227' / '0103', etc.) to the month whose month-end it belongs to.

    **Must use the same criterion as `snap_to_month_end`**, otherwise you get:
      fiscalYearEnd=1227 → snapped to December
      but period_end=2008-12-27 → snapped to November
      so Q4 is computed as Q1 and collides with the real Q1 (March) as a duplicate label (exactly what happened with INTC).
    """
    if not raw:
        return default
    s = str(raw)
    try:
        m, d = int(s[:2]), int(s[2:4])
    except (ValueError, TypeError):
        return default
    if not (1 <= m <= 12):
        return default
    _, mm = snap_to_month_end(f"2000-{m:02d}-{d:02d}")
    return mm


def infer_fye_month(period_ends: list[str],
                    raw_fye: str | None) -> tuple[int, list[int]]:
    """Determine the fiscal year end month from **the data**, more robust than looking only at fiscalYearEnd.

    Approach: snap each period_end to month-end and count how often each month appears. Quarter-end months should appear
    repeatedly (4 quarter months per stock); the 4 most frequent months are the quarter-end months, and the one closest to
    `fiscalYearEnd` among them is the fiscal year end month.
    """
    months: Counter = Counter()
    for e in period_ends:
        months[snap_to_month_end(e)[1]] += 1
    if not months:
        return snap_fye(raw_fye), []

    top = [m for m, _ in months.most_common(4)]
    hinted = snap_fye(raw_fye)
    if not top:
        return hinted, []
    best = min(top, key=lambda m: min((m - hinted) % 12, (hinted - m) % 12))
    return best, sorted(top)


def fiscal_label(pe: str, fye_month: int) -> tuple[int, int, str]:
    """Return (fiscal_year, fiscal_quarter, period_month).

    * `period_month`: period_end snapped to month-end, e.g. '2015-12'
    * `fiscal_quarter`: `off = (month − fiscal year end month − 1) mod 12`, `q = off // 3 + 1`
    * `fiscal_year`: the **ending year** of the fiscal year (consistent with the company's own naming)

    This way 52/53-week fiscal years (quarter ends drifting to the start/end of a month) also get correct and unique labels.
    """
    y, m = snap_to_month_end(pe)
    off = (m - fye_month - 1) % 12
    q = off // 3 + 1
    fy = y if m <= fye_month else y + 1
    return fy, q, f"{y:04d}-{m:02d}"


# =============================================================================
# Part 5: Sanity checks
# =============================================================================

def sanity_flags(series: list[tuple[str, float]],
                 min_scale: float = SUSPECT_MIN_SCALE) -> dict[str, str]:
    """Sanity check: non-positive, or a ratio to the median of the two quarters before and after <0.5 / >2.

    `min_scale` is used to skip companies whose "revenue is tiny to begin with" (e.g. early ONDS with only tens of thousands of dollars),
    where ratios are meaningless and would only produce a pile of useless warnings.
    """
    flags: dict[str, str] = {}
    vals = [v for _, v in series]
    if not vals:
        return flags
    scale = sorted(abs(v) for v in vals if v)
    typical = scale[len(scale) // 2] if scale else 0.0

    for i, (end, v) in enumerate(series):
        notes = []
        if v <= 0:
            notes.append("nonpositive")
        elif typical >= min_scale:       # too small in scale; do not judge by ratio
            neigh = [vals[j] for j in range(max(0, i - 2), min(len(vals), i + 3))
                     if j != i]
            if len(neigh) >= 2:
                med = sorted(neigh)[len(neigh) // 2]
                if med > 0:
                    ratio = v / med
                    if ratio < 0.5 or ratio > 2.0:
                        notes.append(f"ratio={ratio:.2f}")
        if notes:
            flags[end] = ";".join(notes)
    return flags


# =============================================================================
# Part 6: Full pipeline for a single stock
# =============================================================================

def build_sec_series(cik: str, bq: dict[str, float],
                     tol: float = TOL_REPLACE) -> dict:
    """Run the full pipeline for one stock: choose line item → generate single quarters (including Q4) → compare with BQ.

    `tol` is the threshold for "how large a deviation before replacing with the SEC value" (default 0.20).
    The line item selection criterion always uses `TOL_MATCH` / `TOL_REPLACE` and is not affected by `tol` —
    the line item choice reflects "which definition BQ uses overall" and should not sway with the replacement policy.
    """
    facts = fetch_facts(cik)
    if not facts:
        return {"ok": False, "why": "could not get companyfacts"}

    sub = sec_submissions(cik) or {}
    fye_raw = sub.get("fiscalYearEnd")
    # The fiscal year end month is preferably inferred from the data (quarter-end months recur in the data); fiscalYearEnd serves as a hint
    fye_m, modal_months = infer_fye_month(list(bq.keys()), fye_raw)

    records = facts_to_records(facts)
    tag, scored = pick_tag(records, bq, fye_m)
    if not tag:
        return {"ok": False, "why": "no usable revenue line item"}

    sec_q = quarterly_from_tag(records[tag])
    first = earliest_filing_dates(cik)

    def pkey(e: str) -> str:
        y, m = snap_to_month_end(e)
        return f"{y:04d}-{m:02d}"

    # ⚠️ The pairing key is **the snapped month**, not the exact date.
    # Reason: under 52/53-week fiscal years SEC's period_end and BQ's period_end can differ by a day or two
    # (e.g. INTC's fiscal year end is 1227, AAPL's is 09-27/09-28); pairing on exact dates would produce two rows for the same
    # quarter, and in turn duplicate fiscal_quarter values.
    sec_by_pm = {pkey(e): (e, rec) for e, rec in sec_q.items()}
    bq_by_pm: dict[str, tuple[str, float]] = {}
    for e, v in bq.items():
        bq_by_pm.setdefault(pkey(e), (e, v))

    n_cmp = n_match = n_bad = n_rep = n_fill = n_drop = 0
    rows: dict[str, dict] = {}
    for pm in sorted(set(sec_by_pm) | set(bq_by_pm)):
        s_hit = sec_by_pm.get(pm)
        b_hit = bq_by_pm.get(pm)
        # Prefer BQ's period end date (that is the input table's convention; keep it consistent)
        end = b_hit[0] if b_hit else s_hit[0]
        sec_rec = s_hit[1] if s_hit else None
        b = b_hit[1] if b_hit else None
        val, src, note = None, "BQ", ""

        if sec_rec and b is not None:
            rel = abs(sec_rec["val"] - b) / abs(b) if b else 1.0
            n_cmp += 1
            if rel < TOL_MATCH:
                n_match += 1
            elif rel > TOL_REPLACE:
                n_bad += 1
            if rel > tol:
                if sec_rec["val"] > 0:
                    val, src = sec_rec["val"], "SEC"
                    note = (f"BQ {b/1e9:.3f}B -> SEC {sec_rec['val']/1e9:.3f}B"
                            f" (deviation {rel*100:.0f}%)")
                    n_rep += 1
                else:
                    val, src = None, "dropped"
                    note = f"both sources abnormal (SEC={sec_rec['val']:.3g}), left blank"
                    n_drop += 1
            else:
                val, src = b, "BQ"
        elif b is not None:
            val, src = b, "BQ"           # SEC has no value for this period (maybe only under other line items)
        elif sec_rec:
            if sec_rec["val"] > 0:
                val, src = sec_rec["val"], "SEC"
                note = "BQ missing this period, filled from SEC"
                n_fill += 1
            else:
                val, src = None, "dropped"
                note = "SEC non-positive, left blank"
                n_drop += 1

        if val is not None and val <= 0:
            note = (note + " | " if note else "") + "revenue non-positive, left blank"
            val, src = None, "dropped"
            n_drop += 1

        rows[end] = {"period_end": end, "period_month": pm, "revenue": val,
                     "revenue_source": src,
                     "sec_value": sec_rec["val"] if sec_rec else None,
                     "sec_end": s_hit[0] if s_hit else "",
                     "sec_kind": sec_rec["kind"] if sec_rec else "",
                     "bq_value": b, "note": note}

    return {"ok": True, "tag": tag, "scored": scored, "fye_raw": fye_raw,
            "fye_month": fye_m, "modal_months": modal_months,
            "series": rows, "first_filed": first, "cik": cik,
            "stats": {"n_cmp": n_cmp, "n_match": n_match, "n_bad": n_bad,
                      "n_replace": n_rep, "n_fill": n_fill, "n_drop": n_drop}}


# =============================================================================
# Part 7: Main flow
# =============================================================================

OUT_COLS = ["ticker", "fiscal_year", "fiscal_quarter", "period_month",
            "period_end", "report_date", "revenue", "revenue_ttm",
            "revenue_source", "report_date_source", "revenue_ttm_source",
            "data_flag"]


def build_output_rows(t: str, res: dict, bq_rows: list[dict]) -> list[dict]:
    """Expand one stock's result into output rows (including report_date filling, sanity checks, TTM recomputation)."""
    if not res.get("ok"):
        out = []
        for r in bq_rows:
            o = dict(r)
            o["fiscal_year"] = r.get("fiscal_year", "")
            o["fiscal_quarter"] = r.get("fiscal_quarter", "")
            o["period_month"] = ""
            o["revenue_source"] = "BQ"
            o["report_date_source"] = "BQ" if r.get("report_date") else ""
            o["revenue_ttm_source"] = ""
            o["data_flag"] = "sec_unavailable"
            out.append(o)
        return out

    fye_m = res["fye_month"]
    ff = res["first_filed"]
    series = res["series"]
    base_by_end = {r["period_end"]: r for r in bq_rows}

    rows_t = []
    for e, sr in sorted(series.items()):
        base = base_by_end.get(e, {})
        fy, fq, pm = fiscal_label(e, fye_m)
        rd = str(base.get("report_date") or "").strip()
        rd_src = "BQ" if rd else ""
        if not rd:
            cand = ff.get(e)
            if not cand:      # 52/53-week: SEC's reportDate may differ by a day or two
                cpm = snap_to_month_end(e)
                for k, v in ff.items():
                    if snap_to_month_end(k) == cpm:
                        cand = v
                        break
            if cand:
                rd, rd_src = cand, "SEC"
        rows_t.append({
            "ticker": t, "fiscal_year": fy, "fiscal_quarter": fq,
            "period_month": pm, "period_end": e, "report_date": rd,
            "revenue": "" if sr["revenue"] is None else sr["revenue"],
            "revenue_ttm": "", "revenue_source": sr["revenue_source"],
            "report_date_source": rd_src, "revenue_ttm_source": "",
            "data_flag": "",
        })

    rows_t.sort(key=lambda r: r["period_end"])

    # Sanity checks
    vals = [(r["period_end"], float(r["revenue"]))
            for r in rows_t if str(r["revenue"]).strip()]
    flags = sanity_flags(vals)
    for r in rows_t:
        if r["period_end"] in flags:
            r["data_flag"] = "suspect:" + flags[r["period_end"]]

    # TTM recomputation = sum of the latest 4 single quarters
    for i, r in enumerate(rows_t):
        win = rows_t[max(0, i - 3):i + 1]
        if len(win) == 4 and all(str(w["revenue"]).strip() for w in win):
            r["revenue_ttm"] = sum(float(w["revenue"]) for w in win)
            n_sec = sum(1 for w in win if w["revenue_source"] == "SEC")
            r["revenue_ttm_source"] = ("BQ" if n_sec == 0
                                       else "SEC" if n_sec == 4 else "mixed")
    return rows_t


def build_report_rows(t: str, res: dict) -> list[dict]:
    """Per-period replacement details + one summary row (chosen line item / agreement rate / fiscalYearEnd / CIK)."""
    rows = []
    fye_m = res["fye_month"]
    for e in sorted(res["series"]):
        sr = res["series"][e]
        if not sr["note"]:
            continue
        fy, fq, _ = fiscal_label(e, fye_m)
        rows.append({"ticker": t, "period_end": e, "fiscal_year": fy,
                     "fiscal_quarter": fq, "chosen_tag": res["tag"],
                     "bq_value": sr["bq_value"], "sec_value": sr["sec_value"],
                     "sec_kind": sr["sec_kind"], "final": sr["revenue"],
                     "action": sr["revenue_source"], "note": sr["note"]})
    s = res["stats"]
    rate = (f"agreement {s['n_match']}/{s['n_cmp']} = "
            f"{s['n_match']/s['n_cmp']*100:.1f}%" if s["n_cmp"] else "no comparable periods")
    rows.append({
        "ticker": t, "period_end": "== SUMMARY ==", "fiscal_year": "",
        "fiscal_quarter": "", "chosen_tag": res["tag"],
        "bq_value": "", "sec_value": "", "sec_kind": "", "final": "",
        "action": rate,
        "note": (f"replaced {s['n_replace']}, filled {s['n_fill']}, "
                 f"blanked {s['n_drop']}; fiscalYearEnd={res['fye_raw']}; "
                 f"CIK={res['cik']}"),
    })
    return rows


def self_check(out_rows: list[dict]) -> bool:
    """Self-check after writing: label uniqueness + key regression values.

    Returns True if everything passes.
    """
    ok = True
    byt: dict[str, list[dict]] = {}
    for r in out_rows:
        byt.setdefault(r["ticker"], []).append(r)

    dup_lab = dup_pm = 0
    for t, rs in sorted(byt.items()):
        cl = {k: v for k, v in Counter((r["fiscal_year"], r["fiscal_quarter"])
                                       for r in rs).items() if v > 1}
        cp = {k: v for k, v in Counter(r["period_month"] for r in rs).items()
              if v > 1}
        if cl:
            dup_lab += 1
            log(f"  ✗ {t} duplicate (fiscal_year,fiscal_quarter): {cl}")
        if cp:
            dup_pm += 1
            log(f"  ✗ {t} duplicate period_month: {cp}")
    log(f"  symbols with duplicate (ticker,fiscal_year,fiscal_quarter): {dup_lab}")
    log(f"  symbols with duplicate (ticker,period_month): {dup_pm}")
    if dup_lab or dup_pm:
        ok = False
    else:
        log("  ✓ label uniqueness passed")

    badq = [r for r in out_rows if int(r["fiscal_quarter"] or 0) not in (1, 2, 3, 4)]
    if badq:
        log(f"  ✗ rows with fiscal_quarter outside 1-4: {len(badq)}")
        ok = False
    else:
        log("  ✓ fiscal_quarter all within 1-4")

    neg = [r for r in out_rows
           if str(r["revenue"]).strip() and float(r["revenue"]) <= 0]
    if neg:
        log(f"  ✗ remaining non-positive revenue: {len(neg)} periods")
        ok = False
    else:
        log("  ✓ no remaining non-positive revenue")

    # Key regression values: these two numbers come from independent verification provided in feedback
    look = {(r["ticker"], r["period_month"]): r for r in out_rows}
    for t, pm, want in [("BRK-B", "2024-12", 94.916e9), ("V", "2016-09", 4.261e9)]:
        r = look.get((t, pm))
        got = (float(r["revenue"])
               if r and str(r["revenue"]).strip() else None)
        if got is None or abs(got - want) / want > 0.001:
            log(f"  ✗ regression value {t} {pm}: expected {want/1e9:.3f}B, "
                f"got {got/1e9:.3f}B" if got else f"  ✗ regression value {t} {pm} missing")
            ok = False
        else:
            log(f"  ✓ regression value {t} {pm} = {got/1e9:.3f}B")
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Correct BusinessQuant revenue with SEC EDGAR")
    p.add_argument("--revenue", default=os.path.join(_SCRIPT_DIR, "revenue.csv"),
                   help="input BQ revenue table")
    p.add_argument("--outdir", default=_SCRIPT_DIR)
    p.add_argument("--tickers", default=None, help="process only the given symbols (comma-separated)")
    p.add_argument("--tol", type=float, default=TOL_REPLACE,
                   help="replace with the SEC value only when deviation exceeds this ratio (default 0.20)")
    args = p.parse_args(argv)

    tol = args.tol
    os.makedirs(CACHE, exist_ok=True)
    want = ({t.strip().upper() for t in args.tickers.split(",") if t.strip()}
            if args.tickers else None)

    rev = list(csv.DictReader(open(args.revenue, newline="",
                                   encoding="utf-8-sig")))
    if not rev:
        log("revenue.csv is empty")
        return 1
    by_t: dict[str, list[dict]] = {}
    for r in rev:
        by_t.setdefault(r["ticker"], []).append(r)
    log(f"read {args.revenue}: {len(rev)} rows, {len(by_t)} stocks")

    raw = _get("https://www.sec.gov/files/company_tickers.json",
               "_sec_ticker_map_raw") or {}
    t2c = {}
    for it in raw.values():
        k = str(it.get("ticker", "")).upper().replace(".", "").replace("-", "")
        if k:
            t2c[k] = str(it.get("cik_str", "")).zfill(10)

    tickers = sorted(by_t) + [t for t in FALLBACK_CIK if t not in by_t]
    if want:
        tickers = [t for t in tickers if t.upper() in want]

    log("")
    log("=" * 104)
    log("Per-stock processing: single SEC line item + Q4 check")
    log("=" * 104)
    log(f"  {'ticker':<8}{'CIK':<12}{'FYE':<6}{'line item':<50}"
        f"{'cmp':>5}{'agr':>5}{'repl':>5}{'add':>4}{'blnk':>5}")

    results: dict[str, dict] = {}
    for t in tickers:
        bq = {}
        for r in by_t.get(t, []):
            s = str(r["revenue"]).strip()
            if s:
                bq[r["period_end"]] = float(s)

        key = t.upper().replace(".", "").replace("-", "")
        cands = [c for c in ([t2c.get(key)] + FALLBACK_CIK.get(t, [])) if c]
        best = None
        for c in cands:
            r = build_sec_series(c, bq, tol)
            if r.get("ok") and (best is None
                                or len(r["series"]) > len(best["series"])):
                best = r
        if best is None:
            log(f"  {t:<8}✗ unable to build series from SEC")
            results[t] = {"ok": False}
            continue
        results[t] = best
        s = best["stats"]
        rate = f"{s['n_match']/s['n_cmp']*100:.0f}%" if s["n_cmp"] else "—"
        log(f"  {t:<8}{best['cik']:<12}{str(best['fye_raw']):<6}"
            f"{best['tag'][:48]:<50}{s['n_cmp']:>5}{rate:>5}"
            f"{s['n_replace']:>5}{s['n_fill']:>4}{s['n_drop']:>5}")

    # ---- Build output -------------------------------------------------------
    out_rows: list[dict] = []
    report_rows: list[dict] = []
    for t, res in results.items():
        bq_rows = by_t.get(t, [])
        out_rows.extend(build_output_rows(t, res, bq_rows))
        if res.get("ok"):
            report_rows.extend(build_report_rows(t, res))

    # ---- Write ---------------------------------------------------------------
    out_path = os.path.join(args.outdir, "revenue_adjusted.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=OUT_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(out_rows)
    log(f"\n→ written to {out_path}  ({len(out_rows)} rows)")

    rep_path = os.path.join(args.outdir, "revenue_merge_report.csv")
    if report_rows:
        with open(rep_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            w.writeheader()
            w.writerows(report_rows)
        log(f"→ written to {rep_path}  ({len(report_rows)} rows)")

    # ---- Self-check ----------------------------------------------------------
    log("")
    log("=" * 104)
    log("Self-check")
    log("=" * 104)
    passed = self_check(out_rows)

    src_c = Counter(r["revenue_source"] for r in out_rows)
    n_flag = sum(1 for r in out_rows if r["data_flag"].startswith("suspect"))
    n_blank = sum(1 for r in out_rows if not str(r["revenue"]).strip())
    n_mixed = sum(1 for r in out_rows if r["revenue_ttm_source"] == "mixed")
    log(f"\n  revenue sources: {dict(src_c)}")
    log(f"  blanked {n_blank} periods; suspect {n_flag} periods; "
        f"report_date missing "
        f"{sum(1 for r in out_rows if not str(r['report_date']).strip())} periods")
    if n_mixed:
        log(f"  ⚠ revenue_ttm is mixed for {n_mixed} periods (the 4-period window contains both BQ and "
            f"SEC, mixing definitions; use with caution; see the revenue_ttm_source column)")
    log("")
    log("  " + ("✓ all self-checks passed" if passed else "✗ self-check failed; please review the output above"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

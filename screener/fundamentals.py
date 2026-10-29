"""Reported financials for the whole universe from SEC XBRL frames, no model calls.

One frames request (data.sec.gov/api/xbrl/frames/<taxonomy>/<tag>/<unit>/<period>.json) returns
every filer's value for one line item and one calendar period, so ~250 requests cover 1,500
companies and 11 years. Each response is cut down to the universe and cached in
data/screener/frames/. SEC maps each filer's fiscal period to the calendar period it fits best,
so "2025" is a company's fiscal year that falls mostly in 2025.

Derived per company: margins, returns, leverage, growth, Piotroski F-score, working-capital days
(bargaining-power proxies) and revenue elasticity to industrial production (cyclicality).
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
from datetime import date
from pathlib import Path
from statistics import median, pstdev
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from core.tools.sec import _get as sec_get

logger = logging.getLogger("uvicorn.error")

ROOT = Path(__file__).resolve().parent.parent
FRAMES_DIR = ROOT / "data" / "screener" / "frames"

FIRST_YEAR = 2015
TAX_RATE = 0.21  # for after-tax operating return on invested capital

# Annual flows. Several tags mean the same line item; the first one a company reports wins,
# except revenue, where the largest wins (the contract-revenue tag can leave out lease or other revenue).
FLOW_TAGS: Dict[str, List[str]] = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax",
                "RevenuesNetOfInterestExpense",  # banks
                "RegulatedAndUnregulatedOperatingRevenue"],  # utilities
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "interest_expense": ["InterestExpense", "InterestExpenseNonoperating", "InterestExpenseDebt", "InterestAndDebtExpense",
                         "InterestPaidNet"],  # last resort: cash interest paid
    "dna": ["DepreciationDepletionAndAmortization", "DepreciationAndAmortization", "DepreciationAmortizationAndAccretionNet"],
    "dividends": ["PaymentsOfDividendsCommonStock", "PaymentsOfDividends"],
    "buybacks": ["PaymentsForRepurchaseOfCommonStock"],
    # banks that report no revenue total: revenue = net interest income + noninterest income
    "net_interest_income": ["InterestIncomeExpenseNet"],
    "noninterest_income": ["NoninterestIncome"],
}
SHARE_FLOW_TAGS = {"diluted_shares": ["WeightedAverageNumberOfDilutedSharesOutstanding"]}
# Balances at (calendar) year end.
INSTANT_TAGS: Dict[str, List[str]] = {
    "assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermNotesAndLoans", "LongTermNotesPayable",
                       "LongTermDebtAndCapitalLeaseObligations"],
    "debt_current": ["LongTermDebtCurrent", "DebtCurrent"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue", "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "receivables": ["AccountsReceivableNetCurrent"],
    "inventory": ["InventoryNet"],
    "payables": ["AccountsPayableCurrent"],
    # contracted revenue not yet recognized (backlog) and cash collected ahead of delivery: revenue quality
    "rpo": ["RevenueRemainingPerformanceObligation"],
    "deferred_revenue": ["ContractWithCustomerLiabilityCurrent", "DeferredRevenueCurrent"],
}
LATEST_INSTANT_TAGS = ["equity", "long_term_debt", "debt_current", "cash"]  # for market cap -> EV
QUARTER_FLOWS = ["revenue", "net_income"]  # latest-quarter growth and trailing twelve months


def _quarters_back(today: date, n: int) -> List[str]:
    """The n most recent calendar quarters whose 10-Qs are due (45 days after quarter end), newest first."""
    y, q = today.year, (today.month - 1) // 3  # q = last fully finished quarter (0 means Q4 of last year)
    out = []
    while len(out) < n + 1:
        if q == 0:
            y, q = y - 1, 4
        out.append((y, q))
        q -= 1
    end_month = {1: 3, 2: 6, 3: 9, 4: 12}
    due = [(yy, qq) for yy, qq in out if (today - date(yy, end_month[qq], 28)).days > 45]
    return [f"CY{yy}Q{qq}" for yy, qq in due[:n]]


def _frame(taxonomy: str, tag: str, unit: str, period: str, ciks: set) -> Dict[int, dict]:
    """{cik: {"val", "end", "start"?}} for one tag and period, cached (the past rarely changes)."""
    path = FRAMES_DIR / f"{taxonomy}_{tag}_{unit.replace('/', '-')}_{period}.json"
    current_year = date.today().year
    year = int(period[2:6])
    max_age_days = 30 if year < current_year - 1 else 3
    if path.exists() and time.time() - path.stat().st_mtime < max_age_days * 86400:
        rows = json.loads(path.read_text())
    else:
        url = f"https://data.sec.gov/api/xbrl/frames/{taxonomy}/{tag}/{unit}/{period}.json"
        try:
            data = json.loads(sec_get(url))["data"]
        except urllib.error.HTTPError as exc:
            if exc.code != 404:  # 404: nobody reported this tag for this period
                raise
            data = []
        rows = {str(d["cik"]): {k: d[k] for k in ("val", "end", "start") if k in d} for d in data}
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows))
    return {int(c): v for c, v in rows.items() if int(c) in ciks}


def _pick(frames: List[Dict[int, dict]], cik: int, largest: bool = False) -> Optional[dict]:
    rows = [f[cik] for f in frames if cik in f]
    if not rows:
        return None
    return max(rows, key=lambda r: r["val"]) if largest else rows[0]


def fetch_reported(ciks: Iterable[int], today: Optional[date] = None, progress=None) -> Dict[int, Dict[str, Any]]:
    """{cik: {"annual": {year: {item: value}}, "quarters": {period: {item: value}}, "latest": {...}, "shares": ...}}"""
    today = today or date.today()
    ciks = set(ciks)
    last_year = today.year - 1  # the last full calendar year (10-Ks for it are in by March)
    years = list(range(FIRST_YEAR, last_year + 1))
    quarters = _quarters_back(today, 6)  # newest first
    out: Dict[int, Dict[str, Any]] = {c: {"annual": {y: {} for y in years}, "quarters": {q: {} for q in quarters},
                                          "latest": {}} for c in ciks}
    jobs = []
    for item, tags in {**FLOW_TAGS, **SHARE_FLOW_TAGS}.items():
        unit = "shares" if item in SHARE_FLOW_TAGS else "USD"
        for y in years:
            jobs.append(("annual", item, tags, unit, y, f"CY{y}"))
        if item in QUARTER_FLOWS:
            for q in quarters:
                jobs.append(("quarter", item, tags, unit, q, q))
    for item, tags in INSTANT_TAGS.items():
        for y in years[-4:]:  # balance-sheet ratios need the last few year ends only
            jobs.append(("annual", item, tags, "USD", y, f"CY{y}Q4I"))
        if item in LATEST_INSTANT_TAGS:
            jobs.append(("latest", item, tags, "USD", None, f"{quarters[0]}I"))
    total = sum(len(j[2]) for j in jobs) + 2
    done = 0
    for kind, item, tags, unit, key, period in jobs:
        frames = []
        for tag in tags:
            frames.append(_frame("us-gaap", tag, unit, period, ciks))
            done += 1
            if progress and done % 25 == 0:
                progress(done, total)
        for cik in ciks:
            row = _pick(frames, cik, largest=(item == "revenue"))
            if row is None:
                continue
            v = row["val"]
            if kind == "annual":
                out[cik]["annual"][key][item] = v
                if item == "revenue":
                    out[cik]["annual"][key]["period_end"] = row["end"]
            elif kind == "quarter":
                out[cik]["quarters"][key][item] = v
            else:
                out[cik]["latest"][item] = v
    for c in out.values():
        for y in c["annual"].values():
            if not y.get("revenue") and y.get("net_interest_income") is not None and y.get("noninterest_income") is not None:
                y["revenue"] = y["net_interest_income"] + y["noninterest_income"]
    # Cover-page share count (dei), newest quarter that has it
    for q in quarters[:2]:
        f = _frame("dei", "EntityCommonStockSharesOutstanding", "shares", f"{q}I", ciks)
        for cik, row in f.items():
            out[cik].setdefault("cover_shares", (row["val"], row["end"]))
    return out


# ---- derived metrics ----

def _div(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None or b == 0:
        return None
    return a / b


def _cagr(a: Optional[float], b: Optional[float], years: int) -> Optional[float]:
    if not a or not b or a <= 0 or b <= 0:
        return None
    return (b / a) ** (1 / years) - 1


def _gross_profit(y: Dict[str, float]) -> Optional[float]:
    if y.get("gross_profit") is not None:
        return y["gross_profit"]
    if y.get("revenue") and y.get("cost_of_revenue") is not None:
        return y["revenue"] - y["cost_of_revenue"]
    return None


def piotroski(cur: Dict[str, float], prev: Dict[str, float]) -> Optional[int]:
    """Piotroski F-score (0-9) from two fiscal years; None if fewer than 6 of the 9 tests can be run.
    A score from fewer tests is scaled to 9."""
    tests: List[bool] = []

    def add(cond: Optional[bool]) -> None:
        if cond is not None:
            tests.append(bool(cond))

    roa, roa_prev = _div(cur.get("net_income"), prev.get("assets")), _div(prev.get("net_income"), prev.get("assets"))
    add(None if roa is None else roa > 0)
    add(None if cur.get("cfo") is None else cur["cfo"] > 0)
    add(None if roa is None or roa_prev is None else roa > roa_prev)
    add(None if cur.get("cfo") is None or cur.get("net_income") is None else cur["cfo"] > cur["net_income"])
    lev, lev_prev = _div(cur.get("long_term_debt", 0), cur.get("assets")), _div(prev.get("long_term_debt", 0), prev.get("assets"))
    add(None if lev is None or lev_prev is None else lev <= lev_prev)
    cr, cr_prev = _div(cur.get("current_assets"), cur.get("current_liabilities")), _div(prev.get("current_assets"), prev.get("current_liabilities"))
    add(None if cr is None or cr_prev is None else cr > cr_prev)
    sh, sh_prev = cur.get("diluted_shares"), prev.get("diluted_shares")
    add(None if not sh or not sh_prev else sh <= sh_prev * 1.01)
    gm, gm_prev = _div(_gross_profit(cur), cur.get("revenue")), _div(_gross_profit(prev), prev.get("revenue"))
    add(None if gm is None or gm_prev is None else gm > gm_prev)
    at, at_prev = _div(cur.get("revenue"), cur.get("assets")), _div(prev.get("revenue"), prev.get("assets"))
    add(None if at is None or at_prev is None else at > at_prev)
    if len(tests) < 6:
        return None
    return round(sum(tests) * 9 / len(tests))


def elasticity(series: Dict[int, float], driver: Dict[int, float]) -> Optional[float]:
    """Slope of a company's annual revenue growth on the driver's annual growth (needs 6+ years)."""
    xs, ys = [], []
    for y in sorted(series):
        if y - 1 in series and y in driver and y - 1 in driver and series[y - 1] > 0 and series[y] > 0:
            xs.append(driver[y] / driver[y - 1] - 1)
            ys.append(series[y] / series[y - 1] - 1)
    if len(xs) < 6:
        return None
    x, yv = np.array(xs), np.array(ys)
    yv = np.clip(yv, -0.6, 1.5)  # one acquisition year should not dominate the slope
    var = float(np.var(x))
    return float(np.cov(x, yv, bias=True)[0, 1] / var) if var > 0 else None


def derive(raw: Dict[str, Any], driver_growth: Dict[int, float]) -> Dict[str, Any]:
    """Per-company metrics from fetch_reported output. Values are fractions (0.25 = 25%)."""
    annual = {y: v for y, v in raw["annual"].items() if v.get("revenue") or v.get("net_income") is not None}
    if not annual:
        return {}
    years = sorted(annual)
    fy = years[-1]
    cur, prev = annual[fy], annual.get(fy - 1, {})
    m: Dict[str, Any] = {"fiscal_year": fy}
    rev = cur.get("revenue")
    gp = _gross_profit(cur)
    ebit = cur.get("operating_income")
    m.update({
        "revenue": rev, "net_income": cur.get("net_income"), "operating_income": ebit,
        "gross_margin": _div(gp, rev), "operating_margin": _div(ebit, rev), "net_margin": _div(cur.get("net_income"), rev),
        "rnd_intensity": _div(cur.get("rnd"), rev),
    })
    fcf = cur["cfo"] - (cur.get("capex") or 0) if cur.get("cfo") is not None else None
    m["fcf"] = fcf
    m["cfo"], m["capex"], m["dna"] = cur.get("cfo"), cur.get("capex"), cur.get("dna")
    m["fcf_margin"] = _div(fcf, rev)
    ebitda = ebit + cur["dna"] if ebit is not None and cur.get("dna") is not None else None
    m["ebitda"] = ebitda
    debt = (cur.get("long_term_debt") or 0) + (cur.get("debt_current") or 0)
    net_debt = debt - (cur.get("cash") or 0) if cur.get("assets") is not None else None
    m["net_debt"] = net_debt
    m["net_debt_to_ebitda"] = _div(net_debt, ebitda) if ebitda and ebitda > 0 else None
    m["interest_coverage"] = _div(ebit, cur.get("interest_expense")) if cur.get("interest_expense") else None
    m["interest_expense"] = cur.get("interest_expense")
    m["debt"] = debt
    invested = (cur.get("equity") or 0) + debt - (cur.get("cash") or 0)
    m["roic"] = _div(ebit * (1 - TAX_RATE), invested) if ebit is not None and invested > 0 and cur.get("equity") is not None else None
    m["roe"] = _div(cur.get("net_income"), cur.get("equity")) if (cur.get("equity") or 0) > 0 else None
    m["roa"] = _div(cur.get("net_income"), cur.get("assets"))
    m["equity_to_assets"] = _div(cur.get("equity"), cur.get("assets"))
    m["net_income_stability"] = None
    nis = [annual[y].get("net_income") for y in years[-5:] if annual[y].get("net_income") is not None]
    if len(nis) >= 4 and sum(nis) > 0:  # fewer loss years and smaller swings -> steadier earnings
        m["net_income_stability"] = pstdev(nis) / abs(sum(nis) / len(nis))
    m["piotroski"] = piotroski(cur, prev) if prev else None
    # growth
    m["revenue_cagr_3y"] = _cagr(annual.get(fy - 3, {}).get("revenue"), rev, 3)
    m["revenue_cagr_5y"] = _cagr(annual.get(fy - 5, {}).get("revenue"), rev, 5)
    m["net_income_cagr_3y"] = _cagr(annual.get(fy - 3, {}).get("net_income"), cur.get("net_income"), 3)
    qs = raw.get("quarters", {})
    qkeys = sorted(qs, reverse=True)
    if qkeys:
        q0 = qkeys[0]
        y0, qn = int(q0[2:6]), q0[6:]
        a, b = qs.get(f"CY{y0 - 1}{qn}", {}).get("revenue"), qs[q0].get("revenue")
        m["revenue_growth_latest_q"] = _div(b, a) - 1 if a and b and a > 0 else None
        m["latest_quarter"] = q0
        # trailing twelve months: last fiscal year + this year's quarters - last year's same quarters
        this = [k for k in qkeys if k.startswith(f"CY{fy + 1}")]
        december_fy = str(cur.get("period_end", "")).endswith("-12-31")  # otherwise calendar quarters don't line up
        if december_fy and this and all(f"CY{fy}{k[6:]}" in qs for k in this):
            for item in ("revenue", "net_income"):
                vals = [qs[k].get(item) for k in this] + [qs[f"CY{fy}{k[6:]}"].get(item) for k in this]
                if cur.get(item) is not None and None not in vals:
                    n = len(this)
                    m[f"{item}_ttm"] = cur[item] + sum(vals[:n]) - sum(vals[n:])
    m.setdefault("revenue_ttm", rev)
    m.setdefault("net_income_ttm", cur.get("net_income"))
    # bargaining-power proxies
    cogs = cur.get("cost_of_revenue") or (rev - gp if rev is not None and gp is not None else None)
    m["dso"] = _div(cur.get("receivables"), rev) * 365 if _div(cur.get("receivables"), rev) is not None else None
    m["dpo"] = _div(cur.get("payables"), cogs) * 365 if _div(cur.get("payables"), cogs) is not None else None
    m["dio"] = _div(cur.get("inventory"), cogs) * 365 if _div(cur.get("inventory"), cogs) is not None else None
    gms = [_div(_gross_profit(annual[y]), annual[y].get("revenue")) for y in years[-5:]]
    gms = [g for g in gms if g is not None and -1 < g < 1]
    m["gross_margin_stability"] = pstdev(gms) if len(gms) >= 3 else None
    m["gross_margin_5y_median"] = median(gms) if gms else None
    # revenue quality: backlog coverage and growth, how steady revenue has been
    m["rpo"] = cur.get("rpo")
    m["rpo_coverage"] = _div(cur.get("rpo"), rev) if rev and rev > 0 else None
    m["rpo_growth"] = _div(cur.get("rpo"), prev.get("rpo")) - 1 if cur.get("rpo") and (prev.get("rpo") or 0) > 0 else None
    m["deferred_revenue_ratio"] = _div(cur.get("deferred_revenue"), rev) if rev and rev > 0 else None
    revs = [annual[y].get("revenue") for y in years[-6:]]
    growths = [b / a - 1 for a, b in zip(revs, revs[1:]) if a and b and a > 0]
    m["revenue_growth_volatility"] = pstdev(growths) if len(growths) >= 3 else None
    m["revenue_down_years"] = sum(g < 0 for g in growths) if len(growths) >= 3 else None
    # cyclicality: revenue growth vs industrial production growth
    m["revenue_elasticity"] = elasticity({y: annual[y]["revenue"] for y in years if annual[y].get("revenue")}, driver_growth)
    # for market cap and EV
    latest = raw.get("latest", {})
    m["latest_debt"] = (latest.get("long_term_debt") or 0) + (latest.get("debt_current") or 0) if latest else debt
    m["latest_cash"] = latest.get("cash", cur.get("cash"))
    m["latest_equity"] = latest.get("equity", cur.get("equity"))
    m["dividends"] = cur.get("dividends")
    m["buybacks"] = cur.get("buybacks")
    if raw.get("cover_shares"):
        m["shares"], m["shares_as_of"] = raw["cover_shares"]
    elif cur.get("diluted_shares"):
        m["shares"], m["shares_as_of"] = cur["diluted_shares"], f"FY{fy} diluted average"
    m["revenue_history"] = {y: annual[y]["revenue"] for y in years if annual[y].get("revenue")}
    return m

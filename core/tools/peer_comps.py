"""Peer comparison computed from SEC XBRL facts and closing prices, no model calls.

For each ticker: trailing-twelve-month (TTM) revenue, margins, EBITDA, free cash flow, latest
balance sheet, and market multiples at the latest close. TTM = last fiscal year + current
year-to-date - the same year-to-date a year earlier. EBITDA here is GAAP operating income plus
depreciation and amortization; "EBITDA + SBC" adds stock compensation back, which is close to
most companies' adjusted EBITDA. Company-defined adjusted figures differ, so label them as ours. FCF is operating cash flow minus
the capex tags present; company-defined FCF can also subtract items tagged with company-specific
names (e.g. Wayfair's site and software development costs), which the XBRL API leaves out.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from core.tools import prices
from core.tools.sec import company_facts, resolve_company

logger = logging.getLogger("uvicorn.error")

FLOWS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "d_and_a": ["DepreciationDepletionAndAmortization", "DepreciationAmortizationAndAccretionNet",
                "DepreciationAndAmortization", "Depreciation"],
    "sbc": ["ShareBasedCompensation", "AllocatedShareBasedCompensationExpense"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
}
# Capex: the first current property-and-equipment tag, plus any capitalized-software tags.
CAPEX_MAIN = ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"]
CAPEX_EXTRA = ["PaymentsToDevelopSoftware", "PaymentsForSoftware"]
INSTANTS = {
    "cash": ["CashAndCashEquivalentsAtCarryingValue"],
    "short_term_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent",
                               "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "debt_total": ["LongTermDebt", "DebtInstrumentCarryingAmount"],
    "debt_noncurrent": ["LongTermDebtNoncurrent", "ConvertibleNotesPayableNoncurrent", "ConvertibleDebtNoncurrent"],
    "debt_current": ["LongTermDebtCurrent", "ConvertibleNotesPayableCurrent", "ConvertibleDebtCurrent"],
}


def _entries(gaap: Dict[str, Any], tag: str) -> List[dict]:
    """One entry per period (latest filing wins), from 10-K and 10-Q filings only."""
    vals = gaap.get(tag, {}).get("units", {}).get("USD", [])
    best: Dict[Tuple[Optional[str], str], dict] = {}
    for v in vals:
        if v.get("form") not in ("10-K", "10-Q", "10-K/A", "10-Q/A"):
            continue
        key = (v.get("start"), v["end"])
        if key not in best or v.get("filed", "") > best[key].get("filed", ""):
            best[key] = v
    return list(best.values())


def _first_tag(gaap: Dict[str, Any], tags: List[str]) -> List[dict]:
    """Entries of the tag with the most recent data (companies switch tags over time)."""
    options = [e for e in (_entries(gaap, t) for t in tags) if e]
    return max(options, key=lambda es: max(e["end"] for e in es)) if options else []


def _d(s: str) -> date:
    return date.fromisoformat(s)


def _days(e: dict) -> int:
    return (_d(e["end"]) - _d(e["start"])).days if e.get("start") else 0


def _near(a: date, b: date, tol: int = 12) -> bool:
    return abs((a - b).days) <= tol


def ttm(entries: List[dict]) -> Optional[Tuple[float, str]]:
    """(TTM value, period end) from annual and year-to-date entries."""
    annual = [e for e in entries if 330 <= _days(e) <= 380]
    if not annual:
        return None
    a = max(annual, key=lambda e: e["end"])
    ytd = [e for e in entries if e.get("start") and 60 <= _days(e) < 330 and e["end"] > a["end"]
           and _near(_d(e["start"]), _d(a["end"]) + timedelta(days=1))]
    if not ytd:
        return float(a["val"]), a["end"]
    y = max(ytd, key=lambda e: e["end"])
    prior = [e for e in entries if e.get("start") and _near(_d(e["end"]), _d(y["end"]) - timedelta(days=364))
             and abs(_days(e) - _days(y)) <= 12]
    if not prior:
        return float(a["val"]), a["end"]
    return float(a["val"]) + float(y["val"]) - float(prior[0]["val"]), y["end"]


def latest_quarter_growth(entries: List[dict]) -> Optional[Tuple[float, str]]:
    """Year-on-year growth of the latest three-month period, or of the latest fiscal year."""
    for lo, hi in ((80, 100), (330, 380)):
        periods = [e for e in entries if lo <= _days(e) <= hi]
        if not periods:
            continue
        last = max(periods, key=lambda e: e["end"])
        before = [e for e in periods if _near(_d(e["end"]), _d(last["end"]) - timedelta(days=364))]
        if before and before[0]["val"]:
            return float(last["val"]) / float(before[0]["val"]) - 1, last["end"]
    return None


def _instant(gaap: Dict[str, Any], tags: List[str], as_of: date) -> Optional[Tuple[float, str]]:
    entries = [e for e in _first_tag(gaap, tags) if not e.get("start")]
    if not entries:
        return None
    e = max(entries, key=lambda e: e["end"])
    return (float(e["val"]), e["end"]) if (as_of - _d(e["end"])).days <= 200 else None


def _shares(facts: Dict[str, Any], as_of: date) -> Optional[Tuple[float, str, str]]:
    """(shares, date, basis): the cover-page count if recent, else the latest quarter's diluted average.

    Companies with several share classes often tag the cover count by class only, which the
    company-facts API leaves out, so the diluted weighted average is the fallback."""
    rows = facts.get("dei", {}).get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    if rows:
        end = max(r["end"] for r in rows)
        if (as_of - _d(end)).days <= 200:
            filed = max(r.get("filed", "") for r in rows if r["end"] == end)
            total = sum(r["val"] for r in rows if r["end"] == end and r.get("filed", "") == filed)
            return float(total), end, "cover page"
    gaap = facts.get("us-gaap", {})
    for tag in ("WeightedAverageNumberOfDilutedSharesOutstanding", "WeightedAverageNumberOfSharesOutstandingBasic"):
        quarters = [v for v in gaap.get(tag, {}).get("units", {}).get("shares", [])
                    if v.get("start") and 80 <= _days(v) <= 100 and (as_of - _d(v["end"])).days <= 200]
        if quarters:
            q = max(quarters, key=lambda v: (v["end"], v.get("filed", "")))
            return float(q["val"]), q["end"], "diluted weighted average, latest quarter"
    return None


def company_comps(ticker: str) -> Dict[str, Any]:
    """One peer-table row. Money in USD millions; ratios as decimals; missing items are None."""
    company = resolve_company(ticker, ticker)
    facts = company_facts(company["cik"])
    gaap = facts.get("us-gaap", {})
    row: Dict[str, Any] = {"ticker": company["ticker"], "company": company["company"]}

    flows = {k: ttm(_first_tag(gaap, tags)) for k, tags in FLOWS.items()}
    if flows.get("revenue"):  # drop line items a company stopped tagging (stale periods)
        rev_end = _d(flows["revenue"][1])
        flows = {k: v for k, v in flows.items() if v and (rev_end - _d(v[1])).days <= 100}
    m = lambda k: flows[k][0] / 1e6 if flows.get(k) else None
    revenue = m("revenue")
    row["ttm_period_end"] = flows["revenue"][1] if flows.get("revenue") else None
    row["revenue_ttm"] = revenue
    growth = latest_quarter_growth(_first_tag(gaap, FLOWS["revenue"]))
    row["revenue_growth_latest"] = growth[0] if growth else None
    row["revenue_growth_period_end"] = growth[1] if growth else None
    gross = m("gross_profit")
    if gross is None and revenue is not None and m("cost_of_revenue") is not None:
        gross = revenue - m("cost_of_revenue")
    op, da, sbc, ni, ocf = m("operating_income"), m("d_and_a"), m("sbc"), m("net_income"), m("operating_cash_flow")
    current = lambda t: t and row["ttm_period_end"] and (_d(row["ttm_period_end"]) - _d(t[1])).days <= 100
    main = next((t for t in (ttm(_entries(gaap, tag)) for tag in CAPEX_MAIN) if current(t)), None)
    capex_parts = ([main] if main else []) + [t for t in (ttm(_entries(gaap, tag)) for tag in CAPEX_EXTRA) if current(t)]
    capex = sum(v for v, _ in capex_parts) / 1e6 if capex_parts else None
    ebitda = op + da if op is not None and da is not None else None
    fcf = ocf - capex if ocf is not None and capex is not None else None
    ratio = lambda x: x / revenue if x is not None and revenue else None
    row.update({
        "gross_margin": ratio(gross), "operating_margin": ratio(op), "net_income_ttm": ni,
        "ebitda_ttm": ebitda, "ebitda_margin": ratio(ebitda),
        "ebitda_plus_sbc_ttm": ebitda + sbc if ebitda is not None and sbc is not None else None,
        "sbc_pct_revenue": ratio(sbc), "fcf_ttm": fcf, "fcf_margin": ratio(fcf),
    })
    row["ebitda_plus_sbc_margin"] = ratio(row["ebitda_plus_sbc_ttm"])

    today = date.today()
    inst = lambda k: _instant(gaap, INSTANTS[k], today)
    cash = sum(v for v, _ in filter(None, (inst("cash"), inst("short_term_investments")))) / 1e6
    total = inst("debt_total")
    parts = [p for p in (inst("debt_noncurrent"), inst("debt_current")) if p]
    debt = total[0] / 1e6 if total else (sum(v for v, _ in parts) / 1e6 if parts else 0.0)
    row["cash_and_st_investments"], row["debt"], row["net_debt"] = cash, debt, debt - cash
    row["balance_sheet_date"] = (inst("cash") or (None, None))[1]

    shares = _shares(facts, today)
    close = prices.latest(prices.history(company["ticker"]))
    row["shares_outstanding_m"] = shares[0] / 1e6 if shares else None
    row["shares_basis"] = shares[2] if shares else None
    row["price"], row["price_date"] = (close[1], close[0]) if close else (None, None)
    if shares and close:
        mcap = shares[0] / 1e6 * close[1]
        ev = mcap + row["net_debt"]
        row.update({
            "market_cap": mcap, "enterprise_value": ev,
            "ev_sales": ev / revenue if revenue else None,
            "ev_ebitda": ev / ebitda if ebitda and ebitda > 0 else None,
            "ev_ebitda_plus_sbc": ev / row["ebitda_plus_sbc_ttm"] if (row["ebitda_plus_sbc_ttm"] or 0) > 0 else None,
            "pe_ttm": mcap / ni if ni and ni > 0 else None,
            "fcf_yield": fcf / mcap if fcf is not None and mcap else None,
        })
    return row


def peer_table(tickers: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for t in dict.fromkeys(x.upper() for x in tickers):
        try:
            rows.append(company_comps(t))
        except Exception as exc:  # one bad peer must not sink the table
            logger.warning("Peer comps failed for %s: %s", t, exc)
            rows.append({"ticker": t, "error": f"{type(exc).__name__}: {exc}"})
    return rows


def _fmt(v: Any, kind: str) -> str:
    if v is None:
        return "n/a"
    if kind == "pct":
        return f"{v * 100:.1f}%"
    if kind == "x":
        return f"{v:.1f}x"
    if kind == "usd":
        return f"${v:,.2f}"
    return f"{v:,.0f}"


COLUMNS = [
    ("revenue_ttm", "Revenue TTM ($M)", "m"), ("revenue_growth_latest", "Revenue growth, latest qtr YoY", "pct"),
    ("gross_margin", "Gross margin", "pct"), ("operating_margin", "Operating margin", "pct"),
    ("ebitda_plus_sbc_margin", "EBITDA+SBC margin", "pct"), ("sbc_pct_revenue", "SBC % revenue", "pct"),
    ("fcf_ttm", "FCF TTM ($M)", "m"), ("fcf_margin", "FCF margin", "pct"), ("net_debt", "Net debt ($M)", "m"),
    ("price", "Price", "usd"), ("market_cap", "Market cap ($M)", "m"), ("enterprise_value", "EV ($M)", "m"),
    ("ev_sales", "EV/Sales", "x"), ("ev_ebitda", "EV/EBITDA", "x"), ("ev_ebitda_plus_sbc", "EV/(EBITDA+SBC)", "x"),
    ("pe_ttm", "P/E TTM", "x"), ("fcf_yield", "FCF yield", "pct"),
]


def render_table(rows: List[Dict[str, Any]]) -> str:
    """Markdown table, companies as columns, for the evidence block."""
    ok = [r for r in rows if "error" not in r]
    if not ok:
        return "(no peer data)"
    head = "| Metric | " + " | ".join(f"{r['company']} ({r['ticker']})" for r in ok) + " |"
    lines = [head, "|" + "---|" * (len(ok) + 1)]
    lines += [f"| {label} | " + " | ".join(_fmt(r.get(key), kind) for r in ok) + " |" for key, label, kind in COLUMNS]
    lines.append("| Shares basis | " + " | ".join(str(r.get("shares_basis")) for r in ok) + " |")
    lines.append("| TTM period end / price date | " + " | ".join(f"{r.get('ttm_period_end')} / {r.get('price_date')}" for r in ok) + " |")
    failed = [f"{r['ticker']} ({r['error']})" for r in rows if "error" in r]
    return "\n".join(lines) + (f"\n\nNo data for: {', '.join(failed)}" if failed else "")


def annual_history(ticker: str, years: int = 4) -> List[Dict[str, Any]]:
    """The last few fiscal years of one company's key lines (USD millions), oldest first."""
    gaap = company_facts(resolve_company(ticker, ticker)["cik"]).get("us-gaap", {})
    by_end: Dict[str, Dict[str, float]] = {}
    tags = {**FLOWS, "capex": CAPEX_MAIN[:1]}
    for key, options in tags.items():
        for e in _first_tag(gaap, options):
            if 330 <= _days(e) <= 380:
                by_end.setdefault(e["end"], {})[key] = float(e["val"]) / 1e6
    rows = []
    for end in sorted(by_end)[-years:]:
        v = by_end[end]
        rev = v.get("revenue")
        gross = v.get("gross_profit", rev - v["cost_of_revenue"] if rev is not None and "cost_of_revenue" in v else None)
        ebitda_sbc = (v["operating_income"] + v["d_and_a"] + v["sbc"]
                      if all(k in v for k in ("operating_income", "d_and_a", "sbc")) else None)
        fcf = v["operating_cash_flow"] - v["capex"] if "operating_cash_flow" in v and "capex" in v else None
        rows.append({"fiscal_year_end": end, "revenue": rev, "gross_profit": gross,
                     "operating_income": v.get("operating_income"), "net_income": v.get("net_income"),
                     "sbc": v.get("sbc"), "ebitda_plus_sbc": ebitda_sbc, "fcf": fcf})
    for prev, row in zip(rows, rows[1:]):
        if prev["revenue"] and row["revenue"]:
            row["revenue_growth"] = row["revenue"] / prev["revenue"] - 1
    return rows


def render_history(rows: List[Dict[str, Any]], ttm_row: Optional[Dict[str, Any]] = None) -> str:
    """Fiscal years as columns, plus the TTM column from company_comps when given."""
    if not rows:
        return "(no annual history)"
    cols = [r["fiscal_year_end"] for r in rows] + ([f"TTM to {ttm_row['ttm_period_end']}"] if ttm_row else [])
    get = lambda r, k: r.get(k)
    ttm_map = {"revenue": "revenue_ttm", "operating_income": None, "net_income": "net_income_ttm",
               "ebitda_plus_sbc": "ebitda_plus_sbc_ttm", "fcf": "fcf_ttm", "revenue_growth": None,
               "gross_profit": None, "sbc": None}
    lines = ["| USD millions | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for key, label, kind in [("revenue", "Revenue", "m"), ("revenue_growth", "Revenue growth", "pct"),
                             ("gross_profit", "Gross profit", "m"), ("operating_income", "Operating income", "m"),
                             ("net_income", "Net income", "m"), ("sbc", "Stock compensation", "m"),
                             ("ebitda_plus_sbc", "EBITDA + SBC (our calc.)", "m"), ("fcf", "FCF (OCF - PP&E capex)", "m")]:
        cells = [_fmt(get(r, key), kind) for r in rows]
        if ttm_row:
            cells.append(_fmt(ttm_row.get(ttm_map[key]), kind) if ttm_map[key] else "")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

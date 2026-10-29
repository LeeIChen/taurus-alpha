"""Build the screener's dataset: one row per company, one per GICS sub-industry. No model calls.

Labels computed here (the model adds market structure, bargaining power and TAM in profiles.py):
- size: market cap (large >= $20B, mid $5-20B, small < $5B; close to the S&P 500/400/600 cutoffs)
- style: growth vs value percentiles across the universe -> 成長 / 均衡 / 價值
- cycle: market beta, leading-index beta and revenue elasticity to industrial production -> 高 / 中 / 低
- rates: 10-year-yield beta (market held fixed) -> 升息受害 / 不敏感 / 升息受惠
- quality: sector-relative percentiles of returns, margins, cash flow, leverage and the Piotroski F-score
- hedge-fund questions (hedge.py): reverse DCF, ROIC vs WACC at today's 10-year yield, revenue quality,
  pricing-power trend, crowding and fundamental momentum

The dataset is saved to data/screener/dataset.json; build it with scripts/build_screener_data.py.
"""

from __future__ import annotations

import json
import logging
import math
import time
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Callable, Dict, List, Optional

from screener import fundamentals, macro, market
from screener.universe import DATA_DIR, load_universe

logger = logging.getLogger("uvicorn.error")

DATASET_PATH = DATA_DIR / "dataset.json"
LARGE_CAP, MID_CAP = 20e9, 5e9
# Cash flow, operating income and gross margin don't describe a bank or insurer's quality;
# financials are scored on returns, capital and earnings stability instead.
CASH_FLOW_SECTORS_EXCLUDED = {"Financials"}
STYLE_GAP = 0.15  # growth minus value percentile needed to call a company growth or value
RATE_T = 1.65  # |t| of the rate beta for a company to count as rate sensitive (90% two-sided)

SIZE_LABELS = ["大型", "中型", "小型"]
STYLE_LABELS = ["成長", "均衡", "價值"]
CYCLE_LABELS = ["高", "中", "低"]
RATE_LABELS = ["升息受害", "不敏感", "升息受惠"]


def percentiles(values: Dict[str, Optional[float]], higher_is_better: bool = True) -> Dict[str, float]:
    """{key: percentile in [0, 1]} among keys with a value (ties share the mean rank)."""
    items = sorted(((v if higher_is_better else -v), k) for k, v in values.items() if v is not None)
    n = len(items)
    if n == 0:
        return {}
    if n == 1:
        return {items[0][1]: 0.5}
    out, i = {}, 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][0] == items[i][0]:
            j += 1
        for k in range(i, j + 1):
            out[items[k][1]] = ((i + j) / 2) / (n - 1)
        i = j + 1
    return out


def _avg(parts: List[Optional[float]], min_parts: int = 1) -> Optional[float]:
    vals = [p for p in parts if p is not None]
    return sum(vals) / len(vals) if len(vals) >= min_parts else None


def _grouped_percentiles(rows: List[Dict[str, Any]], key: str, group: str, higher_is_better: bool = True) -> Dict[str, float]:
    groups: Dict[str, Dict[str, Optional[float]]] = {}
    for r in rows:
        groups.setdefault(r[group], {})[r["ticker"]] = r.get(key)
    out: Dict[str, float] = {}
    for vals in groups.values():
        out.update(percentiles(vals, higher_is_better))
    return out


def add_valuation(row: Dict[str, Any]) -> None:
    price, shares = row.get("price"), row.get("shares")
    if not price or not shares:
        return
    cap = price * shares
    row["market_cap"] = cap
    row["ev"] = cap + (row.get("latest_debt") or 0) - (row.get("latest_cash") or 0)
    ni, rev = row.get("net_income_ttm"), row.get("revenue_ttm")
    row["pe"] = cap / ni if ni and ni > 0 else None
    row["earnings_yield"] = ni / cap if ni is not None else None
    row["book_to_price"] = row["latest_equity"] / cap if row.get("latest_equity") is not None else None
    row["pb"] = cap / row["latest_equity"] if row.get("latest_equity") and row["latest_equity"] > 0 else None
    row["sales_to_price"] = rev / cap if rev else None
    row["ps"] = cap / rev if rev else None
    financial = row.get("sector") in CASH_FLOW_SECTORS_EXCLUDED
    row["fcf_yield"] = row["fcf"] / cap if row.get("fcf") is not None and not financial else None
    if financial:  # bank "free cash flow" swings with deposits and trading assets
        row["fcf"] = row["fcf_margin"] = None
    row["dividend_yield"] = row["dividends"] / cap if row.get("dividends") else 0.0
    row["ev_ebitda"] = row["ev"] / row["ebitda"] if row.get("ebitda") and row["ebitda"] > 0 else None


def label_companies(rows: List[Dict[str, Any]]) -> None:
    """Add size, style, cycle, rate and quality scores and labels in place."""
    by = {r["ticker"]: r for r in rows}

    def pct(key: str, better: bool = True) -> Dict[str, float]:
        return percentiles({t: r.get(key) for t, r in by.items()}, better)

    value_parts = [pct(k) for k in ("earnings_yield", "book_to_price", "sales_to_price", "fcf_yield", "dividend_yield")]
    growth_parts = [pct(k) for k in ("revenue_cagr_3y", "revenue_growth_latest_q", "net_income_cagr_3y", "revenue_cagr_5y")]
    cycle_parts = [pct("market_beta"), pct("cycle_beta"), pct("revenue_elasticity")]
    quality_specs = [("roic", True), ("roe", True), ("gross_margin", True), ("operating_margin", True),
                     ("fcf_margin", True), ("piotroski", True), ("net_debt_to_ebitda", False),
                     ("interest_coverage", True), ("gross_margin_stability", False)]
    financial_specs = [("roe", True), ("roa", True), ("equity_to_assets", True), ("piotroski", True),
                       ("net_income_stability", False)]
    fin = [r for r in rows if r["sector"] in CASH_FLOW_SECTORS_EXCLUDED]
    rest = [r for r in rows if r["sector"] not in CASH_FLOW_SECTORS_EXCLUDED]
    quality_parts = [_grouped_percentiles(rest, k, "sector", b) for k, b in quality_specs] + \
                    [_grouped_percentiles(fin, k, "sector", b) for k, b in financial_specs]
    # bargaining-power proxies within the sub-industry
    gm_in = _grouped_percentiles(rows, "gross_margin", "sub_industry")
    dso_in = _grouped_percentiles(rows, "dso", "sub_industry", higher_is_better=False)
    dpo_in = _grouped_percentiles(rows, "dpo", "sub_industry")

    for t, r in by.items():
        cap = r.get("market_cap")
        r["size"] = None if cap is None else ("大型" if cap >= LARGE_CAP else "中型" if cap >= MID_CAP else "小型")
        r["value_score"] = _avg([p.get(t) for p in value_parts], 2)
        r["growth_score"] = _avg([p.get(t) for p in growth_parts], 1)
        if r["value_score"] is not None and r["growth_score"] is not None:
            gap = r["growth_score"] - r["value_score"]
            r["style"] = "成長" if gap > STYLE_GAP else "價值" if gap < -STYLE_GAP else "均衡"
        else:
            r["style"] = None
        r["cycle_score"] = _avg([p.get(t) for p in cycle_parts], 2)
        rt = r.get("rate_t")
        r["rate"] = None if rt is None else ("升息受害" if rt <= -RATE_T else "升息受惠" if rt >= RATE_T else "不敏感")
        r["quality_score"] = _avg([p.get(t) for p in quality_parts], 3)
        r["pricing_power_pct"] = gm_in.get(t)  # gross margin vs sub-industry peers
        r["customer_terms_pct"] = dso_in.get(t)  # collects faster than peers -> stronger vs customers
        r["supplier_terms_pct"] = dpo_in.get(t)  # pays later than peers -> stronger vs suppliers
    # cycle tertiles across the universe
    cyc = percentiles({t: r.get("cycle_score") for t, r in by.items()})
    for t, r in by.items():
        p = cyc.get(t)
        r["cycle"] = None if p is None else ("高" if p >= 2 / 3 else "低" if p < 1 / 3 else "中")


def _hhi(revenues: List[float]) -> Optional[float]:
    total = sum(revenues)
    if total <= 0:
        return None
    return sum((r / total * 100) ** 2 for r in revenues)


def industry_table(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregates per sub-industry, including concentration among S&P 1500 members.

    HHI and CR4 here only count US-listed index members, so they overstate concentration where
    private, foreign or smaller rivals matter; the model's market-structure call takes that into account."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(r["sub_industry"], []).append(r)
    out = {}
    for name, members in groups.items():
        revs = sorted((m.get("revenue") or 0 for m in members if (m.get("revenue") or 0) > 0), reverse=True)
        total = sum(revs)

        def med(key: str) -> Optional[float]:
            vals = [m[key] for m in members if m.get(key) is not None]
            return median(vals) if vals else None

        # Aggregate revenue growth over 5 years for members reporting both years
        hist = [m.get("revenue_history") or {} for m in members]
        fy = max((max(map(int, h)) for h in hist if h), default=None)
        agg_cagr = None
        if fy:
            both = [(h.get(fy - 5) or h.get(str(fy - 5)), h.get(fy) or h.get(str(fy))) for h in hist]
            both = [(a, b) for a, b in both if a and b and a > 0 and b > 0]
            if len(both) >= max(1, len(members) // 2):
                agg_cagr = (sum(b for _, b in both) / sum(a for a, _ in both)) ** (1 / 5) - 1

        def share(key: str, label: str) -> float:
            labelled = [m for m in members if m.get(key)]
            return sum(m[key] == label for m in labelled) / len(labelled) if labelled else 0.0

        top = sorted(members, key=lambda m: m.get("revenue") or 0, reverse=True)
        out[name] = {
            "sub_industry": name, "sector": members[0]["sector"], "n": len(members),
            "tickers": [m["ticker"] for m in top],
            "revenue_total": total, "hhi": _hhi(revs), "cr4": sum(revs[:4]) / total if total else None,
            "top_companies": [{"ticker": m["ticker"], "name": m["name"],
                               "revenue_share": (m.get("revenue") or 0) / total if total else None} for m in top[:6]],
            "median_gross_margin": med("gross_margin"), "median_operating_margin": med("operating_margin"),
            "median_roic": med("roic"), "median_revenue_cagr_3y": med("revenue_cagr_3y"),
            "revenue_cagr_5y_aggregate": agg_cagr,
            "median_growth_score": med("growth_score"), "median_value_score": med("value_score"),
            "median_cycle_score": med("cycle_score"), "median_rate_t": med("rate_t"),
            "median_market_beta": med("market_beta"), "median_revenue_elasticity": med("revenue_elasticity"),
            "median_dso": med("dso"), "median_dpo": med("dpo"), "median_gross_margin_stability": med("gross_margin_stability"),
            "median_implied_growth": med("implied_growth"), "median_return_spread": med("return_spread"),
            "median_rpo_coverage": med("rpo_coverage"),
            "growth_share": share("style", "成長"), "value_share": share("style", "價值"),
        }
    # industry-level quantitative labels
    cyc = percentiles({k: v["median_cycle_score"] for k, v in out.items()})
    for k, v in out.items():
        g, val = v["median_growth_score"], v["median_value_score"]
        v["style"] = None if g is None or val is None else ("成長" if g - val > 0.1 else "價值" if g - val < -0.1 else "均衡")
        p = cyc.get(k)
        v["cycle"] = None if p is None else ("高" if p >= 2 / 3 else "低" if p < 1 / 3 else "中")
        rt = v["median_rate_t"]
        v["rate"] = None if rt is None else ("升息受害" if rt <= -1.0 else "升息受惠" if rt >= 1.0 else "不敏感")
    return out


def build_dataset(progress: Callable[[str], None] = lambda s: logger.info(s), today: Optional[date] = None) -> Dict[str, Any]:
    t0 = time.monotonic()
    universe = load_universe()
    progress(f"Universe: {len(universe)} companies")
    reported = fundamentals.fetch_reported({r["cik"] for r in universe}, today=today,
                                           progress=lambda d, n: progress(f"SEC XBRL frames {d}/{n}"))
    driver = macro.annual_growth("INDPRO")
    factors = macro.monthly_factors()
    quotes = market.fetch_all([r["ticker"] for r in universe], progress=lambda d, n: progress(f"Prices {d}/{n}"))
    spy = (quotes.get(market.MARKET) or {}).get("monthly") or {}
    rows = []
    for u in universe:
        row = dict(u)
        row.update(fundamentals.derive(reported.get(u["cik"], {"annual": {}}), driver))
        q = quotes.get(u["ticker"]) or {}
        row["price"] = q.get("price") if q.get("currency", "USD") == "USD" else None
        if q.get("monthly") and spy:
            sens = market.sensitivities(q["monthly"], spy, {"rate": factors["d_ten_year"], "cycle": factors["leading_index"]})
            row.update(sens)
        add_valuation(row)
        rows.append(row)
    label_companies(rows)
    from screener import hedge  # imports this module

    regime = macro.regime()
    hedge.add_metrics(rows, rf=regime["ten_year"] / 100)
    industries = industry_table(rows)
    data = {"built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "regime": regime,
            "companies": rows, "industries": industries}
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATASET_PATH.write_text(json.dumps(finite(data), default=str, allow_nan=False))
    progress(f"Dataset: {len(rows)} companies, {len(industries)} sub-industries, {time.monotonic() - t0:.0f}s")
    return data


def finite(obj: Any) -> Any:
    """Replace NaN and infinities with None so the dataset is valid JSON for the API."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite(v) for v in obj]
    return obj


_cache: Dict[str, Any] = {}


def load_dataset() -> Dict[str, Any]:
    """The saved dataset (reloaded when the file changes)."""
    if not DATASET_PATH.exists():
        raise FileNotFoundError("No screener dataset yet; run: python scripts/build_screener_data.py")
    mtime = DATASET_PATH.stat().st_mtime
    if _cache.get("mtime") != mtime:
        _cache.update(mtime=mtime, data=json.loads(DATASET_PATH.read_text()))
    return _cache["data"]

"""The questions a fundamental hedge fund asks after "is this a good business?", as numbers. No model calls.

1. Who holds the bottleneck? Company proxy: is pricing power rising (gross margin above its 5-year
   median, backlog growing faster than revenue)? The industry's place in the value chain is a model
   label in profiles.py (bottleneck); the company check with web search is in hedge_check.py.
2. How good is the revenue? Backlog (remaining performance obligations) in years of revenue, how much
   revenue swings from year to year, how many down years, how stable the gross margin is.
3. How much is priced in? A reverse DCF: the 10-year growth in free cash flow that today's enterprise
   value implies, against the growth the company has delivered. Banks and insurers: the return on
   equity that the price-to-book implies, against the ROE they earn.
4. Does it still pay at today's rates? WACC from the current 10-year yield; ROIC minus WACC, interest
   cover, and whether buybacks and dividends are being paid with debt.
5. Crowding and catalysts: momentum, valuation premium to the sector and how much growth is priced in
   (crowded trades fall hardest on bad news); the drawdown from the 12-month high; whether revenue
   growth is speeding up or slowing (estimate revisions and event dates need the web check).

Labels go on each company row; build_dataset calls add_metrics() after the other labels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from screener.features import CASH_FLOW_SECTORS_EXCLUDED, _avg, _grouped_percentiles, percentiles

EQUITY_RISK_PREMIUM = 0.05
TAX_RATE = 0.21
TERMINAL_GROWTH = 0.025  # long-run nominal growth after year 10
HORIZON = 10
BANK_GROWTH = 0.03  # long-run book growth for the implied-ROE formula
DEFAULT_DEBT_SPREAD = 0.015  # over the 10-year yield, when interest paid / debt is unusable
GROWTH_BOUNDS = (-0.5, 1.0)
GROWTH_CAPEX_MULTIPLE = 1.5  # capex above 1.5x D&A counts as growth spending

BOTTLENECK_LABELS = ["關鍵瓶頸", "部分瓶頸", "非瓶頸"]
PRICING_LABELS = ["增強", "持平", "減弱"]
QUALITY_LABELS = ["高", "中", "低"]
PRICED_IN_LABELS = ["預期偏低", "大致反映", "預期偏高", "極度樂觀"]
RATE_TEST_LABELS = ["通過", "勉強", "不通過"]
CROWDING_LABELS = ["高", "中", "低"]
MOMENTUM_LABELS = ["加速", "持平", "放緩"]

# how far implied growth (or implied ROE) may sit from what was delivered before it counts as a gap
PRICED_IN_BANDS = (-0.03, 0.03, 0.10)
ACCELERATION_BAND = 0.03


def _clip(v: float, lo: float, hi: float) -> float:
    return min(max(v, lo), hi)


def cost_of_capital(row: Dict[str, Any], rf: float) -> Dict[str, Optional[float]]:
    """CAPM cost of equity (Blume-adjusted beta), cost of debt from interest paid, market-value weights."""
    beta = row.get("market_beta")
    adj_beta = 1.0 if beta is None else 0.67 * _clip(beta, 0.3, 2.5) + 0.33
    ke = rf + adj_beta * EQUITY_RISK_PREMIUM
    debt = max(row.get("latest_debt") or row.get("debt") or 0.0, 0.0)
    interest = row.get("interest_expense")
    book_debt = row.get("debt") or debt
    kd = _clip(interest / book_debt, rf, rf + 0.06) if interest and book_debt and book_debt > 0 else rf + DEFAULT_DEBT_SPREAD
    cap = row.get("market_cap")
    if not cap or cap <= 0:
        return {"cost_of_equity": ke, "cost_of_debt": kd, "wacc": None}
    wacc = (cap * ke + debt * kd * (1 - TAX_RATE)) / (cap + debt)
    return {"cost_of_equity": ke, "cost_of_debt": kd, "wacc": max(wacc, TERMINAL_GROWTH + 0.02)}


def dcf_value(cf0: float, growth: float, wacc: float, years: int = HORIZON, terminal: float = TERMINAL_GROWTH) -> float:
    """Present value of cf0 growing at `growth` for `years`, then at `terminal` forever."""
    pv, cf = 0.0, cf0
    for t in range(1, years + 1):
        cf *= 1 + growth
        pv += cf / (1 + wacc) ** t
    return pv + cf * (1 + terminal) / (wacc - terminal) / (1 + wacc) ** years


def implied_growth(ev: float, cf0: float, wacc: float) -> Optional[float]:
    """The constant 10-year growth that makes the DCF equal `ev` (bisection; clipped to GROWTH_BOUNDS)."""
    if ev <= 0 or cf0 <= 0 or wacc <= TERMINAL_GROWTH:
        return None
    lo, hi = GROWTH_BOUNDS
    if dcf_value(cf0, lo, wacc) >= ev:
        return lo
    if dcf_value(cf0, hi, wacc) <= ev:
        return hi
    for _ in range(80):
        mid = (lo + hi) / 2
        if dcf_value(cf0, mid, wacc) < ev:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _band(gap: Optional[float]) -> Optional[str]:
    if gap is None:
        return None
    lo, mid, hi = PRICED_IN_BANDS
    return "預期偏低" if gap < lo else "大致反映" if gap <= mid else "預期偏高" if gap <= hi else "極度樂觀"


def delivered_growth(row: Dict[str, Any]) -> Optional[float]:
    """What the company has delivered: the mean of its 3- and 5-year revenue CAGRs."""
    return _avg([row.get("revenue_cagr_3y"), row.get("revenue_cagr_5y")])


def base_cash_flow(row: Dict[str, Any]) -> tuple:
    """The cash flow the reverse DCF grows. Free cash flow, except for companies spending well above
    depreciation to grow (pipelines, power plants, data centers, fabs): their reported FCF is net of the
    very investment that produces the growth being solved for, so maintenance capex is taken as D&A."""
    fcf, cfo, capex, dna = row.get("fcf"), row.get("cfo"), row.get("capex"), row.get("dna")
    if cfo is not None and capex and dna and capex > GROWTH_CAPEX_MULTIPLE * dna and cfo - dna > 0:
        return cfo - dna, "維持性現金流"
    if fcf and fcf > 0:
        return fcf, "自由現金流"
    ebit = row.get("operating_income")
    if ebit and ebit > 0:
        return ebit * (1 - TAX_RATE), "稅後營業利益"
    return None, "虧損"


def reverse_dcf(row: Dict[str, Any], rf: float) -> Dict[str, Any]:
    out: Dict[str, Any] = cost_of_capital(row, rf)
    if row.get("sector") in CASH_FLOW_SECTORS_EXCLUDED:
        # justified P/B = (ROE - g) / (ke - g)  ->  implied ROE = g + P/B * (ke - g)
        pb = row.get("pb")
        if pb:
            out["implied_roe"] = BANK_GROWTH + pb * (out["cost_of_equity"] - BANK_GROWTH)
            if row.get("roe") is not None:
                out["implied_gap"] = out["implied_roe"] - row["roe"]
        out["dcf_base"] = "股價淨值比"
        out["priced_in"] = _band(out.get("implied_gap"))
        return out
    ev, wacc = row.get("ev"), out["wacc"]
    cf0, out["dcf_base"] = base_cash_flow(row)
    if cf0 and ev and wacc:
        g = implied_growth(ev, cf0, wacc)
        out["implied_growth"] = g
        hist = delivered_growth(row)
        out["delivered_growth"] = hist
        if g is not None and hist is not None:
            out["implied_gap"] = g - hist
    out["priced_in"] = _band(out.get("implied_gap"))
    return out


def rate_test(row: Dict[str, Any], rf: float, coc: Dict[str, Optional[float]]) -> Dict[str, Any]:
    financial = row.get("sector") in CASH_FLOW_SECTORS_EXCLUDED
    ret, hurdle = (row.get("roe"), coc["cost_of_equity"]) if financial else (row.get("roic"), coc["wacc"])
    spread = ret - hurdle if ret is not None and hurdle is not None else None
    ey = row.get("earnings_yield")
    paid_out = (row.get("buybacks") or 0) + (row.get("dividends") or 0)
    nd = row.get("net_debt_to_ebitda")
    levered_payout = bool(paid_out > max(row.get("fcf") or 0, 0) and nd is not None and nd > 3) if not financial else False
    cover = row.get("interest_coverage")
    if spread is None:
        label = None
    elif spread < 0 or (cover is not None and cover < 2 and not financial):
        label = "不通過"
    elif spread >= 0.02 and not levered_payout and (financial or cover is None or cover >= 4 or (nd is not None and nd <= 0)):
        label = "通過"
    else:
        label = "勉強"
    return {"return_spread": spread, "earnings_yield_spread": ey - rf if ey is not None else None,
            "levered_payout": levered_payout, "rate_test": label}


def _signal(v: Optional[float], band: float) -> Optional[int]:
    if v is None:
        return None
    return 1 if v > band else -1 if v < -band else 0


def pricing_trend(row: Dict[str, Any]) -> Optional[str]:
    """Gross margin above its 5-year median, and backlog growing faster than revenue -> 增強."""
    gm, gm5 = row.get("gross_margin"), row.get("gross_margin_5y_median")
    sig = [_signal(gm - gm5 if gm is not None and gm5 is not None else None, 0.01)]
    rg, cagr = row.get("rpo_growth"), row.get("revenue_cagr_3y")
    sig.append(_signal(rg - cagr if rg is not None and cagr is not None else None, 0.05))
    sig = [s for s in sig if s is not None]
    if not sig:
        return None
    net = sum(sig)
    return "增強" if net >= 1 else "減弱" if net <= -1 else "持平"


def fundamental_momentum(row: Dict[str, Any]) -> Optional[str]:
    q, cagr = row.get("revenue_growth_latest_q"), row.get("revenue_cagr_3y")
    s = _signal(q - cagr if q is not None and cagr is not None else None, ACCELERATION_BAND)
    return None if s is None else {1: "加速", 0: "持平", -1: "放緩"}[s]


def _tertile(p: Optional[float], labels: List[str]) -> Optional[str]:
    """labels ordered high -> low."""
    if p is None:
        return None
    return labels[0] if p >= 2 / 3 else labels[2] if p < 1 / 3 else labels[1]


def add_metrics(rows: List[Dict[str, Any]], rf: float) -> None:
    """Add the hedge-fund labels in place; rf is the 10-year Treasury yield as a fraction (0.05)."""
    for r in rows:
        dcf = reverse_dcf(r, rf)
        r.update(dcf)
        r.update(rate_test(r, rf, dcf))
        r["pricing_trend"] = pricing_trend(r)
        r["fund_momentum"] = fundamental_momentum(r)
        q, cagr = r.get("revenue_growth_latest_q"), r.get("revenue_cagr_3y")
        r["revenue_acceleration"] = q - cagr if q is not None and cagr is not None else None
    by = {r["ticker"]: r for r in rows}

    def pct(key: str, better: bool = True) -> Dict[str, float]:
        return percentiles({t: r.get(key) for t, r in by.items()}, better)

    rq_parts = [pct("rpo_coverage"), pct("revenue_growth_volatility", False), pct("revenue_down_years", False),
                pct("gross_margin_stability", False)]
    rich_parts = [_grouped_percentiles(rows, "ev_ebitda", "sector", higher_is_better=False),
                  _grouped_percentiles(rows, "pe", "sector", higher_is_better=False)]
    crowd_parts = [pct("momentum_12m"), pct("implied_gap")]
    for t, r in by.items():
        r["revenue_quality_score"] = _avg([p.get(t) for p in rq_parts], 2)
        # cheap vs sector -> high percentile above, so the premium is 1 - that
        cheap = _avg([p.get(t) for p in rich_parts])
        premium = None if cheap is None else 1 - cheap
        r["valuation_premium_pct"] = premium
        r["crowding_score"] = _avg([*(p.get(t) for p in crowd_parts), premium], 2)
    rq, cr = pct("revenue_quality_score"), pct("crowding_score")
    for t, r in by.items():
        r["revenue_quality"] = _tertile(rq.get(t), QUALITY_LABELS)
        r["crowding"] = _tertile(cr.get(t), CROWDING_LABELS)

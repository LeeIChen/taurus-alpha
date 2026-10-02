"""Deterministic DCF and P/E valuation from model-chosen assumptions."""

from __future__ import annotations

from typing import Optional

from app.state import ValuationAssumptions, ValuationResult

SENSITIVITY_RATE_STEP = 0.01
SENSITIVITY_GROWTH_STEP = 0.005


def dcf_value_per_share(
    a: ValuationAssumptions,
    discount_rate: Optional[float] = None,
    terminal_growth: Optional[float] = None,
) -> float:
    """Two-stage DCF: explicit FCF growth years, then a Gordon-growth terminal value."""
    r = a.discount_rate if discount_rate is None else discount_rate
    g = a.terminal_growth if terminal_growth is None else terminal_growth
    if r <= g:
        raise ValueError(f"Discount rate ({r:.2%}) must exceed terminal growth ({g:.2%})")
    if a.diluted_shares <= 0:
        raise ValueError("Diluted shares must be positive")

    fcf = a.base_fcf
    present_value = 0.0
    for year, growth in enumerate(a.fcf_growth_rates, start=1):
        fcf *= 1 + growth
        present_value += fcf / (1 + r) ** year
    terminal_value = fcf * (1 + g) / (r - g)
    present_value += terminal_value / (1 + r) ** len(a.fcf_growth_rates)
    return (present_value + a.net_cash) / a.diluted_shares


def value_company(a: ValuationAssumptions, current_price: Optional[float] = None) -> ValuationResult:
    dcf = dcf_value_per_share(a)
    pe = a.forward_eps * a.target_pe
    weight = min(max(a.dcf_weight, 0.0), 1.0)
    target = weight * dcf + (1 - weight) * pe

    sensitivity = []
    for dr in (-SENSITIVITY_RATE_STEP, 0.0, SENSITIVITY_RATE_STEP):
        for dg in (-SENSITIVITY_GROWTH_STEP, 0.0, SENSITIVITY_GROWTH_STEP):
            r, g = a.discount_rate + dr, a.terminal_growth + dg
            if r > g:
                sensitivity.append(
                    {"discount_rate": round(r, 4), "terminal_growth": round(g, 4),
                     "value_per_share": round(dcf_value_per_share(a, r, g), 2)}
                )

    upside = (target / current_price - 1) * 100 if current_price else None
    return ValuationResult(
        assumptions=a,
        dcf_value_per_share=round(dcf, 2),
        pe_target_price=round(pe, 2),
        target_price=round(target, 2),
        current_price=current_price,
        upside_pct=round(upside, 1) if upside is not None else None,
        dcf_sensitivity=sensitivity,
    )

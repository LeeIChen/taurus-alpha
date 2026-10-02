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


# ---- Pipeline valuation: sum of the parts with risk-adjusted NPV per program ----
from datetime import date as _date
from typing import Dict as _Dict, List as _List

from app.state import PipelineAssumptions, PipelineValuationResult, ProgramAssumption, ProgramValue

HORIZON_YEARS = 20
LAUNCH_MARGIN = -0.20  # launch year carries heavy selling cost; margin reaches target by year 4


def _valuation_point(today: _date) -> float:
    return today.year + (today.timetuple().tm_yday - 1) / 365.0


def _years(today: _date) -> _List[int]:
    return list(range(today.year, today.year + HORIZON_YEARS))


def _year_fraction(y: int, today: _date) -> float:
    return 1.0 - (today.timetuple().tm_yday - 1) / 365.0 if y == today.year else 1.0


def _discount(y: int, r: float, today: _date) -> float:
    t = max(y + 0.5 - _valuation_point(today), 0.125)
    return 1 / (1 + r) ** t


def program_sales(p: ProgramAssumption, y: int, today: Optional[_date] = None) -> float:
    """Sales path: marketed products move linearly from current sales to peak; launches follow
    an S-curve (t/T)^1.5 to peak; from decline_start_year sales fall by annual_decline."""
    this_year = (today or _date.today()).year
    if y >= p.decline_start_year:
        level = program_sales(p.model_copy(update={"decline_start_year": 10_000}), p.decline_start_year - 1, today)
        return level * (1 - p.annual_decline) ** (y - p.decline_start_year + 1)
    if y < p.launch_year:
        return 0.0
    if y >= p.peak_year:
        return p.peak_sales
    if p.current_sales > 0:
        start = max(p.launch_year, this_year)
        frac = (y - start) / max(p.peak_year - start, 1)
        return p.current_sales + (p.peak_sales - p.current_sales) * min(max(frac, 0.0), 1.0)
    frac = (y - p.launch_year + 1) / (p.peak_year - p.launch_year + 1)
    return p.peak_sales * frac ** 1.5


def program_margin(p: ProgramAssumption, y: int) -> float:
    if y < p.launch_year:
        return 0.0
    t = y - p.launch_year
    return p.contribution_margin if t >= 3 or p.current_sales > 0 else LAUNCH_MARGIN + (p.contribution_margin - LAUNCH_MARGIN) * t / 3


def _tax(a: PipelineAssumptions, y: int) -> float:
    return 0.0 if y <= a.tax_free_until_year else a.long_run_tax_rate


def program_flows(p: ProgramAssumption, a: PipelineAssumptions, today: _date, risked: bool = True) -> _Dict[int, float]:
    flows = {}
    for y in _years(today):
        prob = p.probability_of_success if risked else 1.0
        op = program_sales(p, y, today) * program_margin(p, y) * p.economic_share * prob
        dev = p.remaining_dev_cost_per_year if y <= p.readout_year + 1 else 0.0
        dev *= 1.0 if (y <= p.readout_year or not risked) else p.probability_of_success
        pre = (op - dev) * _year_fraction(y, today)
        flows[y] = pre * (1 - _tax(a, y)) if pre > 0 else pre
    return flows


def risked_operating_income(a: PipelineAssumptions, y: int) -> float:
    op = 0.0
    for p in a.programs:
        op += program_sales(p, y) * program_margin(p, y) * p.economic_share * p.probability_of_success
        dev = p.remaining_dev_cost_per_year if y <= p.readout_year + 1 else 0.0
        op -= dev * (1.0 if y <= p.readout_year else p.probability_of_success)
    return op - a.corporate_overhead_per_year


def value_pipeline(a: PipelineAssumptions, current_price: Optional[float] = None,
                   today: Optional[_date] = None) -> PipelineValuationResult:
    today = today or _date.today()
    r = a.discount_rate
    if a.diluted_shares <= 0:
        raise ValueError("Diluted shares must be positive")
    if not 0 < r < 1:
        raise ValueError(f"Discount rate {r} must be between 0 and 1")
    npv = lambda flows: sum(cf * _discount(y, r, today) for y, cf in flows.items())

    programs = []
    for p in a.programs:
        rn, un = npv(program_flows(p, a, today)), npv(program_flows(p, a, today, risked=False))
        programs.append(ProgramValue(name=p.name, group=p.group, stage=p.stage,
                                     probability_of_success=p.probability_of_success,
                                     rnpv=round(rn, 1), unrisked_npv=round(un, 1),
                                     per_share=round(rn / a.diluted_shares, 2)))
    corporate = npv({y: -a.corporate_overhead_per_year * _year_fraction(y, today) for y in _years(today)})
    base = corporate + a.net_cash + a.other_adjustments
    dcf = (sum(p.rnpv for p in programs) + base) / a.diluted_shares
    unrisked = (sum(p.unrisked_npv for p in programs) + base) / a.diluted_shares

    eps = risked_operating_income(a, a.pe_year) * (1 - _tax(a, a.pe_year)) / a.diluted_shares
    target_point = _valuation_point(today) + 1
    pe_price = eps * a.target_pe / (1 + r) ** max(a.pe_year - target_point, 0)
    weight = min(max(a.dcf_weight, 0.0), 1.0)
    target = weight * dcf * (1 + r) + (1 - weight) * pe_price
    upside = (target / current_price - 1) * 100 if current_price else None
    return PipelineValuationResult(
        assumptions=a, programs=programs, corporate_value=round(corporate, 1),
        dcf_value_per_share=round(dcf, 2), unrisked_value_per_share=round(unrisked, 2),
        pe_eps=round(eps, 2), pe_target_price=round(pe_price, 2), target_price=round(target, 2),
        current_price=current_price, upside_pct=round(upside, 1) if upside is not None else None,
        implied_note=_implied_note(a, current_price, today) if current_price else "",
    )


def _implied_note(a: PipelineAssumptions, price: float, today: _date) -> str:
    """How much the largest non-marketed group's peak sales must scale (at 100% PoS) to reach the price."""
    groups: _Dict[str, float] = {}
    for p in a.programs:
        if p.probability_of_success < 1:
            groups[p.group] = groups.get(p.group, 0.0) + p.peak_sales
    if not groups:
        return ""
    group = max(groups, key=groups.get)

    def value_at(k: float) -> float:
        progs = [p.model_copy(update={"peak_sales": p.peak_sales * k, "probability_of_success": 1.0})
                 if p.group == group else p for p in a.programs]
        return value_pipeline(a.model_copy(update={"programs": progs}), None, today).dcf_value_per_share

    lo, hi = 0.0, 50.0
    if value_at(hi) < price:
        return f"Even 50x the modelled {group} peak sales with certain success does not reach ${price:,.2f}."
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if value_at(mid) < price else (lo, mid)
    return (f"To justify ${price:,.2f} with everything else unchanged, {group} programs would need about "
            f"${groups[group] * hi / 1000:,.1f}B of combined peak sales with every program succeeding "
            f"(modelled: ${groups[group] / 1000:,.1f}B before risk adjustment).")

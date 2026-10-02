"""Moderna (MRNA) valuation: program-level risk-adjusted NPV (sum of the parts) + P/E cross-check.

All money in USD millions, Moderna's economic share. Valuation date 2026-10-01.
Cash flows are annual, mid-year discounted; 2026 counts only Q4 (25%).
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List

YEARS = list(range(2026, 2046))
VAL_DATE = 2026.75  # Oct 1, 2026
PRICE = 188.94  # Oct 1, 2026 close
BASIC_SHARES = 399.2  # 10-Q cover, Jul 24 2026
DILUTED_SHARES = 405.0  # + ~6M dilutive equity awards (assumption)
CONVERT_PRINCIPAL = 3000.0
CONVERT_PRICE = 210.58
CONVERT_SHARES = CONVERT_PRINCIPAL / 1000 * 4.7487  # 14.25M
CASH_PRO_FORMA = 8050.0  # Jun-30 6.9B - 0.95B settlement - ~0.5B Q3 burn + 2.63B net convert (after capped call)
DEBT_LOAN = 600.0
LITIGATION_CONTINGENT = 1300.0
P_LITIGATION_LOSS = 0.35  # judgment: Federal Circuit appeal on 28 U.S.C. 1498 immunity


def tax_rate(year: int) -> float:
    # ~$11B of 2023-25 losses -> NOLs shield profits for several years (assumption)
    return 0.0 if year <= 2031 else 0.15


@dataclass
class Program:
    name: str
    group: str
    stage: str
    pos: float  # probability of technical/regulatory success (1.0 = marketed)
    pos_basis: str
    launch: int
    peak_sales: float  # global net sales at Moderna-share basis (100% for wholly owned)
    peak_year: int
    margin: float  # contribution margin at peak after COGS, SG&A, lifecycle R&D
    share: float = 1.0  # Moderna economic share (0.5 for Merck 50/50)
    decline_start: int = 2039
    decline_rate: float = 0.10
    dev_cost: Dict[int, float] = field(default_factory=dict)  # Moderna-share R&D before success (unrisked)
    readout: int = 2026  # year the risk resolves; costs after it are probability-weighted
    sales_override: Dict[int, float] = field(default_factory=dict)


def sales_curve(p: Program) -> Dict[int, float]:
    if p.sales_override:
        return {y: p.sales_override.get(y, 0.0) for y in YEARS}
    out = {}
    ramp_years = max(p.peak_year - p.launch, 1)
    for y in YEARS:
        if y < p.launch:
            s = 0.0
        elif y < p.peak_year:
            # S-curve style ramp: (t/T)^1.5
            s = p.peak_sales * ((y - p.launch + 1) / (ramp_years + 1)) ** 1.5
        elif y < p.decline_start:
            s = p.peak_sales
        else:
            s = p.peak_sales * (1 - p.decline_rate) ** (y - p.decline_start + 1)
        out[y] = s
    return out


def margin_curve(p: Program, y: int) -> float:
    # Launch years carry heavy SG&A: margin ramps from -20% in launch year to target by year 4
    if y < p.launch:
        return 0.0
    t = y - p.launch
    return p.margin if t >= 3 else -0.20 + (p.margin + 0.20) * t / 3


def discount(y: int, r: float) -> float:
    t = max(y + 0.5 - VAL_DATE, 0.125)
    return 1 / (1 + r) ** t


def year_fraction(y: int) -> float:
    return 0.25 if y == 2026 else 1.0


def program_flows(p: Program, risked: bool = True) -> Dict[int, float]:
    sales = sales_curve(p)
    flows = {}
    for y in YEARS:
        prob = p.pos if risked else 1.0
        op = sales[y] * margin_curve(p, y) * p.share * prob
        dev = p.dev_cost.get(y, 0.0)
        dev *= 1.0 if (y <= p.readout or not risked) else p.pos
        pre_tax = (op - dev) * year_fraction(y)
        flows[y] = pre_tax * (1 - tax_rate(y)) if pre_tax > 0 else pre_tax
    return flows


def npv(flows: Dict[int, float], r: float, terminal_growth: float = None) -> float:
    v = sum(cf * discount(y, r) for y, cf in flows.items())
    if terminal_growth is not None:
        last = YEARS[-1]
        tv = flows[last] * (1 + terminal_growth) / (r - terminal_growth)
        v += tv * discount(last, r)
    return v


# ------------------------------------------------------------------ programs
def build_programs(overrides: Dict[str, dict] = None) -> List[Program]:
    progs = [
        # ---------------- Respiratory / infectious disease (wholly owned)
        Program("Spikevax + mNEXSPIKE (COVID)", "COVID", "Marketed", 1.0,
                "Marketed", 2020, 0, 2026, 0.30,
                sales_override={2026: 1650, 2027: 1450, 2028: 1330, 2029: 1260, 2030: 1200,
                                **{y: 1200 * 0.97 ** (y - 2030) for y in range(2031, 2046)}}),
        Program("mRESVIA (RSV)", "Other respiratory", "Marketed", 1.0, "Marketed", 2024, 300, 2031, 0.30,
                sales_override={2026: 60, 2027: 100, 2028: 150, 2029: 200, 2030: 250,
                                **{y: 300 * 0.98 ** max(y - 2031, 0) for y in range(2031, 2046)}}),
        Program("mFLUSIVA (seasonal flu, adults 50+)", "Flu", "Approved Aug 2026", 0.80,
                "Approved (technical PoS 100%); 80% = probability of commercial adoption given ACIP/HHS "
                "uncertainty (ACIP reconstituted, COVID recs removed; CIDRAP May 2026)",
                2026, 1200, 2031, 0.40, decline_start=2040, decline_rate=0.05),
        Program("mCOMBRIAX (flu + COVID combo)", "Flu", "EU approved; US refile pending", 0.60,
                "EU authorization received; US refiling awaiting FDA guidance. 60% blends EU approval "
                "(certain) with an uncertain US path (judgment)",
                2027, 600, 2032, 0.40, decline_start=2040, decline_rate=0.05),
        Program("mRNA-1403 (norovirus)", "Other respiratory", "Phase 3 (missed interim)", 0.35,
                "BIO 2011-20 vaccine Phase 3→filing 58.1%, filing→approval 100%; cut to 35% after the "
                "Phase 3 interim missed early-success criteria (Q2 2026 release)",
                2029, 1000, 2034, 0.40, readout=2027, dev_cost={2026: 40, 2027: 120, 2028: 60},
                decline_start=2042, decline_rate=0.05),
        # ---------------- Rare disease (wholly owned)
        Program("mRNA-3927 (propionic acidemia)", "Rare disease", "Registrational, enrolled", 0.50,
                "BIO rare disease Phase 3→filing 60.4% × filing→approval 93.6% = 56.5%; trimmed to 50% for a "
                "first-of-kind chronic mRNA protein-replacement therapy",
                2028, 400, 2033, 0.55, readout=2026, dev_cost={2026: 30, 2027: 80, 2028: 40}),
        Program("mRNA-3705 (methylmalonic acidemia)", "Rare disease", "Pivotal decision deferred", 0.25,
                "BIO rare disease LOA from Phase 2 = 25.2%; pivotal start deferred until PA data",
                2031, 500, 2036, 0.55, readout=2029, dev_cost={2027: 60, 2028: 120, 2029: 120, 2030: 40}),
        # ---------------- Intismeran (50/50 with Merck): peak sales are GLOBAL; share=0.5
        Program("Intismeran: adjuvant melanoma (INTerpath-001)", "Intismeran", "Phase 3 positive (Aug 2026)", 0.90,
                "Phase 3 met RFS and DMFS at first interim. BIO immuno-oncology filing→approval 98.4%; cut to 90% "
                "because the HR, OS and filing timing are undisclosed and CBER review of a first-in-class "
                "individualized mRNA therapy adds regulatory risk",
                2027, 2500, 2033, 0.58, share=0.5, readout=2027, dev_cost={2026: 60, 2027: 120, 2028: 80, 2029: 40}),
        Program("Intismeran: adjuvant NSCLC stage II-IIIB (INTerpath-002)", "Intismeran", "Phase 3", 0.50,
                "BIO immuno-oncology Phase 3 LOA 48.2%; ~50% given same-agent Phase 3 success in melanoma, offset "
                "by no randomized NSCLC data and lower tumor mutational burden than melanoma",
                2029, 3000, 2035, 0.58, share=0.5, readout=2028, decline_start=2041,
                dev_cost={2026: 70, 2027: 150, 2028: 150, 2029: 80}),
        Program("Intismeran: NSCLC non-pCR post-neoadjuvant (INTerpath-009)", "Intismeran", "Phase 3", 0.45,
                "BIO immuno-oncology Phase 3 LOA 48.2%; 45% for a harder residual-disease population",
                2031, 800, 2036, 0.58, share=0.5, readout=2030, decline_start=2041,
                dev_cost={2026: 40, 2027: 90, 2028: 100, 2029: 100, 2030: 60}),
        Program("Intismeran: high-risk stage I NSCLC (INTerpath-014)", "Intismeran", "Phase 3 (just started)", 0.40,
                "BIO immuno-oncology Phase 3 LOA 48.2%; 40% for a monotherapy arm in early-stage disease with a "
                "2034 primary completion",
                2033, 1500, 2039, 0.58, share=0.5, readout=2032, decline_start=2043,
                dev_cost={y: 90 for y in range(2026, 2033)}),
        Program("Intismeran: adjuvant RCC (INTerpath-004)", "Intismeran", "Phase 2 (data 2027)", 0.25,
                "BIO immuno-oncology LOA from Phase 2 = 19.4%; raised to 25% for platform validation and an "
                "established adjuvant pembrolizumab backbone (KEYNOTE-564)",
                2031, 800, 2036, 0.58, share=0.5, readout=2027, decline_start=2041,
                dev_cost={2026: 20, 2027: 40, 2028: 100, 2029: 100, 2030: 60}),
        Program("Intismeran: muscle-invasive bladder (INTerpath-005)", "Intismeran", "Phase 1/2", 0.20,
                "BIO immuno-oncology LOA from Phase 2 = 19.4% → 20%", 2032, 700, 2037, 0.58, share=0.5,
                readout=2030, decline_start=2041, dev_cost={y: 40 for y in range(2026, 2031)}),
        Program("Intismeran: high-risk NMIBC (INTerpath-011)", "Intismeran", "Phase 2", 0.15,
                "BIO immuno-oncology LOA from Phase 2 = 19.4%; 15% for an open-label BCG-combination design "
                "with a 2031 completion", 2034, 800, 2039, 0.58, share=0.5, readout=2031, decline_start=2043,
                dev_cost={y: 35 for y in range(2026, 2032)}),
        Program("Intismeran: 1L advanced melanoma (INTerpath-012)", "Intismeran", "Phase 2", 0.20,
                "BIO immuno-oncology LOA from Phase 2 = 19.4% → 20%", 2031, 600, 2036, 0.58, share=0.5,
                readout=2028, decline_start=2041, dev_cost={2026: 15, 2027: 30, 2028: 30, 2029: 60, 2030: 60}),
        Program("Intismeran: 1L metastatic squamous NSCLC (INTerpath-013)", "Intismeran", "Phase 2", 0.15,
                "BIO solid-tumor LOA from Phase 2 = 9.3%, immuno-oncology 19.4%; 15% because vaccines have a "
                "weak record in metastatic disease", 2032, 800, 2037, 0.58, share=0.5, readout=2029,
                decline_start=2041, dev_cost={2026: 15, 2027: 30, 2028: 30, 2029: 30, 2030: 60, 2031: 60}),
        # ---------------- Other oncology (wholly owned)
        Program("mRNA-4359 (checkpoint cancer therapy)", "Other oncology", "Phase 1/2", 0.10,
                "BIO solid-tumor LOA from Phase 2 = 9.3% → 10%", 2032, 800, 2037, 0.55,
                readout=2028, decline_start=2042, dev_cost={2026: 25, 2027: 80, 2028: 80, 2029: 120, 2030: 120, 2031: 60}),
    ]
    for p in progs:
        for k, v in (overrides or {}).get(p.name, {}).items():
            setattr(p, k, v)
    return progs


def corporate_flows(overhead: float = 550.0) -> Dict[int, float]:
    # Unallocated corporate G&A + platform R&D kept after cost cuts (assumption); early pipeline valued at 0
    flows = {}
    for y in YEARS:
        base = 900.0 if y == 2026 else 750.0 if y == 2027 else 650.0 if y == 2028 else overhead * 1.02 ** (y - 2029)
        pre = -base * year_fraction(y) - (40.0 if y <= 2030 else 0) * year_fraction(y)  # loan interest
        flows[y] = pre
    return flows


def valuation(r: float = 0.10, overrides=None, corp_overhead: float = 550.0, p_lit: float = P_LITIGATION_LOSS):
    progs = build_programs(overrides)
    rows = []
    for p in progs:
        v = npv(program_flows(p, risked=True), r)
        vu = npv(program_flows(p, risked=False), r)
        rows.append({"name": p.name, "group": p.group, "stage": p.stage, "pos": p.pos, "pos_basis": p.pos_basis,
                     "launch": p.launch, "peak_sales": p.peak_sales, "share": p.share,
                     "rnpv": v, "unrisked_npv": vu})
    corp = npv(corporate_flows(corp_overhead), r)
    net_cash = CASH_PRO_FORMA - DEBT_LOAN - CONVERT_PRINCIPAL
    lit = -LITIGATION_CONTINGENT * p_lit
    equity = sum(x["rnpv"] for x in rows) + corp + net_cash + lit
    shares = DILUTED_SHARES
    per_share = equity / shares
    # If value exceeds the conversion price, treat the convert as equity (add back principal, add shares)
    if per_share > CONVERT_PRICE:
        per_share = (equity + CONVERT_PRINCIPAL) / (shares + CONVERT_SHARES)
    return {"rows": rows, "corporate": corp, "net_cash": net_cash, "litigation": lit,
            "equity": equity, "per_share": per_share, "shares": shares, "r": r}


def risked_pnl(year: int, overrides=None, corp_overhead: float = 550.0):
    """Probability-weighted operating income for one year (for the P/E method)."""
    progs = build_programs(overrides)
    op = 0.0
    for p in progs:
        s = sales_curve(p)[year]
        op += s * margin_curve(p, year) * p.share * p.pos
        dev = p.dev_cost.get(year, 0.0)
        op -= dev * (1.0 if year <= p.readout else p.pos)
    op -= corporate_flows(corp_overhead)[year] * -1
    return op


def pe_target(eps_year: int = 2030, pe: float = 20.0, r: float = 0.10, overrides=None, corp_overhead: float = 550.0):
    op = risked_pnl(eps_year, overrides, corp_overhead)
    net = op * (1 - tax_rate(eps_year))
    eps = net / (DILUTED_SHARES + CONVERT_SHARES)  # fully diluted for an earnings multiple
    value_at_year_start = eps * pe  # value at the start of eps_year (forward multiple)
    target_date = VAL_DATE + 1  # 12-month target, Oct 2027
    years = eps_year - target_date
    return {"eps_year": eps_year, "op_income": op, "net_income": net, "eps": eps, "pe": pe,
            "value_at_eps_year_start": value_at_year_start,
            "pv_at_target_date": value_at_year_start / (1 + r) ** years}


if __name__ == "__main__":
    base = valuation()
    print(f"SOTP equity ${base['equity']:,.0f}M -> ${base['per_share']:.2f}/share")
    for x in sorted(base["rows"], key=lambda x: -x["rnpv"]):
        print(f"  {x['name'][:58]:58} PoS {x['pos']:.0%}  rNPV {x['rnpv']:8,.0f}  /sh {x['rnpv']/base['shares']:7.2f}  unrisked {x['unrisked_npv']:8,.0f}")
    print(f"  corporate {base['corporate']:,.0f}  net cash {base['net_cash']:,.0f}  litigation {base['litigation']:,.0f}")
    for yr in (2028, 2030, 2032):
        print("risked op income", yr, round(risked_pnl(yr)))
    print(pe_target())

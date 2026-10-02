import json
from pathlib import Path

import mrna_model as m

HERE = Path(__file__).resolve().parent

INT = [p.name for p in m.build_programs() if p.group == "Intismeran"]


def scale_intismeran(k, pos_add=0.0):
    out = {}
    for p in m.build_programs():
        if p.group == "Intismeran":
            out[p.name] = {"peak_sales": p.peak_sales * k, "pos": min(p.pos + pos_add, 0.98)}
    return out


def merge(*dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, {}).update(v)
    return out


base = m.valuation()
pe20 = m.pe_target(2032, 20.0)
pe25 = m.pe_target(2032, 25.0)

# --- scenarios
bear = merge(
    scale_intismeran(0.6, pos_add=-0.10),
    {"Intismeran: adjuvant melanoma (INTerpath-001)": {"peak_sales": 1500, "pos": 0.85},
     "mFLUSIVA (seasonal flu, adults 50+)": {"peak_sales": 600, "pos": 0.6},
     "mCOMBRIAX (flu + COVID combo)": {"pos": 0.35},
     "Spikevax + mNEXSPIKE (COVID)": {"sales_override": {y: v * (0.85 ** max(y - 2026, 0)) / (0.97 ** max(y - 2030, 0) if y > 2030 else 1)
                                                         for y, v in m.sales_curve(m.build_programs()[0]).items()}}})
bull = merge(
    scale_intismeran(1.6, pos_add=0.15),
    {"Intismeran: adjuvant melanoma (INTerpath-001)": {"peak_sales": 5000, "pos": 0.95},
     "mFLUSIVA (seasonal flu, adults 50+)": {"peak_sales": 2000, "pos": 0.9},
     "mCOMBRIAX (flu + COVID combo)": {"pos": 0.8, "peak_sales": 1000},
     "mRNA-1403 (norovirus)": {"pos": 0.5}})
scen = {}
for name, ov in (("Bear", bear), ("Base", None), ("Bull", bull)):
    v = m.valuation(overrides=ov)
    p20 = m.pe_target(2032, 20.0, overrides=ov)
    scen[name] = {"dcf": v["per_share"], "pe20": p20["pv_at_target_date"], "eps2032": p20["eps"]}

# --- sensitivities
sens_r = {f"{r:.0%}": m.valuation(r=r)["per_share"] for r in (0.08, 0.09, 0.10, 0.11, 0.12)}
sens_mel = {f"${pk/1000:.1f}B": m.valuation(overrides={"Intismeran: adjuvant melanoma (INTerpath-001)": {"peak_sales": pk}})["per_share"]
            for pk in (1500, 2500, 4000, 5400, 8000)}
sens_overhead = {f"${o}M": m.valuation(corp_overhead=o)["per_share"] for o in (300, 550, 800)}

# --- implied expectations: intismeran-wide peak-sales multiple (all tumor types) at 100% PoS needed for $188.94
def solve(target, pos_mode):
    lo, hi = 0.5, 40.0
    for _ in range(60):
        mid = (lo + hi) / 2
        ov = {}
        for p in m.build_programs():
            if p.group == "Intismeran":
                ov[p.name] = {"peak_sales": p.peak_sales * mid, **({"pos": 1.0} if pos_mode == "certain" else {})}
        v = m.valuation(overrides=ov, r=0.10)["per_share"]
        lo, hi = (mid, hi) if v < target else (lo, mid)
    return mid

base_int_peak = sum(p.peak_sales for p in m.build_programs() if p.group == "Intismeran")
k_risked = solve(m.PRICE, "risked")
k_certain = solve(m.PRICE, "certain")

# --- blended target: 12-month DCF (roll forward at r) and P/E (20x 2032 EPS, PV to Oct 2027)
dcf_12m = base["per_share"] * 1.10
blend = 0.5 * dcf_12m + 0.5 * pe20["pv_at_target_date"]

out = {
    "price": m.PRICE, "shares": m.DILUTED_SHARES, "market_cap": m.PRICE * m.DILUTED_SHARES,
    "base": {k: base[k] for k in ("equity", "per_share", "corporate", "net_cash", "litigation", "r")},
    "rows": base["rows"],
    "pe20": pe20, "pe25": pe25, "dcf_12m": dcf_12m, "blend": blend,
    "scenarios": scen, "sens_r": sens_r, "sens_mel": sens_mel, "sens_overhead": sens_overhead,
    "implied": {"base_int_peak_global": base_int_peak, "k_risked": k_risked, "k_certain": k_certain,
                "int_peak_needed_risked": base_int_peak * k_risked, "int_peak_needed_certain": base_int_peak * k_certain},
    "risked_op": {y: m.risked_pnl(y) for y in (2027, 2028, 2029, 2030, 2031, 2032, 2033, 2035)},
}
json.dump(out, open(HERE / "mrna_results.json", "w"), indent=2, default=float)

print(f"DCF/SOTP now ${base['per_share']:.2f} | 12m ${dcf_12m:.2f} | P/E 20x 2032 EPS ${pe20['eps']:.2f} -> ${pe20['pv_at_target_date']:.2f} | 25x -> ${pe25['pv_at_target_date']:.2f} | blend ${blend:.2f}")
print("scenarios", {k: {kk: round(vv, 2) for kk, vv in v.items()} for k, v in scen.items()})
print("disc rate", {k: round(v, 2) for k, v in sens_r.items()})
print("melanoma peak", {k: round(v, 2) for k, v in sens_mel.items()})
print("overhead", {k: round(v, 2) for k, v in sens_overhead.items()})
print(f"intismeran global peak (all tumors) base ${base_int_peak/1000:.1f}B; needed for $189: risked x{k_risked:.1f} = ${base_int_peak*k_risked/1000:.0f}B, certain x{k_certain:.1f} = ${base_int_peak*k_certain/1000:.0f}B")
print("risked op income", {y: round(v) for y, v in out["risked_op"].items()})
for x in sorted(base["rows"], key=lambda x: -x["rnpv"]):
    print(f"  {x['name'][:58]:58} {x['rnpv']:7,.0f} {x['rnpv']/m.DILUTED_SHARES:6.2f}")

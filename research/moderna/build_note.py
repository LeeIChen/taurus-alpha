"""Build the Moderna research note HTML from mrna_results.json (no hand-copied numbers)."""

import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

R = json.load(open(HERE / "mrna_results.json"))
P = R["price"]
SH = R["shares"]
rows = sorted(R["rows"], key=lambda x: -x["rnpv"])

bull = R["scenarios"]["Bull"]; base = R["scenarios"]["Base"]; bear = R["scenarios"]["Bear"]
blend = lambda s: 0.5 * s["dcf"] * 1.10 + 0.5 * s["pe20"]
TARGET = round(R["blend"])
BEAR, BULL = blend(bear), blend(bull)
downside = (TARGET / P - 1) * 100
mcap = R["market_cap"] / 1000
ev = (R["market_cap"] - R["base"]["net_cash"]) / 1000
unrisked_total = (sum(x["unrisked_npv"] for x in rows) + R["base"]["corporate"] + R["base"]["net_cash"] + R["base"]["litigation"]) / SH


def money(v, d=0):
    return f"${v:,.{d}f}"


def ps(v):
    s = f"{abs(v):.2f}"
    return f"−${s}" if v < 0 else f"${s}"


SHORT = {
    "Spikevax + mNEXSPIKE (COVID)": "COVID (Spikevax, mNEXSPIKE)",
    "mRESVIA (RSV)": "RSV (mRESVIA)",
    "mFLUSIVA (seasonal flu, adults 50+)": "Flu (mFLUSIVA)",
    "mCOMBRIAX (flu + COVID combo)": "Flu + COVID combo (mCOMBRIAX)",
    "mRNA-1403 (norovirus)": "Norovirus (mRNA-1403)",
    "mRNA-3927 (propionic acidemia)": "Propionic acidemia (mRNA-3927)",
    "mRNA-3705 (methylmalonic acidemia)": "MMA (mRNA-3705)",
    "Intismeran: adjuvant melanoma (INTerpath-001)": "Intismeran · adj. melanoma (001)",
    "Intismeran: adjuvant NSCLC stage II-IIIB (INTerpath-002)": "Intismeran · adj. NSCLC II–IIIB (002)",
    "Intismeran: NSCLC non-pCR post-neoadjuvant (INTerpath-009)": "Intismeran · NSCLC non-pCR (009)",
    "Intismeran: high-risk stage I NSCLC (INTerpath-014)": "Intismeran · stage I NSCLC (014)",
    "Intismeran: adjuvant RCC (INTerpath-004)": "Intismeran · adj. kidney (004)",
    "Intismeran: muscle-invasive bladder (INTerpath-005)": "Intismeran · MIBC bladder (005)",
    "Intismeran: high-risk NMIBC (INTerpath-011)": "Intismeran · NMIBC bladder (011)",
    "Intismeran: 1L advanced melanoma (INTerpath-012)": "Intismeran · 1L melanoma (012)",
    "Intismeran: 1L metastatic squamous NSCLC (INTerpath-013)": "Intismeran · 1L sq. NSCLC (013)",
    "mRNA-4359 (checkpoint cancer therapy)": "mRNA-4359 (checkpoint)",
    "Platform & corporate overhead": "Platform & corporate overhead",
    "Net cash (after $3.0B convert, $0.6B loan)": "Net cash (pro forma)",
    "Arbutus/Genevant contingent payment (35% × $1.3B)": "Patent contingency (35% × $1.3B)",
}

# ---------------- chart 1: per-share contribution (diverging bars)
items = [(x["name"], x["rnpv"] / SH, x["group"]) for x in rows]
items += [("Platform & corporate overhead", R["base"]["corporate"] / SH, "Balance sheet & overhead"),
          ("Net cash (after $3.0B convert, $0.6B loan)", R["base"]["net_cash"] / SH, "Balance sheet & overhead"),
          ("Arbutus/Genevant contingent payment (35% × $1.3B)", R["base"]["litigation"] / SH, "Balance sheet & overhead")]
items.sort(key=lambda t: -t[1])
lo, hi = min(v for _, v, _ in items), max(v for _, v, _ in items)
lo, hi = -20, 12
W, row_h, label_w, pad_r = 780, 26, 260, 56
plot_w = W - label_w - pad_r
x0 = label_w + plot_w * (-lo) / (hi - lo)
sx = lambda v: plot_w * v / (hi - lo)
H = row_h * len(items) + 44
svg = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="c1t" class="chart"><title id="c1t">Contribution of each program to value per share</title>']
for t in range(lo, hi + 1, 4):
    x = x0 + sx(t)
    svg.append(f'<line x1="{x:.1f}" y1="8" x2="{x:.1f}" y2="{H-28}" class="grid"/>')
    svg.append(f'<text x="{x:.1f}" y="{H-10}" class="tick" text-anchor="middle">{ps(t) if t else "$0"}</text>')
for i, (name, v, grp) in enumerate(items):
    y = 10 + i * row_h
    w = abs(sx(v))
    x = x0 if v >= 0 else x0 - w
    cls = "pos" if v >= 0 else "neg"
    svg.append(f'<g class="bar" tabindex="0"><title>{html.escape(name)}: {ps(v)} per share ({grp})</title>'
               f'<rect x="{label_w-8}" y="{y-3}" width="{W-label_w+8}" height="{row_h}" class="hit"/>'
               f'<text x="{label_w-12}" y="{y+13}" class="lbl" text-anchor="end">{html.escape(SHORT[name])}</text>'
               f'<rect x="{x:.1f}" y="{y+2}" width="{max(w,1.5):.1f}" height="{row_h-8}" rx="3" class="{cls}"/>'
               f'<text x="{(x0+sx(v)+6) if v>=0 else (x0+sx(v)-6):.1f}" y="{y+13}" class="val" text-anchor="{"start" if v>=0 else "end"}">{ps(v)}</text></g>')
svg.append(f'<line x1="{x0:.1f}" y1="6" x2="{x0:.1f}" y2="{H-28}" class="zero"/></svg>')
chart1 = "\n".join(svg)

# ---------------- chart 2: value range vs market price (dot plot)
W2, H2, L2, R2 = 760, 210, 150, 30
pw = W2 - L2 - R2
s2 = lambda v: L2 + pw * v / 200
marks = [("Our bear case", BEAR, "ours"), ("Our 12-month target", TARGET, "target"), ("Our bull case", BULL, "ours"),
         ("Citi (Sell)", 80, "street"), ("Street consensus", 120.67, "street"), ("Street high", 135, "street")]
svg2 = [f'<svg viewBox="0 0 {W2} {H2}" role="img" aria-labelledby="c2t" class="chart"><title id="c2t">Our value range and Street targets versus the share price</title>']
for t in range(0, 201, 25):
    svg2.append(f'<line x1="{s2(t):.1f}" y1="14" x2="{s2(t):.1f}" y2="{H2-30}" class="grid"/><text x="{s2(t):.1f}" y="{H2-12}" class="tick" text-anchor="middle">${t}</text>')
svg2.append(f'<text x="{L2-12}" y="52" class="lbl" text-anchor="end">This note</text><text x="{L2-12}" y="122" class="lbl" text-anchor="end">Sell-side</text>')
svg2.append(f'<line x1="{s2(BEAR):.1f}" y1="47" x2="{s2(BULL):.1f}" y2="47" class="range"/>')
for name, v, kind in marks:
    y = 47 if kind in ("ours", "target") else 117
    r = 8 if kind == "target" else 6
    svg2.append(f'<g class="dot" tabindex="0"><title>{name}: ${v:,.0f}</title><circle cx="{s2(v):.1f}" cy="{y}" r="{r}" class="{kind}"/>'
                f'<text x="{s2(v):.1f}" y="{y-14}" class="val" text-anchor="middle">${v:,.0f}</text>'
                f'<text x="{s2(v):.1f}" y="{y+24}" class="cap" text-anchor="middle">{name.replace("Our ","").replace("Street ","")}</text></g>')
svg2.append(f'<line x1="{s2(P):.1f}" y1="10" x2="{s2(P):.1f}" y2="{H2-30}" class="price"/>'
            f'<text x="{s2(P)-6:.1f}" y="24" class="pricelbl" text-anchor="end">Price ${P:.2f}</text></svg>')
chart2 = "\n".join(svg2)

# ---------------- tables
def prog_rows():
    out = []
    for x in rows:
        share = "50% (Merck)" if x["share"] < 1 else "100%"
        peak = "—" if x["peak_sales"] == 0 else money(x["peak_sales"] / 1000, 1) + "B"
        out.append(f'<tr><td class="name">{html.escape(x["name"])}</td><td>{html.escape(x["stage"])}</td>'
                   f'<td class="n">{x["pos"]:.0%}</td><td class="n">{x["launch"]}</td><td class="n">{peak}</td><td>{share}</td>'
                   f'<td class="n">{money(x["rnpv"])}</td><td class="n strong">{ps(x["rnpv"]/SH)}</td><td class="n muted">{ps(x["unrisked_npv"]/SH)}</td></tr>')
    return "\n".join(out)


def pos_rows():
    return "\n".join(f'<tr><td class="name">{html.escape(x["name"])}</td><td class="n">{x["pos"]:.0%}</td><td>{html.escape(x["pos_basis"])}</td></tr>'
                     for x in sorted(R["rows"], key=lambda x: -x["pos"]))


sens_r = "".join(f'<td class="n{" strong" if k=="10%" else ""}">{ps(v)}</td>' for k, v in R["sens_r"].items())
sens_r_h = "".join(f"<th>{k}</th>" for k in R["sens_r"])
sens_m = "".join(f'<td class="n{" strong" if k=="$2.5B" else ""}">{ps(v)}</td>' for k, v in R["sens_mel"].items())
sens_m_h = "".join(f"<th>{k}</th>" for k in R["sens_mel"])
sens_o = "".join(f'<td class="n{" strong" if k=="$550M" else ""}">{ps(v)}</td>' for k, v in R["sens_overhead"].items())
sens_o_h = "".join(f"<th>{k}/yr</th>" for k in R["sens_overhead"])
imp = R["implied"]

DATA = dict(bullgap=f"{(1-BULL/P)*100:.0f}", TARGET=TARGET, P=P, downside=downside, BEAR=BEAR, BULL=BULL, mcap=mcap, ev=ev,
            dcf=R["base"]["per_share"], dcf12=R["dcf_12m"], pe_eps=R["pe20"]["eps"], pe20=R["pe20"]["pv_at_target_date"],
            pe25=R["pe25"]["pv_at_target_date"], unrisked=unrisked_total,
            int_base=imp["base_int_peak_global"] / 1000, int_need=imp["int_peak_needed_certain"] / 1000,
            corp=R["base"]["corporate"] / SH, cash=R["base"]["net_cash"] / SH, lit=R["base"]["litigation"] / SH,
            op28=R["risked_op"]["2028"], op30=R["risked_op"]["2030"], op32=R["risked_op"]["2032"],
            bear_dcf=bear["dcf"], bear_pe=bear["pe20"], base_dcf=base["dcf"], base_pe=base["pe20"], bull_dcf=bull["dcf"], bull_pe=bull["pe20"],
            bear_eps=bear["eps2032"], base_eps=base["eps2032"], bull_eps=bull["eps2032"])

tpl = open(HERE / "note_template.html").read()
for k, v in DATA.items():
    if isinstance(v, float):
        tpl = tpl.replace("{{" + k + "}}", f"{v:,.2f}")
        tpl = tpl.replace("{{" + k + ":0}}", f"{v:,.0f}")
        tpl = tpl.replace("{{" + k + ":1}}", f"{v:,.1f}")
    else:
        tpl = tpl.replace("{{" + k + "}}", str(v))
tpl = (tpl.replace("{{CHART1}}", chart1).replace("{{CHART2}}", chart2).replace("{{PROG_ROWS}}", prog_rows())
       .replace("{{POS_ROWS}}", pos_rows()).replace("{{SENS_R}}", sens_r).replace("{{SENS_R_H}}", sens_r_h)
       .replace("{{SENS_M}}", sens_m).replace("{{SENS_M_H}}", sens_m_h).replace("{{SENS_O}}", sens_o).replace("{{SENS_O_H}}", sens_o_h))
assert "{{" not in tpl, [l for l in tpl.splitlines() if "{{" in l][:5]
open(HERE / "mrna_note.html", "w").write(tpl)
print("written", len(tpl), "chars; target", TARGET, "bear", round(BEAR, 1), "bull", round(BULL, 1), "downside", round(downside, 1))

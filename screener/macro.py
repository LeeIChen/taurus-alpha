"""Macro regime for the screener: a composite leading index, rate direction, and monthly series
used to measure each stock's sensitivity to rates and to the cycle. No model calls.

ISM PMI is not on FRED and the OECD leading indicator stopped in 2024, so the leading index is
built from Conference Board LEI components that FRED carries: average weekly manufacturing hours,
initial jobless claims (inverted), building permits, new orders for nondefense capital goods,
the 10-year minus 3-month yield spread, and consumer sentiment. Each component's 6-month change
(the spread enters as a level) is z-scored over its history since 2000; the composite is their mean.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

from core.tools.fred import _fetch

# series id -> (label, transform): "log6" = 6-month log change, "diff6" = 6-month difference,
# "level" = monthly average; sign -1 inverts (more jobless claims is a weaker signal)
LEI_COMPONENTS = {
    "AWHMAN": ("Average weekly hours, manufacturing", "diff6", 1),
    "ICSA": ("Initial jobless claims", "log6", -1),
    "PERMIT": ("Building permits", "log6", 1),
    "NEWORDER": ("New orders, nondefense capital goods ex aircraft", "log6", 1),
    "T10Y3M": ("10-year minus 3-month Treasury spread", "level", 1),
    "UMCSENT": ("Consumer sentiment (University of Michigan)", "log6", 1),
}
HISTORY_FROM = date(2000, 1, 1)


def monthly(series_id: str) -> Dict[str, float]:
    """{"YYYY-MM": monthly average} for a FRED series of any frequency."""
    sums: Dict[str, List[float]] = defaultdict(list)
    for d, v in _fetch(series_id):
        if d >= HISTORY_FROM:
            sums[d.strftime("%Y-%m")].append(v)
    return {k: mean(v) for k, v in sorted(sums.items())}


def _shift(month: str, n: int) -> str:
    y, m = int(month[:4]), int(month[5:7])
    t = y * 12 + (m - 1) + n
    return f"{t // 12:04d}-{t % 12 + 1:02d}"


def _transform(series: Dict[str, float], how: str) -> Dict[str, float]:
    out = {}
    for k, v in series.items():
        if how == "level":
            out[k] = v
            continue
        prev = series.get(_shift(k, -6))
        if prev is None:
            continue
        if how == "log6":
            if v > 0 and prev > 0:
                out[k] = math.log(v / prev)
        else:
            out[k] = v - prev
    return out


def leading_index() -> Dict[str, float]:
    """{"YYYY-MM": composite z-score}; months where at least 4 of 6 components are in."""
    zs: Dict[str, List[float]] = defaultdict(list)
    for sid, (_, how, sign) in LEI_COMPONENTS.items():
        t = _transform(monthly(sid), how)
        if len(t) < 24:
            continue
        mu, sd = mean(t.values()), pstdev(t.values()) or 1.0
        for k, v in t.items():
            zs[k].append(sign * (v - mu) / sd)
    return {k: mean(v) for k, v in sorted(zs.items()) if len(v) >= 4}


def _change(series: Dict[str, float], months: int) -> Tuple[float, float, float]:
    last = max(series)
    before = series.get(_shift(last, -months))
    if before is None:
        before = series[min(series, key=lambda k: abs((int(k[:4]) * 12 + int(k[5:])) - (int(last[:4]) * 12 + int(last[5:]) - months)))]
    return series[last], before, series[last] - before


def regime() -> Dict[str, Any]:
    """Where the cycle and rates are now, in words the wizard can show, plus the numbers behind it."""
    lei = leading_index()
    last = max(lei)
    lei_now, lei_3m, lei_d = _change(lei, 3)
    if lei_now >= 0:
        phase = "擴張" if lei_d >= 0 else "高峰放緩"
    else:
        phase = "復甦" if lei_d >= 0 else "收縮"
    y_now, y_6m, y_d = _change(monthly("DGS10"), 6)
    f_now, f_6m, f_d = _change(monthly("FEDFUNDS"), 6)
    rates = "上升" if y_d >= 0.25 else ("下降" if y_d <= -0.25 else "持平")
    policy = "升息" if f_d >= 0.2 else ("降息" if f_d <= -0.2 else "按兵不動")
    hints = {
        "擴張": "領先指標在零以上且仍在上升：景氣循環股（高敏感度）通常表現較好，但要留意估值。",
        "高峰放緩": "領先指標在零以上但開始下降：常見於循環末段，防禦型與高品質公司較能抵抗獲利下修。",
        "收縮": "領先指標在零以下且下降：偏好防禦型、低負債公司；循環股的獲利還可能下修。",
        "復甦": "領先指標在零以下但已回升：歷史上循環股與小型股常在這個階段領先。",
    }
    rate_hint = {
        "上升": "長天期利率上升：對利率敏感（升息受害）的產業如公用事業、REITs、高估值成長股承壓；銀行與保險常受惠。",
        "下降": "長天期利率下降：利率敏感（升息受害）的產業與長存續期的成長股通常受惠。",
        "持平": "長天期利率大致持平：利率敏感度對這段期間的報酬影響較小。",
    }
    components = []
    for sid, (label, how, sign) in LEI_COMPONENTS.items():
        m = monthly(sid)
        if m:
            k = max(m)
            components.append({"series_id": sid, "label": label, "month": k, "value": round(m[k], 3),
                               "six_months_ago": round(m[_shift(k, -6)], 3) if _shift(k, -6) in m else None})
    return {
        "as_of": last, "phase": phase, "phase_hint": hints[phase],
        "leading_index": round(lei_now, 2), "leading_index_3m_ago": round(lei_3m, 2),
        "ten_year": round(y_now, 2), "ten_year_6m_ago": round(y_6m, 2), "rates": rates, "rate_hint": rate_hint[rates],
        "fed_funds": round(f_now, 2), "fed_funds_6m_ago": round(f_6m, 2), "policy": policy,
        "leading_series": {k: round(v, 3) for k, v in list(lei.items())[-36:]},
        "components": components,
    }


def annual_growth(series_id: str = "INDPRO") -> Dict[int, float]:
    """{year: annual average level}, the driver for revenue elasticity (industrial production)."""
    by_year: Dict[int, List[float]] = defaultdict(list)
    for k, v in monthly(series_id).items():
        by_year[int(k[:4])].append(v)
    return {y: mean(v) for y, v in by_year.items() if len(v) == 12}


def monthly_factors() -> Dict[str, Dict[str, float]]:
    """Monthly regressors for stock sensitivities: change in the 10-year yield (pp) and the leading index."""
    y10 = monthly("DGS10")
    return {"d_ten_year": {k: v - y10[_shift(k, -1)] for k, v in y10.items() if _shift(k, -1) in y10},
            "leading_index": leading_index()}

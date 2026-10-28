"""Macro series from FRED (Federal Reserve Bank of St. Louis), no model calls.

Uses the public CSV download (fred.stlouisfed.org/graph/fredgraph.csv), which needs no API key.
Responses are cached in data/fred/<SERIES>.csv for 12 hours.
"""

from __future__ import annotations

import csv
import io
import logging
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("uvicorn.error")

ROOT = Path(__file__).resolve().parent.parent.parent
FRED_DIR = ROOT / "data" / "fred"
CACHE_HOURS = 12

# Always fetched: rates, inflation, jobs, sentiment, real consumer spending.
CORE_SERIES = {
    "FEDFUNDS": "Effective federal funds rate (%)",
    "DGS10": "10-year Treasury yield (%)",
    "CPIAUCSL": "CPI, all items (index; YoY shown)",
    "UNRATE": "Unemployment rate (%)",
    "UMCSENT": "University of Michigan consumer sentiment (index)",
    "PCEC96": "Real personal consumption expenditures ($bn, chained 2017)",
}
# The planner can add sector series by id; these get readable labels.
KNOWN_SERIES = {
    **CORE_SERIES,
    "MORTGAGE30US": "30-year fixed mortgage rate (%)",
    "EXHOSLUSM495S": "Existing home sales (annual rate, units)",
    "HOUST": "Housing starts (thousands, annual rate)",
    "RSFHFS": "Retail sales: furniture and home furnishing stores ($M)",
    "PCEDG": "PCE: durable goods ($bn, nominal)",
    "DGORDER": "Manufacturers' new orders: durable goods ($M)",
    "RSXFS": "Retail sales excluding food services ($M)",
    "ECOMPCTSA": "E-commerce share of total retail sales (%)",
    "INDPRO": "Industrial production (index)",
    "DCOILWTICO": "WTI crude oil ($/barrel)",
    "T10Y2Y": "10-year minus 2-year Treasury spread (pp)",
    "BAMLH0A0HYM2": "High-yield credit spread (pp)",
}
# Index levels are easier to read as year-on-year change.
YOY_SERIES = {"CPIAUCSL", "PCEC96", "PCEDG", "RSFHFS", "RSXFS", "DGORDER", "INDPRO"}


def _fetch(series_id: str) -> List[Tuple[date, float]]:
    FRED_DIR.mkdir(parents=True, exist_ok=True)
    cache = FRED_DIR / f"{series_id}.csv"
    if not cache.exists() or time.time() - cache.stat().st_mtime > CACHE_HOURS * 3600:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "taurus-alpha"}), timeout=30) as r:
                cache.write_bytes(r.read())
        except OSError as exc:
            if not cache.exists():
                raise
            logger.warning("FRED %s refresh failed (%s); using cached copy", series_id, exc)
    rows = list(csv.reader(io.StringIO(cache.read_text())))
    out = []
    for row in rows[1:]:
        try:
            out.append((date.fromisoformat(row[0]), float(row[1])))
        except (ValueError, IndexError):  # "." marks a missing observation
            continue
    return out


def _at_or_before(points: List[Tuple[date, float]], day: date) -> Optional[Tuple[date, float]]:
    best = None
    for d, v in points:
        if d > day:
            break
        best = (d, v)
    return best


def series_summary(series_id: str) -> Dict[str, Any]:
    """Latest value with its change over about 3 and 12 months (YoY % for index-level series)."""
    points = _fetch(series_id)
    if not points:
        raise ValueError(f"FRED series {series_id} has no data")
    last_d, last = points[-1]
    ago3, ago12 = _at_or_before(points, last_d - timedelta(days=91)), _at_or_before(points, last_d - timedelta(days=365))
    out = {"series_id": series_id, "label": KNOWN_SERIES.get(series_id, series_id), "date": last_d.isoformat(),
           "latest": last, "value_3m_ago": ago3[1] if ago3 else None, "value_12m_ago": ago12[1] if ago12 else None,
           "url": f"https://fred.stlouisfed.org/series/{series_id}"}
    if series_id in YOY_SERIES and ago12:
        ago24 = _at_or_before(points, last_d - timedelta(days=730))
        out["yoy_pct"] = (last / ago12[1] - 1) * 100
        out["yoy_pct_12m_ago"] = (ago12[1] / ago24[1] - 1) * 100 if ago24 else None
    return out


def macro_snapshot(extra: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rows = []
    for sid in dict.fromkeys([*CORE_SERIES, *(s.strip().upper() for s in extra or [] if s.strip())]):
        try:
            rows.append(series_summary(sid))
        except Exception as exc:  # an unknown id from the planner must not sink the step
            logger.warning("FRED %s failed: %s", sid, exc)
            rows.append({"series_id": sid, "error": f"{type(exc).__name__}: {exc}"})
    return rows


def render_table(rows: List[Dict[str, Any]]) -> str:
    def f(v: Optional[float]) -> str:
        return "n/a" if v is None else (f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}")
    ok = [r for r in rows if "error" not in r]
    lines = ["| Series | Latest (date) | 3 months earlier | 12 months earlier | YoY |", "|---|---|---|---|---|"]
    for r in ok:
        yoy = f"{r['yoy_pct']:+.1f}% (a year earlier {r['yoy_pct_12m_ago']:+.1f}%)" if r.get("yoy_pct") is not None and r.get("yoy_pct_12m_ago") is not None \
            else (f"{r['yoy_pct']:+.1f}%" if r.get("yoy_pct") is not None else "")
        lines.append(f"| {r['label']} [{r['series_id']}] | {f(r['latest'])} ({r['date']}) | {f(r['value_3m_ago'])} | "
                     f"{f(r['value_12m_ago'])} | {yoy} |")
    failed = [r["series_id"] for r in rows if "error" in r]
    return "\n".join(lines) + (f"\n\nUnavailable series: {', '.join(failed)}" if failed else "")

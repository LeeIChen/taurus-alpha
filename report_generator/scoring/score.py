"""Score saved research runs against what actually happened, and fold in human reviews.

Automatic (needs only prices):
- Return and excess return vs SPY at 30, 90, 180 and 365 days after the run.
- Direction check per completed horizon: Buy is right if the stock beat SPY, Sell if it lagged,
  Hold if it stayed within ±10% of SPY.
- 12-month target error: (price after 365 days - target) / target.

Needs human review (report_generator/scoring/reviews.py):
- Catalyst outcomes once their date has passed.
- Probability-of-success calibration (Brier score) once a program's outcome is known.
- Analyst quality scores and pairwise preferences.

Usage: python -m report_generator.scoring.score      (prints a table, writes runs/scores.json)
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from report_generator.run_log import RUNS_DIR
from core.tools import prices
from report_generator.scoring.reviews import load_reviews

HORIZONS = (30, 90, 180, 365)
BENCHMARK = "SPY"
HOLD_BAND = 10.0  # Hold is right if the stock ends within ±10% of the benchmark
_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}


def load_runs() -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(RUNS_DIR.glob("*.json")):
        if path.name == "scores.json":
            continue
        record = json.loads(path.read_text())
        if record.get("status") != "succeeded":
            continue
        record["run_id"] = path.stem
        runs.append(record)
    return runs


def run_summary(record: Dict[str, Any]) -> Dict[str, Any]:
    """Rating, target, ticker and price at the time of the run, tolerant of older record formats."""
    r, req = record["result"], record["request"]
    thesis = r.get("thesis") or {}
    target = thesis.get("price_target")
    for key in ("pipeline_valuation", "valuation"):
        if target is None and r.get(key):
            target = r[key].get("target_price")
    ticker = r.get("ticker") or req.get("ticker")
    if not ticker:
        from core.tools.sec import resolve_company  # older runs stored only the company name
        try:
            ticker = resolve_company(req["company_name"])["ticker"]
        except Exception:  # resolution is best-effort for legacy records
            ticker = None
    return {
        "run_id": record["run_id"],
        "company": req.get("company_name"),
        "ticker": ticker,
        "run_date": record["started_at"][:10],
        "rating": thesis.get("rating"),
        "target": target,
        "price_at_run": r.get("current_price") or thesis.get("current_price"),
        "catalysts": thesis.get("catalysts") or [],
        "programs": [{"name": p["name"], "probability_of_success": p["probability_of_success"], "stage": p["stage"]}
                     for p in (r.get("pipeline_valuation") or {}).get("programs", [])],
    }


def _direction_correct(rating: Optional[str], excess: float) -> Optional[bool]:
    if not rating:
        return None
    rating = rating.lower()
    if rating == "buy":
        return excess > 0
    if rating == "sell":
        return excess < 0
    if rating == "hold":
        return abs(excess) <= HOLD_BAND
    return None


def window_end(timing: str) -> Optional[date]:
    """Latest date a catalyst window can mean: 'Oct 24, 2026', 'Nov 2026', 'Q4 2026', 'H1 2027', '2027'."""
    t = timing.lower()
    years = [int(y) for y in re.findall(r"\b(20\d\d)\b", t)]
    if not years:
        return None
    year = max(years)
    m = re.search(r"\bq([1-4])\b", t)
    if m:
        return _month_end(year, int(m.group(1)) * 3)
    m = re.search(r"\bh([12])\b", t)
    if m:
        return date(year, 6, 30) if m.group(1) == "1" else date(year, 12, 31)
    months = [_MONTHS[w[:3]] for w in re.findall(r"\b([a-z]{3,9})\b", t) if w[:3] in _MONTHS]
    if months:
        month = max(months)
        days = [int(d) for d in re.findall(r"\b([0-3]?\d)\b", re.sub(r"\b20\d\d\b", "", t)) if 1 <= int(d) <= 31]
        return date(year, month, max(days)) if days else _month_end(year, month)
    return date(year, 12, 31)


def _month_end(year: int, month: int) -> date:
    return date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)


def score_run(summary: Dict[str, Any], today: date, bench: Dict[str, float]) -> Dict[str, Any]:
    out = dict(summary)
    run_day = date.fromisoformat(summary["run_date"])
    closes = prices.history(summary["ticker"]) if summary["ticker"] else {}
    start = prices.close_on_or_before(closes, run_day)
    bench_start = prices.close_on_or_before(bench, run_day)
    p0 = summary["price_at_run"] or (start[1] if start else None)
    out["price_at_run"] = p0
    out["horizons"] = {}
    if p0 and bench_start:
        for h in HORIZONS:
            day = run_day + timedelta(days=h)
            if day > today:
                out["horizons"][f"{h}d"] = {"status": "pending", "date": day.isoformat()}
                continue
            px, bx = prices.close_on_or_before(closes, day), prices.close_on_or_before(bench, day)
            if not px or not bx:
                out["horizons"][f"{h}d"] = {"status": "no price", "date": day.isoformat()}
                continue
            ret = (px[1] / p0 - 1) * 100
            excess = ret - (bx[1] / bench_start[1] - 1) * 100
            out["horizons"][f"{h}d"] = {"status": "done", "date": px[0], "price": px[1], "return_pct": round(ret, 1),
                                        "excess_pct": round(excess, 1),
                                        "direction_correct": _direction_correct(summary["rating"], excess)}
        now, bnow = prices.latest(closes), prices.latest(bench)
        if now and bnow:
            out["latest"] = {"date": now[0], "price": now[1], "return_pct": round((now[1] / p0 - 1) * 100, 1),
                             "excess_pct": round((now[1] / p0 - bnow[1] / bench_start[1]) * 100, 1)}
            if summary["target"] and summary["target"] != p0:
                out["latest"]["progress_to_target_pct"] = round((now[1] - p0) / (summary["target"] - p0) * 100, 1)
        y1 = out["horizons"].get("365d", {})
        if y1.get("status") == "done" and summary["target"]:
            out["target_error_pct"] = round((y1["price"] / summary["target"] - 1) * 100, 1)
    out["catalysts_due"] = [c for c in summary["catalysts"] if (end := window_end(c.get("timing", ""))) and end < today]
    return out


def brier(pairs: List[tuple]) -> Optional[float]:
    """Mean squared error of probabilities against 0/1 outcomes (lower is better; 0.25 = coin flip)."""
    return round(sum((p - o) ** 2 for p, o in pairs) / len(pairs), 4) if pairs else None


def score_all(today: Optional[date] = None) -> Dict[str, Any]:
    today = today or date.today()
    bench = prices.history(BENCHMARK)
    reviews = load_reviews()
    scored, pos_pairs = [], []
    for record in load_runs():
        s = score_run(run_summary(record), today, bench)
        rv = reviews.get(s["run_id"], [])
        s["reviews"] = len(rv)
        overall = [x["overall"] for x in rv if x.get("overall")]
        s["human_overall"] = round(sum(overall) / len(overall), 2) if overall else None
        for x in rv:  # program outcomes recorded by reviewers feed PoS calibration
            for name, outcome in (x.get("program_outcomes") or {}).items():
                p = next((pg["probability_of_success"] for pg in s["programs"] if pg["name"] == name), None)
                if p is not None and outcome in ("success", "failure"):
                    pos_pairs.append((p, 1.0 if outcome == "success" else 0.0))
        scored.append(s)

    summary: Dict[str, Any] = {"as_of": today.isoformat(), "runs": len(scored), "by_horizon": {}}
    for h in HORIZONS:
        done = [s["horizons"][f"{h}d"] for s in scored if s["horizons"].get(f"{h}d", {}).get("status") == "done"]
        judged = [d["direction_correct"] for d in done if d["direction_correct"] is not None]
        summary["by_horizon"][f"{h}d"] = {
            "completed": len(done),
            "hit_rate": round(sum(judged) / len(judged), 3) if judged else None,
            "avg_excess_pct": round(sum(d["excess_pct"] for d in done) / len(done), 1) if done else None,
        }
    summary["pos_brier"] = brier(pos_pairs)
    summary["pos_outcomes"] = len(pos_pairs)
    summary["human_reviews"] = sum(s["reviews"] for s in scored)
    return {"summary": summary, "runs": scored}


def main() -> None:
    result = score_all()
    (RUNS_DIR / "scores.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    s = result["summary"]
    print(f"Scored {s['runs']} runs as of {s['as_of']} (benchmark {BENCHMARK})\n")
    print(f"{'run':38} {'rating':6} {'target':>8} {'p0':>8} {'latest':>8} {'ret%':>7} {'vsSPY%':>7}  horizons      due  reviews")
    for r in result["runs"]:
        lt = r.get("latest", {})
        hz = " ".join("✓" if v.get("direction_correct") else "✗" if v.get("direction_correct") is False
                      else "·" for v in r["horizons"].values())
        fmt = lambda v: f"{v:8.2f}" if isinstance(v, (int, float)) else f"{'—':>8}"
        print(f"{r['run_id'][:38]:38} {str(r['rating'] or '—'):6} {fmt(r['target'])} {fmt(r['price_at_run'])} "
              f"{fmt(lt.get('price'))} {lt.get('return_pct', '—'):>7} {lt.get('excess_pct', '—'):>7}  {hz:12} "
              f"{len(r['catalysts_due']):>4} {r['reviews']:>8}")
    print()
    for h, v in s["by_horizon"].items():
        print(f"  {h:>4}: {v['completed']} completed, hit rate {v['hit_rate']}, avg excess {v['avg_excess_pct']}%")
    print(f"  PoS calibration (Brier): {s['pos_brier']} from {s['pos_outcomes']} recorded outcomes")
    print(f"  Human reviews: {s['human_reviews']}")


if __name__ == "__main__":
    main()

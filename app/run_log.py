"""Save every /research run to runs/ as one timestamped JSON file. Local disk only."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("uvicorn.error")

RUNS_DIR = Path(os.environ.get("TAURUS_RUNS_DIR", Path(__file__).resolve().parent.parent / "runs"))


def save_run(
    request: Dict[str, Any],
    started_at: datetime,
    duration_s: float,
    result: Optional[Dict[str, Any]] = None,
    failure: Optional[str] = None,
) -> Optional[Path]:
    """Write one run record. Never raises: a logging problem must not fail the request."""
    record = {
        "started_at": started_at.isoformat(),
        "duration_s": round(duration_s, 1),
        "status": "failed" if failure else "succeeded",
        "request": request,
        "failure": failure,
        "result": result,
    }
    slug = re.sub(r"[^a-z0-9]+", "-", str(request.get("company_name", "unknown")).lower()).strip("-")
    stem = f"{started_at.strftime('%Y-%m-%dT%H-%M-%SZ')}_{slug or 'unknown'}"
    path = RUNS_DIR / f"{stem}.json"
    try:
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
        n = 2
        while path.exists():  # same company started within the same second
            path = RUNS_DIR / f"{stem}-{n}.json"
            n += 1
        path.write_text(json.dumps(record, indent=2, default=str) + "\n")
        if result:
            path.with_suffix(".md").write_text(render_markdown(record))
    except OSError as exc:
        logger.warning("Could not save run to %s: %s", path, exc)
        return None
    logger.info("Saved run to %s", path)
    return path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def render_markdown(record: Dict[str, Any]) -> str:
    """Readable companion to the JSON record: tasks, thesis, catalysts, valuation, memo, sources."""
    r = record["result"] or {}
    req = record["request"]
    t = r.get("thesis") or {}
    lines = [f"# {req.get('company_name')} ({r.get('ticker') or req.get('ticker') or ''})",
             "", f"Question: {req.get('query')}", f"Run started {record['started_at']} · {record['duration_s']}s",
             ""]
    if t:
        price = t.get("current_price")
        lines += [f"**Rating: {t.get('rating')}** · Price target ${t.get('price_target'):,.2f} · "
                  f"Price {f'${price:,.2f}' if price else 'n/a'} ({t.get('current_price_source')}) · "
                  f"{t.get('horizon_months')}-month horizon", "", t.get("summary", ""), ""]
    plan = r.get("plan") or {}
    lines += ["## Research tasks", ""] + [f"{i}. {x}" for i, x in enumerate(plan.get("tasks", []), 1)]
    lines += ["", "Metrics: " + ", ".join(plan.get("required_metrics", [])), ""]
    if t:
        lines += ["## Investment thesis", ""] + [f"- {x}" for x in t.get("thesis_points", [])]
        lines += ["", "## Catalysts", "", "| Timing | Event | Direction | Expected impact |", "|---|---|---|---|"]
        lines += [f"| {c['timing']} | {c['event']} | {c['direction']} | {c['expected_impact']} |" for c in t.get("catalysts", [])]
        lines += ["", "## Risks to the thesis", ""] + [f"- {x}" for x in t.get("risks_to_thesis", [])]
        lines += ["", "## What would change our view", ""] + [f"- {x}" for x in t.get("what_would_change_our_view", [])]
    pv = r.get("pipeline_valuation")
    if pv:
        lines += ["", "## Sum-of-the-parts valuation", "",
                  f"DCF ${pv['dcf_value_per_share']:,.2f} · P/E ${pv['pe_target_price']:,.2f} (EPS {pv['pe_eps']}) · "
                  f"target ${pv['target_price']:,.2f}", "", pv.get("implied_note", ""), "",
                  "| Program | Stage | PoS | rNPV ($M) | Per share |", "|---|---|---|---|---|"]
        lines += [f"| {p['name']} | {p['stage']} | {p['probability_of_success']:.0%} | {p['rnpv']:,.0f} | ${p['per_share']:,.2f} |"
                  for p in sorted(pv["programs"], key=lambda p: -p["rnpv"])]
    elif r.get("valuation"):
        v = r["valuation"]
        lines += ["", "## Valuation", "", f"DCF ${v['dcf_value_per_share']:,.2f} · P/E ${v['pe_target_price']:,.2f} · "
                  f"target ${v['target_price']:,.2f}"]
    lines += ["", f"Faithfulness score: {r.get('faithfulness_score', 0):.2f}"]
    if r.get("error"):
        lines += ["", "Warnings:", "", "```", r["error"], "```"]
    lines += ["", "---", "", r.get("report", ""), "", "## All sources", ""]
    lines += [f"- [{s['source_id']}] {s['title']} — {s['url']}" + (" (cited)" if s.get("cited") else "")
              for s in r.get("sources", [])]
    return "\n".join(lines) + "\n"

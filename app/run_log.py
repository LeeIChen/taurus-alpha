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
    checkpoint: Optional[Dict[str, Any]] = None,
    resumed_from: Optional[str] = None,
    cost: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Write one run record. Never raises: a logging problem must not fail the request.

    A failed run keeps `checkpoint`, the state after its last completed step, for resuming."""
    record = {
        "started_at": started_at.isoformat(),
        "duration_s": round(duration_s, 1),
        "status": "failed" if failure else "succeeded",
        "request": request,
        "failure": failure,
        "result": result,
    }
    if resumed_from:
        record["resumed_from"] = resumed_from
    if checkpoint:
        record["checkpoint"] = checkpoint
    if cost:  # a failed run records what it spent; a successful one has it in result.cost
        record["cost"] = cost
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
    """Two layers: the memo as the reader sees it, then an appendix with how the run went.

    Thesis, catalysts and valuation are already in the memo; the structured copies stay in the JSON."""
    r = record["result"] or {}
    req = record["request"]
    lines = [r.get("report", "").strip(), "", "---", "", "## Appendix: run details", "",
             f"- Question: {req.get('query')}",
             f"- Run started {record['started_at']}, took {record['duration_s']}s",
             f"- Faithfulness score: {r.get('faithfulness_score', 0):.2f}",
             f"- Editor revisions: {r.get('revisions', 0)}"]
    cost = r.get("cost") or {}
    if cost.get("by_step"):
        lines.append(f"- Claude API cost: ${cost['total_usd']:.2f}")
        lines += ["", "### Cost by step", "", "| Step | Calls | Input | Cache read | Cache write | Output | Searches | USD |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {k} | {v['calls']} | {v['input']:,} | {v['cache_read']:,} | {v['cache_write']:,} | {v['output']:,} | "
                  f"{v['web_searches']} | ${v['usd']:.2f} |" for k, v in cost["by_step"].items()]
    if r.get("edit_notes"):
        lines += ["", "### Editor notes still open", ""] + [f"- {n}" for n in r["edit_notes"]]
    if r.get("error"):
        lines += ["", "### Verifier and pipeline warnings", ""]
        for block in r["error"].splitlines():
            if block.startswith("Unsupported claims: "):  # the verifier joins its claims with " | "
                lines += ["- Unsupported claims:"]
                lines += [f"  - {c}" for c in block.removeprefix("Unsupported claims: ").split(" | ")]
            else:
                lines.append(f"- {block}")
    plan = r.get("plan") or {}
    if plan.get("tasks"):
        lines += ["", "### Research plan", ""] + [f"{i}. {x}" for i, x in enumerate(plan["tasks"], 1)]
    return "\n".join(lines) + "\n"

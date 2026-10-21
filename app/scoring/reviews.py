"""Human feedback on research runs: reviews, pairwise preferences, and export for DPO training.

Stored as plain JSON under runs/reviews/ so a person can read, edit or audit them:
- <run_id>.json      list of reviews of one run (scores, corrections, outcomes, errors found)
- preferences.jsonl  one line per pairwise comparison (which of two runs is better, and why)

export_dpo() turns preferences into {prompt, chosen, rejected} rows, the standard input for
Direct Preference Optimization (a simpler form of RLHF) on an open-weight model with LoRA.
Claude itself can't be trained this way; for Claude the same data guides prompts and
best-of-n selection.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.run_log import RUNS_DIR

REVIEW_DIR = RUNS_DIR / "reviews"
SECTIONS = ("research_and_sources", "valuation_assumptions", "thesis_and_catalysts", "memo_clarity")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_reviews() -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for path in sorted(REVIEW_DIR.glob("*.json")):
        out[path.stem] = json.loads(path.read_text())
    return out


def add_review(run_id: str, review: Dict[str, Any]) -> Dict[str, Any]:
    """Append one review. Expected keys (all optional except reviewer):
    reviewer, overall (1-5), sections {name: {score 1-5, comment}}, factual_errors [str],
    program_corrections {program: {probability_of_success, peak_sales, comment}},
    program_outcomes {program: "success"|"failure"|"pending"},
    catalyst_outcomes {event: "as expected"|"better"|"worse"|"delayed"|"did not happen"},
    comment."""
    if not (RUNS_DIR / f"{run_id}.json").exists():
        raise FileNotFoundError(run_id)
    if not review.get("reviewer"):
        raise ValueError("reviewer is required")
    if review.get("overall") is not None and not 1 <= int(review["overall"]) <= 5:
        raise ValueError("overall must be 1-5")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    path = REVIEW_DIR / f"{run_id}.json"
    existing = json.loads(path.read_text()) if path.exists() else []
    entry = {**review, "created_at": _now()}
    existing.append(entry)
    path.write_text(json.dumps(existing, indent=2) + "\n")
    return entry


def add_preference(run_a: str, run_b: str, winner: str, reviewer: str, reason: str = "") -> Dict[str, Any]:
    """Record which of two runs answering the same kind of question is better ("a", "b" or "tie")."""
    for run_id in (run_a, run_b):
        if not (RUNS_DIR / f"{run_id}.json").exists():
            raise FileNotFoundError(run_id)
    if winner not in ("a", "b", "tie"):
        raise ValueError("winner must be a, b or tie")
    if not reviewer:
        raise ValueError("reviewer is required")
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    entry = {"run_a": run_a, "run_b": run_b, "winner": winner, "reviewer": reviewer, "reason": reason,
             "created_at": _now()}
    with (REVIEW_DIR / "preferences.jsonl").open("a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def load_preferences() -> List[Dict[str, Any]]:
    path = REVIEW_DIR / "preferences.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def _answer(run_id: str) -> Optional[Dict[str, Any]]:
    record = json.loads((RUNS_DIR / f"{run_id}.json").read_text())
    r = record.get("result") or {}
    return {"request": record["request"], "text": r.get("report", ""), "thesis": r.get("thesis")}


def export_dpo(path: Optional[Path] = None) -> Path:
    """Write preference pairs as JSONL rows {prompt, chosen, rejected, meta}; ties are skipped."""
    path = path or REVIEW_DIR / "dpo_pairs.jsonl"
    rows = []
    for p in load_preferences():
        if p["winner"] == "tie":
            continue
        a, b = _answer(p["run_a"]), _answer(p["run_b"])
        chosen, rejected = (a, b) if p["winner"] == "a" else (b, a)
        req = chosen["request"]
        rows.append({
            "prompt": f"Company: {req.get('company_name')}\nQuestion: {req.get('query')}",
            "chosen": chosen["text"],
            "rejected": rejected["text"],
            "meta": {"run_a": p["run_a"], "run_b": p["run_b"], "winner": p["winner"],
                     "reviewer": p["reviewer"], "reason": p["reason"]},
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path

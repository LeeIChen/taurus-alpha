"""HTTP endpoints for the screener wizard (GET /screener serves the page).

POST /screen                      start a session -> first step's payload
GET  /screen/{sid}                the current step's payload (to resume a session)
POST /screen/{sid}/answer         answer the current step -> next payload
POST /screen/{sid}/report         generate the full research report for one shortlisted company (paid, background)
GET  /screen/jobs/{job_id}        report job status and result
GET  /screen/status               dataset age, missing industry profiles and their estimated cost
POST /screen/profiles             profile missing industries with Claude (paid; needs confirm=true)
POST /screen/tam/{sub_industry}   check one industry's TAM on the web (paid; needs confirm=true)
GET  /screen/export/{kind}        training data: pairs (reward model) or labels (labeler)
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from langgraph.types import Command
from pydantic import BaseModel

from core.llm import start_meter, summarize_costs
from screener import preferences, profiles
from screener.features import load_dataset
from screener.graph import get_graph

logger = logging.getLogger("uvicorn.error")
router = APIRouter(tags=["screener"])
PAGE = Path(__file__).resolve().parent / "static" / "screener.html"
REPORT_EST_USD = 3.6

_session_locks: Dict[str, threading.Lock] = {}
_jobs: Dict[str, Dict[str, Any]] = {}


def _lock(sid: str) -> threading.Lock:
    return _session_locks.setdefault(sid, threading.Lock())


def _config(sid: str) -> dict:
    return {"configurable": {"thread_id": sid}}


def _payload(sid: str, request_cost: float = 0.0) -> Dict[str, Any]:
    snap = get_graph().get_state(_config(sid))
    if not snap or not snap.values and not snap.next:
        raise HTTPException(status_code=404, detail=f"No screener session {sid}")
    for task in snap.tasks:
        if task.interrupts:
            return {"session_id": sid, "request_cost_usd": round(request_cost, 4),
                    "session_cost_usd": round(snap.values.get("cost_usd") or 0, 4), **task.interrupts[0].value}
    return {"session_id": sid, "step": "done", "finished": True, "request_cost_usd": round(request_cost, 4),
            "session_cost_usd": round(snap.values.get("cost_usd") or 0, 4)}


@router.get("/screener", include_in_schema=False)
def page() -> FileResponse:
    return FileResponse(PAGE)


@router.get("/screen/status")
def status() -> Dict[str, Any]:
    try:
        data = load_dataset()
    except FileNotFoundError as exc:
        return {"ready": False, "detail": str(exc)}
    missing = profiles.missing(data["industries"])
    return {"ready": True, "built_at": data["built_at"], "companies": len(data["companies"]),
            "industries": len(data["industries"]), "missing_profiles": len(missing),
            "profile_estimate_usd": round(len(missing) * profiles.EST_USD_PER_INDUSTRY, 2),
            "feedback_pairs": len(preferences.pairs())}


class StartRequest(BaseModel):
    goals: str = ""


@router.post("/screen")
def start(req: StartRequest) -> Dict[str, Any]:
    try:
        load_dataset()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    sid = uuid.uuid4().hex[:12]
    meter = start_meter()
    with _lock(sid):
        get_graph().invoke({"goals": req.goals, "choices": {}, "cost_usd": 0.0}, _config(sid))
    return _payload(sid, summarize_costs(meter)["total_usd"])


@router.get("/screen/jobs/{job_id}")
def job(job_id: str) -> Dict[str, Any]:
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="No such job")
    return _jobs[job_id]


@router.get("/screen/export/{kind}")
def export(kind: str) -> List[Dict[str, Any]]:
    if kind == "pairs":
        return preferences.export_pairs()
    if kind == "labels":
        from screener.training.export import label_examples
        return label_examples()
    raise HTTPException(status_code=404, detail="kind must be pairs or labels")


@router.get("/screen/{sid}")
def current(sid: str) -> Dict[str, Any]:
    return _payload(sid)


@router.post("/screen/{sid}/answer")
def answer(sid: str, body: Dict[str, Any]) -> Dict[str, Any]:
    _payload(sid)  # 404 for unknown sessions
    if body.get("action") == "deep_dive" and not body.get("confirm"):
        raise HTTPException(status_code=402, detail="Deep dives call Claude; resend with confirm=true")
    meter = start_meter()
    with _lock(sid):
        get_graph().invoke(Command(resume=body), _config(sid))
    return _payload(sid, summarize_costs(meter)["total_usd"])


class Confirm(BaseModel):
    confirm: bool = False
    limit: Optional[int] = None


@router.post("/screen/profiles")
def classify(req: Confirm) -> Dict[str, Any]:
    data = load_dataset()
    todo = profiles.missing(data["industries"])[: req.limit or None]
    if not req.confirm:
        return {"todo": len(todo), "estimate_usd": round(len(todo) * profiles.EST_USD_PER_INDUSTRY, 2)}
    meter = start_meter()
    result = profiles.classify_all(data["industries"], {c["ticker"]: c for c in data["companies"]}, names=todo)
    return {**result, "cost_usd": summarize_costs(meter)["total_usd"]}


@router.post("/screen/tam/{sub_industry}")
def tam(sub_industry: str, req: Confirm) -> Dict[str, Any]:
    if not req.confirm:
        return {"estimate_usd": profiles.EST_USD_PER_TAM_CHECK}
    meter = start_meter()
    try:
        out = profiles.verify_tam(sub_industry)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {**out, "cost_usd": summarize_costs(meter)["total_usd"]}


class ReportRequest(BaseModel):
    ticker: str
    confirm: bool = False


@router.post("/screen/{sid}/report")
def report(sid: str, req: ReportRequest) -> Dict[str, Any]:
    """Run the report generator for one company in the background (~11 minutes, about $3.60)."""
    if not req.confirm:
        return {"estimate_usd": REPORT_EST_USD, "minutes": 11}
    rows = {c["ticker"]: c for c in load_dataset()["companies"]}
    if req.ticker not in rows:
        raise HTTPException(status_code=404, detail=f"{req.ticker} is not in the universe")
    from report_generator.api import ResearchRequest, _run
    from report_generator.workflow import initial_state

    row = rows[req.ticker]
    request = ResearchRequest(company_name=row["name"], ticker=row["ticker"],
                              query=f"幫我生成{row['name']}股票研究報告，要包含合理股價，競爭對手，價值鏈，產業分析，總經分析")
    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {"job_id": job_id, "ticker": req.ticker, "status": "running", "started": time.time(), "session": sid}

    def run() -> None:
        try:
            resp = _run(request, initial_state(request.company_name, request.query, None, request.ticker))
            _jobs[job_id].update(status="done", report=resp.report, cost_usd=resp.cost.get("total_usd"),
                                 rating=resp.thesis.rating if resp.thesis else None)
        except Exception as exc:
            logger.exception("Report job %s failed", job_id)
            _jobs[job_id].update(status="failed", error=getattr(exc, "detail", None) or f"{type(exc).__name__}: {exc}")

    threading.Thread(target=run, daemon=True).start()
    return _jobs[job_id]

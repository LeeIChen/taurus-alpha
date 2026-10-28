"""HTTP endpoints for the single-company research report: POST /research and resume."""

from __future__ import annotations

import logging
import json
import time
from typing import Any, Dict, List, Optional

import anthropic
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.llm import CallTimeout, ClaudeRefusal, start_meter, summarize_costs
from report_generator.run_log import RUNS_DIR, save_run, utc_now
from report_generator.scoring.review_ui import router as review_router
from report_generator.state import (
    Citation,
    FinancialMetricResult,
    InvestmentThesis,
    PipelineValuationResult,
    Source,
    TaskPlan,
    ValuationResult,
)
from report_generator.workflow import build_graph, dump_state, initial_state, load_state, next_step

logger = logging.getLogger("uvicorn.error")


router = APIRouter(tags=["report"])
router.include_router(review_router)


class ResearchRequest(BaseModel):
    company_name: str
    query: str
    ticker: Optional[str] = Field(default=None, description="Optional ticker; otherwise resolved from company_name at SEC")
    current_price: Optional[float] = Field(default=None, gt=0, description="Optional share price; otherwise taken from web research")


class ResearchResponse(BaseModel):
    company_name: str
    ticker: Optional[str]
    plan: Optional[TaskPlan]
    thesis: Optional[InvestmentThesis]
    valuation_method: Optional[str]
    current_price: Optional[float]
    price_source: Optional[str]
    valuation: Optional[ValuationResult]
    pipeline_valuation: Optional[PipelineValuationResult]
    financial_results: List[FinancialMetricResult]
    citations: List[Citation]
    sources: List[Source]
    research_notes: str
    report: str
    edit_notes: List[str] = []  # editor notes still open after the last revision
    revisions: int = 0
    cost: Dict[str, Any] = {}  # total and per-step USD, plus every call's tokens
    market_data: Dict[str, Any] = {}  # peer table, history and macro series (computed)
    section_candidates: Dict[str, List[Dict[str, Any]]] = {}  # best-of-n drafts per section
    draft_versions: List[str] = []  # memos before each revision
    edit_history: List[List[str]] = []  # notes each revision applied
    faithfulness_score: float
    error: Optional[str]


@router.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    # Sync handler: FastAPI runs it in a worker thread, so the blocking graph call is fine.
    state = initial_state(request.company_name, request.query, request.current_price, request.ticker)
    return _run(request, state)


@router.post("/research/resume/{run_id}", response_model=ResearchResponse)
def resume(run_id: str) -> ResearchResponse:
    """Re-run a failed run from the step after its last completed one (saved as its checkpoint)."""
    path = RUNS_DIR / f"{run_id}.json"
    if not path.is_file() or path.parent != RUNS_DIR:
        raise HTTPException(status_code=404, detail=f"No run {run_id}")
    record = json.loads(path.read_text())
    if record.get("status") != "failed" or not record.get("checkpoint"):
        raise HTTPException(status_code=409, detail="Only a failed run with a saved checkpoint can be resumed")
    return _run(ResearchRequest(**record["request"]), load_state(record["checkpoint"]), resumed_from=run_id)


def _run(request: ResearchRequest, state: dict, resumed_from: Optional[str] = None) -> ResearchResponse:
    started_at, t0 = utc_now(), time.monotonic()
    meter = start_meter()  # every Claude call in this run adds its tokens and cost
    first = next_step(state)
    if first is None:
        raise HTTPException(status_code=409, detail="That run already finished every step")
    logger.info("Running %s from step %s", request.company_name, first)
    result = state
    try:
        # stream_mode="values" yields the full state after each step: the checkpoint if a later one fails
        for result in build_graph(first).stream(state, stream_mode="values"):
            pass
    except Exception as exc:
        save_run(request.model_dump(), started_at, time.monotonic() - t0, failure=f"{type(exc).__name__}: {exc}",
                 cost=summarize_costs(meter),
                 checkpoint=dump_state(result), resumed_from=resumed_from)
        if isinstance(exc, CallTimeout):
            raise HTTPException(status_code=504, detail=f"{exc}; resume with POST /research/resume/<run_id>")
        if isinstance(exc, ClaudeRefusal):
            raise HTTPException(status_code=422, detail=str(exc))
        if isinstance(exc, anthropic.RateLimitError):
            raise HTTPException(status_code=429, detail="Upstream rate limit; retry later")
        if isinstance(exc, anthropic.APIStatusError):
            raise HTTPException(status_code=502, detail=f"Claude API error: {exc.status_code}")
        if isinstance(exc, anthropic.APIConnectionError):
            raise HTTPException(status_code=503, detail="Could not reach Claude API")
        raise
    response = ResearchResponse(
        company_name=request.company_name,
        ticker=result.get("ticker"),
        plan=result["plan"],
        thesis=result.get("thesis"),
        valuation_method=result.get("valuation_method"),
        current_price=result.get("current_price"),
        price_source=result.get("price_source"),
        valuation=result.get("valuation"),
        pipeline_valuation=result.get("pipeline_valuation"),
        financial_results=result["financial_results"],
        citations=result["citations"],
        sources=result.get("sources") or [],
        research_notes=result.get("research_notes") or "",
        report=result["draft_report"],
        edit_notes=result.get("edit_notes") or [],
        revisions=result.get("revisions") or 0,
        market_data=result.get("market_data") or {},
        section_candidates=result.get("section_candidates") or {},
        draft_versions=result.get("draft_versions") or [],
        edit_history=result.get("edit_history") or [],
        faithfulness_score=result["faithfulness_score"],
        error=result["error"],
    )
    response.cost = summarize_costs(meter)
    logger.info("Run cost $%.2f: %s", response.cost["total_usd"],
                ", ".join(f"{k} ${v['usd']:.2f}" for k, v in response.cost["by_step"].items()))
    save_run(request.model_dump(), started_at, time.monotonic() - t0, result=response.model_dump(),
             resumed_from=resumed_from)
    return response

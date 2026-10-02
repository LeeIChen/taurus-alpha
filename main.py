"""FastAPI entrypoint. Run with: uvicorn main:app --reload"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.llm import ClaudeRefusal
from app.run_log import save_run, utc_now
from app.state import (
    Citation,
    FinancialMetricResult,
    InvestmentThesis,
    PipelineValuationResult,
    Source,
    TaskPlan,
    ValuationResult,
)
from app.tools.rag_search import default_index, load_filings
from app.workflow import initial_state, research_graph

FILINGS_DIR = Path(os.environ.get("TAURUS_FILINGS_DIR", Path(__file__).parent / "data" / "filings"))
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    count = load_filings(FILINGS_DIR)
    mode = "hybrid (BM25 + embeddings)" if default_index.has_embeddings else "BM25 only"
    logger.info("Loaded %d filing chunks from %s; search mode: %s", count, FILINGS_DIR, mode)
    yield


app = FastAPI(title="taurus-alpha", lifespan=lifespan)


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
    faithfulness_score: float
    error: Optional[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    # Sync handler: FastAPI runs it in a worker thread, so the blocking graph call is fine.
    started_at, t0 = utc_now(), time.monotonic()
    try:
        result = research_graph.invoke(
            initial_state(request.company_name, request.query, request.current_price, request.ticker)
        )
    except Exception as exc:
        save_run(request.model_dump(), started_at, time.monotonic() - t0, failure=f"{type(exc).__name__}: {exc}")
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
        faithfulness_score=result["faithfulness_score"],
        error=result["error"],
    )
    save_run(request.model_dump(), started_at, time.monotonic() - t0, result=response.model_dump())
    return response

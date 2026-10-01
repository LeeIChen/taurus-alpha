"""FastAPI entrypoint. Run with: uvicorn main:app --reload"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.llm import ClaudeRefusal
from app.state import Citation, FinancialMetricResult, TaskPlan
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


class ResearchResponse(BaseModel):
    plan: Optional[TaskPlan]
    financial_results: List[FinancialMetricResult]
    citations: List[Citation]
    report: str
    faithfulness_score: float
    error: Optional[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/research", response_model=ResearchResponse)
def research(request: ResearchRequest) -> ResearchResponse:
    # Sync handler: FastAPI runs it in a worker thread, so the blocking graph call is fine.
    try:
        result = research_graph.invoke(initial_state(request.company_name, request.query))
    except ClaudeRefusal as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Upstream rate limit; retry later")
    except anthropic.APIStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Claude API error: {exc.status_code}")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=503, detail="Could not reach Claude API")
    return ResearchResponse(
        plan=result["plan"],
        financial_results=result["financial_results"],
        citations=result["citations"],
        report=result["draft_report"],
        faithfulness_score=result["faithfulness_score"],
        error=result["error"],
    )

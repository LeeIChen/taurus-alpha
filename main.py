"""FastAPI entrypoint. Run with: uvicorn main:app --reload

- /screener          interactive stock screener (screener/)
- /research          single-company research report (report_generator/)
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from core.tools.rag_search import default_index, load_filings
from report_generator.api import router as report_router

FILINGS_DIR = Path(os.environ.get("TAURUS_FILINGS_DIR", Path(__file__).parent / "data" / "filings"))
logger = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_: FastAPI):
    count = load_filings(FILINGS_DIR)
    mode = "hybrid (BM25 + embeddings)" if default_index.has_embeddings else "BM25 only"
    logger.info("Loaded %d filing chunks from %s; search mode: %s", count, FILINGS_DIR, mode)
    yield


app = FastAPI(title="taurus-alpha", lifespan=lifespan)
app.include_router(report_router)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}

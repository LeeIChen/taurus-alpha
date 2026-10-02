"""Company intake: resolve the company at SEC, make sure its latest filings are searchable,
and read key reported financials from XBRL."""

from __future__ import annotations

import logging
from typing import List

from app.state import InvestmentAgentState, Source
from app.tools import rag_search
from app.tools.sec import (
    FILINGS_DIR,
    SecAccessError,
    fetch_company_filings,
    filings_are_fresh,
    reported_financials,
    resolve_company,
)

logger = logging.getLogger("uvicorn.error")


def ingest_node(state: InvestmentAgentState) -> dict:
    issues: List[str] = []
    try:
        company = resolve_company(state["company_name"], state.get("ticker"))
    except (ValueError, SecAccessError, OSError) as exc:
        return {"current_step": "ingest", "reported_financials": {},
                "error": _append(state.get("error"), f"Could not resolve company at SEC: {exc}")}

    ticker, name = company["ticker"], company["company"]
    rag_search.register_company(name, ticker)
    rag_search.register_alias(state["company_name"], name)

    paths = sorted((FILINGS_DIR / ticker).glob("*.jsonl"))
    if not filings_are_fresh(ticker):
        try:
            paths = fetch_company_filings(ticker, name, company["cik"])
        except (SecAccessError, OSError) as exc:
            issues.append(f"Could not download {ticker} filings ({exc}); using {len(paths)} cached file(s)")
    added = rag_search.add_filing_files(paths)
    logger.info("Ingest %s (%s): %d filing files, %d new chunks", name, ticker, len(paths), added)

    try:
        financials = reported_financials(company["cik"])
    except (SecAccessError, OSError, KeyError) as exc:
        financials = {}
        issues.append(f"Could not read XBRL financials: {exc}")

    sources = [
        Source(source_id=f"F{i}", title=f"{ticker} {p.stem.replace('_', ' for period ')} (SEC EDGAR)",
               url=_filing_url(p), kind="filing")
        for i, p in enumerate(paths, start=1)
    ]
    update = {"ticker": ticker, "reported_financials": financials, "sources": sources, "current_step": "ingest"}
    if not paths:
        issues.append(f"No filings available for {ticker}; analysis will rely on web research")
    if issues:
        update["error"] = _append(state.get("error"), "; ".join(issues))
    return update


def _filing_url(path) -> str:
    import json

    with path.open() as f:
        first = f.readline()
    return json.loads(first)["metadata"].get("source_url", "") if first else ""


def _append(previous, issue: str) -> str:
    return f"{previous}\n{issue}" if previous else issue

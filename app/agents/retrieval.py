"""Data Retrieval Agent: runs hybrid search for every planned task and metric.

Searches the target company's filings for every query, and also the filings of
any other covered company a query names (e.g. "hyperscaler capex at Microsoft").

These pages are the shared evidence that analysis, valuation, thesis and the verifier read (and
cache), so they are capped: the section writers retrieve their own pages for writing. Pages are
taken round-robin by rank, so every query keeps its best match before any query gets a third.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict

from app.state import InvestmentAgentState
from app.tools.rag_search import canonical_company, companies_mentioned, rag_search

logger = logging.getLogger("uvicorn.error")

TARGET_TOP_K = 3
PEER_TOP_K = 2
MAX_TARGET_PAGES = 32  # about 45k tokens
MAX_PEER_PAGES = 8


def retrieval_node(state: InvestmentAgentState) -> dict:
    plan = state["plan"]
    target = canonical_company(state["company_name"])
    queries = plan.tasks + plan.required_metrics

    seen: Dict[str, Dict[str, Any]] = {}
    best_rank: Dict[str, int] = {}

    def keep(doc: Dict[str, Any], rank: int) -> None:
        if doc["doc_id"] not in seen or doc["score"] > seen[doc["doc_id"]]["score"]:
            seen[doc["doc_id"]] = doc
        best_rank[doc["doc_id"]] = min(rank, best_rank.get(doc["doc_id"], rank))

    for query in queries:
        for rank, doc in enumerate(rag_search(query, company=target, top_k=TARGET_TOP_K)):
            keep(doc, rank)
        for peer in companies_mentioned(query):
            if peer != target:
                for rank, doc in enumerate(rag_search(query, company=peer, top_k=PEER_TOP_K)):
                    keep(doc, rank)

    ranked = sorted(seen.values(), key=lambda d: (best_rank[d["doc_id"]], -d["score"]))
    own = [d for d in ranked if d["metadata"].get("company") == target][:MAX_TARGET_PAGES]
    peers = [d for d in ranked if d["metadata"].get("company") != target][:MAX_PEER_PAGES]
    # Target company first, then peers; best matches first within each.
    docs = sorted(own + peers, key=lambda d: (d["metadata"].get("company") != target, -d["score"]))
    logger.info("Retrieved %d pages: %s", len(docs), dict(Counter(d["metadata"].get("company") for d in docs)))

    update: dict = {"retrieved_docs": docs, "current_step": "retrieval"}
    if not any(d["metadata"].get("company") == target for d in docs):
        update["error"] = f"No documents found for {state['company_name']}"
    return update

"""Data Retrieval Agent: runs hybrid search for every planned task and metric.

Searches the target company's filings for every query, and also the filings of
any other covered company a query names (e.g. "hyperscaler capex at Microsoft").
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any, Dict

from app.state import InvestmentAgentState
from app.tools.rag_search import canonical_company, companies_mentioned, rag_search

logger = logging.getLogger("uvicorn.error")

TARGET_TOP_K = 5
PEER_TOP_K = 3


def retrieval_node(state: InvestmentAgentState) -> dict:
    plan = state["plan"]
    target = canonical_company(state["company_name"])
    queries = plan.tasks + plan.required_metrics

    seen: Dict[str, Dict[str, Any]] = {}

    def keep(doc: Dict[str, Any]) -> None:
        if doc["doc_id"] not in seen or doc["score"] > seen[doc["doc_id"]]["score"]:
            seen[doc["doc_id"]] = doc

    for query in queries:
        for doc in rag_search(query, company=target, top_k=TARGET_TOP_K):
            keep(doc)
        for peer in companies_mentioned(query):
            if peer != target:
                for doc in rag_search(query, company=peer, top_k=PEER_TOP_K):
                    keep(doc)

    # Target company first, then peers; best matches first within each.
    docs = sorted(seen.values(), key=lambda d: (d["metadata"].get("company") != target, -d["score"]))
    logger.info("Retrieved %d pages: %s", len(docs), dict(Counter(d["metadata"].get("company") for d in docs)))

    update: dict = {"retrieved_docs": docs, "current_step": "retrieval"}
    if not any(d["metadata"].get("company") == target for d in docs):
        update["error"] = f"No documents found for {state['company_name']}"
    return update

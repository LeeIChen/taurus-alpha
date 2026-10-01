"""Data Retrieval Agent: runs hybrid search for every planned task and metric."""

from __future__ import annotations

from typing import Any, Dict

from app.state import InvestmentAgentState
from app.tools.rag_search import rag_search


def retrieval_node(state: InvestmentAgentState) -> dict:
    plan = state["plan"]
    queries = plan.tasks + plan.required_metrics

    seen: Dict[str, Dict[str, Any]] = {}
    for query in queries:
        for doc in rag_search(query, company=state["company_name"]):
            if doc["doc_id"] not in seen or doc["score"] > seen[doc["doc_id"]]["score"]:
                seen[doc["doc_id"]] = doc
    docs = sorted(seen.values(), key=lambda d: d["score"], reverse=True)

    update: dict = {"retrieved_docs": docs, "current_step": "retrieval"}
    if not docs:
        update["error"] = f"No documents found for {state['company_name']}"
    return update

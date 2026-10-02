"""Faithfulness check: scores how well the draft report is supported by its sources."""

from __future__ import annotations

import json
from typing import List

from pydantic import BaseModel, Field

from app.agents.analysis import format_documents
from app.llm import ask_structured
from app.state import InvestmentAgentState

SYSTEM = """You audit investment memos for faithfulness. Check every factual claim and number in
the draft against the source documents, computed metrics and valuation. A claim is supported only
if the sources state it or it follows directly from the computed metrics or valuation. Opinions,
clearly labelled views, and valuation assumptions presented as assumptions do not count as claims,
but a target price or DCF figure that differs from the valuation output does."""


class _FaithfulnessVerdict(BaseModel):
    supported_claims: int
    total_claims: int
    unsupported_claims: List[str] = Field(description="Quote each claim the sources do not support")


def verifier_node(state: InvestmentAgentState) -> dict:
    metrics = json.dumps([r.model_dump() for r in state["financial_results"]], indent=2)
    valuation = state.get("valuation")
    prompt = (
        f"Draft report:\n{state['draft_report']}\n\n"
        f"Computed metrics:\n{metrics}\n\n"
        f"Valuation:\n{valuation.model_dump_json(indent=2) if valuation else 'none'}\n\n"
        f"Source documents:\n{format_documents(state['retrieved_docs'])}"
    )
    verdict = ask_structured(SYSTEM, prompt, _FaithfulnessVerdict)

    total = max(verdict.total_claims, 0)
    score = 1.0 if total == 0 else min(max(verdict.supported_claims / total, 0.0), 1.0)
    update: dict = {"faithfulness_score": score, "current_step": "verifier"}
    if verdict.unsupported_claims:
        issues = "Unsupported claims: " + " | ".join(verdict.unsupported_claims)
        update["error"] = f"{state['error']}\n{issues}" if state.get("error") else issues
    return update

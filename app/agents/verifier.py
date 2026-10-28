"""Faithfulness check: scores how well the draft report is supported by its sources."""

from __future__ import annotations

import json
from app.agents.analysis import format_documents, shared_evidence
from app.agents.sections import cited_pages
from app.agents.writer import valuation_payload
from app.agents.schemas import run_step
from app.state import InvestmentAgentState

SYSTEM = """You audit investment memos for faithfulness. Check every factual claim and number in
the draft against the filing pages, SEC-reported financials, web research notes (whose statements
carry source ids), computed metrics, valuation and thesis. A claim is supported only if the inputs
state it or it follows directly from the computed metrics or valuation. A citation to a source id
that does not exist, or that does not support the claim, makes the claim unsupported. Opinions,
clearly labelled views, and valuation assumptions presented as assumptions do not count as claims,
but a target price or DCF figure that differs from the valuation output does."""


def verifier_node(state: InvestmentAgentState) -> dict:
    metrics = json.dumps([r.model_dump() for r in state["financial_results"]], indent=2)
    thesis = state.get("thesis")
    # Pages a section writer retrieved and the memo cites, beyond the (cached) shared evidence
    shared_ids = {d["doc_id"] for d in state["retrieved_docs"]}
    cited = cited_pages(state["draft_report"], state.get("sources") or [])
    extra = {d["doc_id"]: d for docs in (state.get("section_docs") or {}).values() for d in docs
             if d["doc_id"] in cited and d["doc_id"] not in shared_ids}
    prompt = (
        f"Cited filing pages that are not in the shared evidence:\n"
        f"{format_documents(list(extra.values()))}\n\n"
        f"Draft report:\n{state['draft_report']}\n\n"
        f"Computed metrics:\n{metrics}\n\n"
        f"Valuation:\n{json.dumps(valuation_payload(state), indent=1, default=str)}\n\n"
        f"Thesis:\n{thesis.model_dump_json(indent=1) if thesis else 'none'}"
    )
    verdict = run_step("verdict", SYSTEM, prompt, state, shared_evidence(state))

    total = max(verdict.total_claims, 0)
    score = 1.0 if total == 0 else min(max(verdict.supported_claims / total, 0.0), 1.0)
    update: dict = {"faithfulness_score": score, "current_step": "verifier"}
    if verdict.unsupported_claims:
        issues = "Unsupported claims: " + " | ".join(verdict.unsupported_claims)
        update["error"] = f"{state['error']}\n{issues}" if state.get("error") else issues
    return update

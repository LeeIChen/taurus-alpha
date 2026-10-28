"""Thesis: sets the rating, price target, thesis points, risks and catalysts.

The memo itself is written section by section in app/agents/sections.py."""

from __future__ import annotations

import json

from app.agents.analysis import shared_evidence
from app.agents.schemas import run_step
from app.state import InvestmentAgentState

THESIS_SYSTEM = """You are the senior analyst who owns the recommendation. Using the valuation,
computed metrics, research notes and filings, set the investment thesis.

Rating rule: Buy if the 12-month price target is at least 15% above the current price, Sell if at
least 15% below, otherwise Hold. You may depart from the rule only when the catalyst path clearly
justifies it, and you must say why in the summary. Use the valuation's target price unless it is
missing. Every thesis point and catalyst must cite source ids (e.g. [S4], [F1]) or filing pages.
Catalysts must be dated or windowed, in chronological order."""


def valuation_payload(state: InvestmentAgentState) -> dict:
    pv, v = state.get("pipeline_valuation"), state.get("valuation")
    return {
        "method": state.get("valuation_method"),
        "current_price": state.get("current_price"),
        "price_source": state.get("price_source"),
        "pipeline_sotp": pv.model_dump() if pv else None,
        "company_dcf": v.model_dump() if v else None,
    }


def thesis_node(state: InvestmentAgentState) -> dict:
    payload = {
        "valuation": valuation_payload(state),
        "metrics": [r.model_dump(exclude={"code_executed"}) for r in state["financial_results"]],
        "citations": [c.model_dump() for c in state["citations"]],
        "questions_to_answer": state["plan"].tasks,
        "known_issues": state.get("error"),
    }
    analysis = f"Analysis (JSON):\n{json.dumps(payload, indent=1, default=str)}"
    thesis = run_step("thesis", THESIS_SYSTEM, analysis, state, shared_evidence(state))
    return {"thesis": thesis, "current_step": "thesis"}

"""Writer Agent: turns the computed metrics and citations into an investment memo."""

from __future__ import annotations

import json

from app.llm import ask_text
from app.state import InvestmentAgentState

SYSTEM = """You write concise investment research memos in Markdown for a portfolio manager.
Structure: Summary (with a clear view: bullish / neutral / bearish and why), Key Metrics,
Findings, Risks, Open Questions. Use only the metrics and citations provided, and cite sources
inline as [source_doc p.page_number]. Flag any gaps in the data. End with a one-line note that
this is research support, not investment advice."""


def writer_node(state: InvestmentAgentState) -> dict:
    payload = {
        "metrics": [r.model_dump() for r in state["financial_results"]],
        "citations": [c.model_dump() for c in state["citations"]],
        "tasks": state["plan"].tasks,
        "known_issues": state.get("error"),
    }
    prompt = (
        f"Company: {state['company_name']}\n"
        f"Question: {state['user_query']}\n\n"
        f"Analysis (JSON):\n{json.dumps(payload, indent=2)}"
    )
    return {"draft_report": ask_text(SYSTEM, prompt), "current_step": "writer"}

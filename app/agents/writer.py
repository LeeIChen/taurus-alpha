"""Writer Agent: turns the analysis, valuation and source pages into an investment memo."""

from __future__ import annotations

import json

from app.agents.analysis import format_documents
from app.llm import ask_text
from app.state import InvestmentAgentState

SYSTEM = """You write concise investment research memos in Markdown for a portfolio manager.
Structure: Summary (with a clear view: bullish / neutral / bearish, the target price, and why),
Key Metrics, Valuation & Target Price, Findings, Risks, Open Questions.

Ground every statement in the computed metrics, the valuation, or the source documents, and cite
sources inline as [source_doc p.page_number]. The source documents are the retrieved filing pages;
read them before calling anything a data gap, and only list a gap if the documents truly lack it.
Documents may come from peer companies; name the company when you use one.

In Valuation & Target Price, show the DCF value per share, the P/E-based price, the blended target
and its weights, the key assumptions with their rationale, and the DCF sensitivity range. Label
projections as assumptions, not reported facts. If a current price was provided, state the upside
or downside; otherwise say the target cannot be compared to the market price here.
End with a one-line note that this is research support, not investment advice."""


def writer_node(state: InvestmentAgentState) -> dict:
    valuation = state.get("valuation")
    payload = {
        "metrics": [r.model_dump() for r in state["financial_results"]],
        "valuation": valuation.model_dump() if valuation else None,
        "citations": [c.model_dump() for c in state["citations"]],
        "tasks": state["plan"].tasks,
        "known_issues": state.get("error"),
    }
    prompt = (
        f"Company: {state['company_name']}\n"
        f"Question: {state['user_query']}\n\n"
        f"Analysis (JSON):\n{json.dumps(payload, indent=2)}\n\n"
        f"Source documents:\n{format_documents(state['retrieved_docs'])}"
    )
    return {"draft_report": ask_text(SYSTEM, prompt), "current_step": "writer"}

"""Writer Agent: sets the investment thesis and recommendation, then writes the memo."""

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

MEMO_SYSTEM = """You write investment research memos in Markdown for a portfolio manager.
Structure:
1. Rating, price target, current price (with its source) and upside/downside, then the thesis summary.
2. Investment thesis: the key points.
3. Valuation: method and why; for a sum of the parts, a table of every program with stage,
   probability of success, its basis, risk-adjusted value and value per share; overhead, net cash and
   other adjustments; the DCF value, the P/E value and the blended target; key assumptions labelled as
   assumptions; what the current price implies.
4. Key metrics.
5. Catalyst path: a dated table with expected impact and direction.
6. Risks to the thesis and what would change our view.
7. Research tasks covered: list each plan task.
8. Sources: a numbered list of every source id you cited, with title and URL.

Ground every statement in the inputs and cite sources inline as [S#], [F#] or [source_doc p.N].
Read the documents before calling anything a data gap. End with a one-line note that this is
research support, not investment advice."""


def _valuation_payload(state: InvestmentAgentState) -> dict:
    pv, v = state.get("pipeline_valuation"), state.get("valuation")
    return {
        "method": state.get("valuation_method"),
        "current_price": state.get("current_price"),
        "price_source": state.get("price_source"),
        "pipeline_sotp": pv.model_dump() if pv else None,
        "company_dcf": v.model_dump() if v else None,
    }


def writer_node(state: InvestmentAgentState) -> dict:
    payload = {
        "valuation": _valuation_payload(state),
        "metrics": [r.model_dump(exclude={"code_executed"}) for r in state["financial_results"]],
        "citations": [c.model_dump() for c in state["citations"]],
        "tasks": state["plan"].tasks,
        "known_issues": state.get("error"),
    }
    analysis = f"Analysis (JSON):\n{json.dumps(payload, indent=1, default=str)}"
    evidence = shared_evidence(state)
    thesis = run_step("thesis", THESIS_SYSTEM, analysis, state, evidence)
    memo = run_step(
        "memo_markdown", MEMO_SYSTEM,
        f"{analysis}\n\nInvestment thesis (JSON):\n{thesis.model_dump_json(indent=1)}", state, evidence,
    )
    return {"thesis": thesis, "draft_report": memo, "current_step": "writer"}

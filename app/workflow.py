"""LangGraph state machine:

planner -> ingest -> market_data -> web_research -> retrieval -> analysis -> valuation -> thesis
thesis -> sections --Send--> write_section x8 (in parallel) -> assemble -> editor
editor -> reviser -> editor while the editor has notes (at most MAX_REVISIONS), then -> verifier

A failed run saves the state after its last completed step (see main.py), so it can resume
from the next step instead of paying for planning and web research again.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph
from pydantic import TypeAdapter

from app.agents.analysis import analysis_node
from app.agents.editor import after_editor, editor_node, reviser_node
from app.agents.ingest import ingest_node
from app.agents.market_data import market_data_node
from app.agents.planner import plan_node
from app.agents.retrieval import retrieval_node
from app.agents.sections import assemble_node, fan_out, sections_node, write_section
from app.agents.valuation import valuation_node
from app.agents.verifier import verifier_node
from app.agents.web_research import web_research_node
from app.agents.writer import thesis_node
from app.state import InvestmentAgentState

NODES = {
    "planner": plan_node,
    "ingest": ingest_node,
    "market_data": market_data_node,
    "web_research": web_research_node,
    "retrieval": retrieval_node,
    "analysis": analysis_node,
    "valuation": valuation_node,
    "thesis": thesis_node,
    "sections": sections_node,
    "write_section": write_section,
    "assemble": assemble_node,
    "editor": editor_node,
    "reviser": reviser_node,
    "verifier": verifier_node,
}
# Plain edges; editor -> reviser | verifier and sections -> write_section are conditional.
CHAIN = ["planner", "ingest", "market_data", "web_research", "retrieval", "analysis", "valuation",
         "thesis", "sections"]
STEP_NAMES = list(NODES)
_STATE = TypeAdapter(InvestmentAgentState)


def build_graph(first_step: str = "planner"):
    """The whole pipeline, entered at `first_step` (earlier steps are skipped on resume)."""
    graph = StateGraph(InvestmentAgentState)
    for name, node in NODES.items():
        graph.add_node(name, node)
    graph.add_edge(START, first_step)
    for a, b in zip(CHAIN, CHAIN[1:]):
        graph.add_edge(a, b)
    graph.add_conditional_edges("sections", fan_out, ["write_section"])
    graph.add_edge("write_section", "assemble")
    graph.add_edge("assemble", "editor")
    graph.add_conditional_edges("editor", after_editor, ["reviser", "verifier"])
    graph.add_edge("reviser", "editor")
    graph.add_edge("verifier", END)
    return graph.compile()


# Where to resume after each completed step. "writer" is the memo step of runs saved before the
# section writers existed; their memo goes straight to the editor.
_NEXT = {**{a: b for a, b in zip(CHAIN, CHAIN[1:])}, "start": "planner", "assemble": "editor",
         "reviser": "editor", "writer": "editor", "verifier": None}


def next_step(state: InvestmentAgentState) -> Optional[str]:
    """The step after the last completed one, or None if the pipeline finished."""
    done = state.get("current_step")
    if done == "editor":
        return after_editor(state)
    return _NEXT.get(done, "planner")


def dump_state(state: InvestmentAgentState) -> Dict[str, Any]:
    return _STATE.dump_python(state, mode="json")


def load_state(data: Dict[str, Any]) -> InvestmentAgentState:
    """Validate a saved checkpoint; keys added since it was saved get their initial values."""
    defaults = initial_state(data.get("company_name", ""), data.get("user_query", ""))
    return _STATE.validate_python({**defaults, **data})


def initial_state(
    company_name: str, user_query: str, current_price: Optional[float] = None, ticker: Optional[str] = None
) -> InvestmentAgentState:
    return {
        "company_name": company_name,
        "user_query": user_query,
        "ticker": ticker,
        "plan": None,
        "reported_financials": {},
        "research_notes": "",
        "sources": [],
        "retrieved_docs": [],
        "financial_results": [],
        "citations": [],
        "current_price": current_price,
        "price_source": "user-provided" if current_price else None,
        "valuation_method": None,
        "valuation": None,
        "pipeline_valuation": None,
        "thesis": None,
        "market_data": {},
        "memo_sections": {},
        "section_docs": {},
        "section_candidates": {},
        "draft_report": "",
        "edit_notes": [],
        "revisions": 0,
        "draft_versions": [],
        "edit_history": [],
        "faithfulness_score": 0.0,
        "current_step": "start",
        "error": None,
    }

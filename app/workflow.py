"""LangGraph state machine:

planner -> ingest -> web_research -> retrieval -> analysis -> valuation -> writer -> verifier

A failed run saves the state after its last completed step (see main.py), so it can resume
from the next step instead of paying for planning and web research again.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import END, START, StateGraph
from pydantic import TypeAdapter

from app.agents.analysis import analysis_node
from app.agents.ingest import ingest_node
from app.agents.planner import plan_node
from app.agents.retrieval import retrieval_node
from app.agents.valuation import valuation_node
from app.agents.verifier import verifier_node
from app.agents.web_research import web_research_node
from app.agents.writer import writer_node
from app.state import InvestmentAgentState

STEPS = [
    ("planner", plan_node),
    ("ingest", ingest_node),
    ("web_research", web_research_node),
    ("retrieval", retrieval_node),
    ("analysis", analysis_node),
    ("valuation", valuation_node),
    ("writer", writer_node),
    ("verifier", verifier_node),
]


STEP_NAMES = [name for name, _ in STEPS]
_STATE = TypeAdapter(InvestmentAgentState)


def build_graph(first_step: str = STEP_NAMES[0]):
    """The pipeline from `first_step` to the end."""
    steps = STEPS[STEP_NAMES.index(first_step):]
    graph = StateGraph(InvestmentAgentState)
    for name, node in steps:
        graph.add_node(name, node)
    graph.add_edge(START, steps[0][0])
    for (a, _), (b, _) in zip(steps, steps[1:]):
        graph.add_edge(a, b)
    graph.add_edge(steps[-1][0], END)
    return graph.compile()


def next_step(state: InvestmentAgentState) -> Optional[str]:
    """The step after the last completed one, or None if the pipeline finished."""
    done = state.get("current_step")
    if done not in STEP_NAMES:
        return STEP_NAMES[0]
    i = STEP_NAMES.index(done) + 1
    return STEP_NAMES[i] if i < len(STEP_NAMES) else None


def dump_state(state: InvestmentAgentState) -> Dict[str, Any]:
    return _STATE.dump_python(state, mode="json")


def load_state(data: Dict[str, Any]) -> InvestmentAgentState:
    return _STATE.validate_python(data)


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
        "draft_report": "",
        "faithfulness_score": 0.0,
        "current_step": "start",
        "error": None,
    }

"""LangGraph state machine:

planner -> ingest -> web_research -> retrieval -> analysis -> valuation -> writer -> verifier
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph

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


def build_graph():
    graph = StateGraph(InvestmentAgentState)
    for name, node in STEPS:
        graph.add_node(name, node)
    graph.add_edge(START, STEPS[0][0])
    for (a, _), (b, _) in zip(STEPS, STEPS[1:]):
        graph.add_edge(a, b)
    graph.add_edge(STEPS[-1][0], END)
    return graph.compile()


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


research_graph = build_graph()

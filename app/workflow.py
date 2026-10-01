"""LangGraph state machine: planner -> retrieval -> analysis -> writer -> verifier."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.agents.analysis import analysis_node
from app.agents.planner import plan_node
from app.agents.retrieval import retrieval_node
from app.agents.verifier import verifier_node
from app.agents.writer import writer_node
from app.state import InvestmentAgentState


def build_graph():
    graph = StateGraph(InvestmentAgentState)
    graph.add_node("planner", plan_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("analysis", analysis_node)
    graph.add_node("writer", writer_node)
    graph.add_node("verifier", verifier_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "retrieval")
    graph.add_edge("retrieval", "analysis")
    graph.add_edge("analysis", "writer")
    graph.add_edge("writer", "verifier")
    graph.add_edge("verifier", END)
    return graph.compile()


def initial_state(company_name: str, user_query: str) -> InvestmentAgentState:
    return {
        "company_name": company_name,
        "user_query": user_query,
        "plan": None,
        "retrieved_docs": [],
        "financial_results": [],
        "citations": [],
        "draft_report": "",
        "faithfulness_score": 0.0,
        "current_step": "start",
        "error": None,
    }


research_graph = build_graph()

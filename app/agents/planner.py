"""Planner Agent: turns the user's question into sub-tasks and the metrics to compute."""

from __future__ import annotations

from app.llm import ask_structured
from app.state import InvestmentAgentState, TaskPlan

SYSTEM = """You are the planning lead of an equity research team covering listed companies.
Break the user's investment question into 3-6 focused sub-tasks, each answerable from company
filings, earnings call transcripts, or recent news. Also list the financial metrics that must
be computed to answer it (e.g. EBITDA margin, revenue CAGR, net debt / EBITDA)."""


def plan_node(state: InvestmentAgentState) -> dict:
    prompt = f"Company: {state['company_name']}\nQuestion: {state['user_query']}"
    plan = ask_structured(SYSTEM, prompt, TaskPlan)
    return {"plan": plan, "current_step": "planner"}

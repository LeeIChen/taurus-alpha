"""Planner Agent: turns the user's question into sub-tasks and the metrics to compute."""

from __future__ import annotations

from app.llm import ask_structured
from app.state import InvestmentAgentState, TaskPlan
from app.tools.rag_search import available_companies

SYSTEM = """You are the planning lead of an equity research team covering listed companies.
Break the user's investment question into 3-6 focused sub-tasks and list the financial metrics
that must be computed to answer it (e.g. EBITDA margin, revenue CAGR, net debt / EBITDA).

The only sources available are the latest 10-K and 10-Q filings for: {companies}.
There are no market prices, analyst estimates, earnings call transcripts or news, so plan only
work these filings can support. When another covered company's filings would help (for example,
hyperscaler capital expenditure as a demand signal for a chip supplier), name that company
explicitly in the sub-task so its filings are searched too.

The analysis will also value the company with a DCF and a P/E multiple, so always include the
inputs: free cash flow (operating cash flow and capex), diluted shares outstanding, cash and
marketable securities, total debt, and diluted EPS."""


def plan_node(state: InvestmentAgentState) -> dict:
    system = SYSTEM.format(companies=", ".join(available_companies()) or "none loaded")
    prompt = f"Company: {state['company_name']}\nQuestion: {state['user_query']}"
    plan = ask_structured(system, prompt, TaskPlan)
    return {"plan": plan, "current_step": "planner"}

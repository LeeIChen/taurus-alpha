"""Planner Agent: turns the user's question into sub-tasks and the metrics to compute."""

from __future__ import annotations

from app.llm import ask_structured
from app.state import InvestmentAgentState, TaskPlan
from app.tools.rag_search import available_companies

SYSTEM = """You are the planning lead of an equity research team covering listed companies.
Break the user's investment question into 4-8 focused research tasks and list the financial metrics
that must be computed to answer it (e.g. revenue growth, gross margin, cash burn, net cash).

Sources available to the team:
- The target company's latest 10-K and 10-Q (fetched automatically) and SEC-reported financials.
- The latest 10-K and 10-Q for: {companies}. Name another covered company explicitly in a task when
  its filings would help, so they are searched too.
- Web research with citations: recent news, press releases, regulatory decisions, clinical trial
  results and registries, published success-rate studies, share price and sell-side targets.

Always include tasks that cover: recent developments and the upcoming catalyst path; every product
and development program that drives value (for pipeline companies, one task per major program or
tumor type, including probability-of-success benchmarks); market size and pricing; balance sheet,
financing and contingent liabilities; and the inputs for valuation (DCF or sum of the parts, and P/E)."""


def plan_node(state: InvestmentAgentState) -> dict:
    system = SYSTEM.format(companies=", ".join(available_companies()) or "none loaded")
    prompt = f"Company: {state['company_name']}\nQuestion: {state['user_query']}"
    plan = ask_structured(system, prompt, TaskPlan)
    return {"plan": plan, "current_step": "planner"}

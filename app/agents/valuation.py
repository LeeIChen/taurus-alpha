"""Valuation Agent: chooses a method, sets cited assumptions, and the math runs in code.

- company_dcf: one company-level DCF plus P/E (mature, cash-generating businesses).
- pipeline_sotp: sum of the parts, one risk-adjusted NPV per product or program, plus P/E on
  risk-adjusted earnings (companies whose value depends on development-stage programs).
"""

from __future__ import annotations

import json
from datetime import date
from app.agents.analysis import shared_evidence
from app.agents.schemas import run_step
from app.state import InvestmentAgentState
from app.tools.valuation import value_company, value_pipeline

SYSTEM = """You are the valuation lead on an equity research team. Choose the method:
- "pipeline_sotp" when a material part of value depends on products not yet approved or launched
  (biotech, pharma pipelines). Model every marketed product and every material development program
  as its own line. Probability of success must start from a published benchmark for the phase and
  therapeutic area (e.g. BIO/Informa/QLS clinical development success rates, cited by source id),
  then adjust for program-specific evidence, and pos_basis must say both. Marketed products are 1.0.
  Use economic_share for partnerships (e.g. 0.5 for a 50/50 profit split) and peak sales for the whole
  indication. Include corporate overhead not allocated to programs, net cash including convertible
  debt, and probability-weighted contingent liabilities.
- "company_dcf" for mature businesses valued on company-level free cash flow.

Reported figures (cash, debt, shares, sales, cash flow) must come from the filings, SEC-reported
financials or cited web sources. Projections are your judgment and must be justified in the
rationale with source ids. Use the latest share price found in the research notes unless one is
provided, and say where it came from. Money in USD millions, shares in millions, rates as decimals."""


def valuation_node(state: InvestmentAgentState) -> dict:
    user_price = state.get("current_price")
    metrics = json.dumps([m.model_dump(exclude={"code_executed"}) for m in state.get("financial_results") or []], indent=1)
    prompt = (
        f"Today is {date.today().isoformat()}.\n"
        + (f"Current share price (user-provided): ${user_price:,.2f}\n" if user_price else "")
        + f"\nComputed metrics:\n{metrics}"
    )
    draft = run_step("valuation", SYSTEM, prompt, state, shared_evidence(state))
    price = user_price or draft.current_price
    update: dict = {"current_price": price, "valuation": None, "pipeline_valuation": None,
                    "current_step": "valuation",
                    "valuation_method": draft.method, "price_source": "user-provided" if user_price else draft.current_price_source}
    try:
        if draft.method == "pipeline_sotp" and draft.pipeline:
            update["pipeline_valuation"] = value_pipeline(draft.pipeline, current_price=price)
        elif draft.company_dcf:
            update["valuation"] = value_company(draft.company_dcf, current_price=price)
        else:
            raise ValueError(f"method {draft.method} returned no assumptions")
    except ValueError as exc:
        issue = f"Valuation failed: {exc}"
        update["error"] = f"{state['error']}\n{issue}" if state.get("error") else issue
    return update

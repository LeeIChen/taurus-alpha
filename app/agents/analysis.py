"""Financial Analysis Agent: computes each required metric in the code executor and values the company."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.llm import ask_structured
from app.state import Citation, FinancialMetricResult, InvestmentAgentState, ValuationAssumptions
from app.tools.code_executor import run_python
from app.tools.valuation import value_company

SYSTEM = """You are a buy-side financial analyst. Using only the provided source documents,
compute each required metric. For every metric, write a short standalone Python script
(standard library only) that hard-codes the input figures from the documents and prints only
the final numeric value. Add a citation for every figure you use, quoting the snippet it came
from. If the documents lack the inputs for a metric, omit that metric rather than estimating.
Documents may come from several companies; check each document's company before using it.

Then set the valuation assumptions for a DCF and a P/E target price:
- Reported figures (free cash flow = operating cash flow minus capex, diluted shares, cash and
  marketable securities, debt, trailing EPS) must come from the target company's latest filings
  and be cited.
- Projections (5 years of FCF growth, terminal growth, discount rate, forward EPS, target P/E,
  DCF weight) are your judgment. Justify each in the rationale using the filings: recent growth,
  margins, concentration, capital intensity, and what peer filings show about demand.
- No market prices are in the documents. Do not claim a current share price or a peer's P/E;
  justify the multiple from growth, profitability and risk instead.
Express FCF, cash and debt in USD millions and shares in millions."""


class _MetricDraft(BaseModel):
    metric_name: str
    formula_used: str = Field(description="Formula in words or symbols, e.g. 'EBITDA / Revenue'")
    code: str = Field(description="Python script that prints only the metric's numeric value")


class _AnalysisDraft(BaseModel):
    metrics: List[_MetricDraft]
    citations: List[Citation]
    valuation: ValuationAssumptions


def format_documents(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "(no documents retrieved)"
    return "\n\n".join(
        f'<document source_doc="{d["doc_id"]}" company="{d["metadata"].get("company", "")}" '
        f'doc_type="{d["metadata"].get("doc_type", "")}" page_number="{d["metadata"].get("page_number", 0)}">\n'
        f'{d["content"]}\n</document>'
        for d in docs
    )


def _append_error(state: InvestmentAgentState, update: dict, issue: str) -> None:
    previous = update.get("error") or state.get("error")
    update["error"] = f"{previous}\n{issue}" if previous else issue


def analysis_node(state: InvestmentAgentState) -> dict:
    plan = state["plan"]
    price = state.get("current_price")
    price_line = (
        f"Current share price (user-provided): ${price:,.2f}" if price
        else "Current share price: not provided"
    )
    prompt = (
        f"Target company: {state['company_name']}\n"
        f"Question: {state['user_query']}\n"
        f"{price_line}\n"
        f"Required metrics: {', '.join(plan.required_metrics)}\n\n"
        f"Source documents:\n{format_documents(state['retrieved_docs'])}"
    )
    draft = ask_structured(SYSTEM, prompt, _AnalysisDraft)

    results: List[FinancialMetricResult] = []
    failures: List[str] = []
    for metric in draft.metrics:
        run = run_python(metric.code)
        try:
            value = float(run.stdout.strip().splitlines()[-1]) if run.ok else None
        except (ValueError, IndexError):
            value = None
        if value is None:
            reason = run.stderr.strip().splitlines()[-1] if run.stderr.strip() else "no numeric output"
            failures.append(f"{metric.metric_name}: {reason}")
            continue
        results.append(
            FinancialMetricResult(
                metric_name=metric.metric_name,
                value=value,
                formula_used=metric.formula_used,
                code_executed=metric.code,
            )
        )

    update: dict = {
        "financial_results": results,
        "citations": draft.citations,
        "valuation": None,
        "current_step": "analysis",
    }
    if failures:
        _append_error(state, update, "Metric calculation failed for " + "; ".join(failures))
    try:
        update["valuation"] = value_company(draft.valuation, current_price=price)
    except ValueError as exc:
        _append_error(state, update, f"Valuation failed: {exc}")
    return update

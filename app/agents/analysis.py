"""Financial Analysis Agent: computes each required metric in the code executor."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.llm import ask_structured
from app.state import Citation, FinancialMetricResult, InvestmentAgentState
from app.tools.code_executor import run_python

SYSTEM = """You are a buy-side financial analyst. Using only the provided source documents,
compute each required metric. For every metric, write a short standalone Python script
(standard library only) that hard-codes the input figures from the documents and prints only
the final numeric value. Add a citation for every figure you use, quoting the snippet it came
from. If the documents lack the inputs for a metric, omit that metric rather than estimating."""


class _MetricDraft(BaseModel):
    metric_name: str
    formula_used: str = Field(description="Formula in words or symbols, e.g. 'EBITDA / Revenue'")
    code: str = Field(description="Python script that prints only the metric's numeric value")


class _AnalysisDraft(BaseModel):
    metrics: List[_MetricDraft]
    citations: List[Citation]


def format_documents(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "(no documents retrieved)"
    return "\n\n".join(
        f'<document source_doc="{d["doc_id"]}" page_number="{d["metadata"].get("page_number", 0)}">\n'
        f'{d["content"]}\n</document>'
        for d in docs
    )


def analysis_node(state: InvestmentAgentState) -> dict:
    plan = state["plan"]
    prompt = (
        f"Company: {state['company_name']}\n"
        f"Question: {state['user_query']}\n"
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
        "current_step": "analysis",
    }
    if failures:
        issues = "Metric calculation failed for " + "; ".join(failures)
        update["error"] = f"{state['error']}\n{issues}" if state.get("error") else issues
    return update

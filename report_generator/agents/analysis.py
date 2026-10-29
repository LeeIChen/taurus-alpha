"""Financial Analysis Agent: computes each required metric in the code executor."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from report_generator.agents.schemas import run_step
from report_generator.state import FinancialMetricResult, InvestmentAgentState
from core.tools import fred, peer_comps
from report_generator.tools.code_executor import run_python

SYSTEM = """You are a buy-side financial analyst. Using the provided filing pages, SEC-reported
financials and sourced web research notes, compute each required metric. For every metric, write a
short standalone Python script (standard library only) that hard-codes the input figures and prints
only the final numeric value. Add a citation for every figure: for filing pages use the source_doc
and page_number; for web research use the source id (e.g. "S4") as source_doc and 0 as page_number;
for SEC-reported financials use "SEC XBRL" and 0. If no source supports a metric's inputs, omit
the metric rather than estimating. Documents may come from several companies; check each
document's company before using it. Skip any metric the peer_comps or financial_history tables
already give (source X1): those need no code."""


def format_documents(docs: List[Dict[str, Any]]) -> str:
    if not docs:
        return "(no documents retrieved)"
    return "\n\n".join(
        f'<document source_doc="{d["doc_id"]}" company="{d["metadata"].get("company", "")}" '
        f'doc_type="{d["metadata"].get("doc_type", "")}" page_number="{d["metadata"].get("page_number", 0)}">\n'
        f'{d["content"]}\n</document>'
        for d in docs
    )


def market_tables(state: InvestmentAgentState) -> str:
    """Peer table, financial history and macro table (sources X1 and M1), rendered deterministically."""
    md = state.get("market_data") or {}
    if not md:
        return ""
    peers = md.get("peers") or []
    target = next((r for r in peers if r.get("ticker") == (state.get("ticker") or "").upper() and "error" not in r), None)
    return (
        f"<peer_comps source_id=\"X1\" as_of=\"{md.get('as_of')}\">\n{peer_comps.render_table(peers)}\n</peer_comps>\n\n"
        f"<financial_history source_id=\"X1\">\n{peer_comps.render_history(md.get('history') or [], target)}\n"
        f"</financial_history>\n\n"
        f"<macro_data source_id=\"M1\" as_of=\"{md.get('as_of')}\">\n{fred.render_table(md.get('macro') or [])}\n</macro_data>\n\n"
    )


def shared_evidence(state: InvestmentAgentState) -> str:
    """Evidence block shared (and prompt-cached) by every step after retrieval.

    Must be byte-identical across steps: it only reads state that is fixed once retrieval
    finishes, and serializes deterministically (sorted keys, stable order).
    """
    sources = "\n".join(f"[{s.source_id}] {s.title} — {s.url}" for s in state.get("sources") or []) or "(none)"
    return (
        f"<company>{state['company_name']} ({state.get('ticker') or ''})</company>\n"
        f"<question>{state['user_query']}</question>\n\n"
        f"<sec_reported_financials>\n{json.dumps(state.get('reported_financials') or {}, indent=1, sort_keys=True)}\n"
        f"</sec_reported_financials>\n\n"
        f"{market_tables(state)}"
        f"<sources>\n{sources}\n</sources>\n\n"
        f"<web_research_notes>\n{state.get('research_notes') or '(none)'}\n</web_research_notes>\n\n"
        f"<filing_pages>\n{format_documents(state['retrieved_docs'])}\n</filing_pages>"
    )


def _append_error(state: InvestmentAgentState, update: dict, issue: str) -> None:
    previous = update.get("error") or state.get("error")
    update["error"] = f"{previous}\n{issue}" if previous else issue


def analysis_node(state: InvestmentAgentState) -> dict:
    plan = state["plan"]
    prompt = f"Required metrics: {', '.join(plan.required_metrics)}"
    draft = run_step("analysis", SYSTEM, prompt, state, shared_evidence(state))

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

    update: dict = {"financial_results": results, "citations": draft.citations, "current_step": "analysis"}
    if failures:
        _append_error(state, update, "Metric calculation failed for " + "; ".join(failures))
    return update

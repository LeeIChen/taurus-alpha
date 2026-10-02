"""Web Research Agent: gathers recent, sourced facts the filings cannot provide."""

from __future__ import annotations

import logging
from datetime import date

import anthropic

from app.llm import ask_with_web
from app.state import InvestmentAgentState, Source

logger = logging.getLogger("uvicorn.error")

SYSTEM = """You are a senior equity research analyst gathering evidence for an investment
recommendation. Use web search and web fetch to find current, verifiable facts. Prefer primary
sources: company press releases and SEC filings, regulators (FDA, EMA, SEC), trial registries,
peer-reviewed papers, and established financial and trade press. Be economical: search for what
the tasks need, fetch a page only when a search snippet is not enough.

Cite every fact by putting the exact URL of its source in angle brackets right after it, e.g.
"Revenue was $145 million <https://www.sec.gov/...>". Use the URL of the page you actually read.

Write research notes in Markdown with these sections:
1. Recent developments (last 12 months), with dates.
2. Products and pipeline: every marketed product and development program, with stage, indication,
   partner and economics (e.g. profit splits), latest data, and next milestone with expected timing.
   For each pipeline program, give the benchmark probability of success for its phase and therapeutic
   area from published studies (e.g. BIO/Informa/QLS clinical development success rates) and any
   program-specific evidence that should move it up or down.
3. Market size and pricing evidence for the main products (patient numbers, prices, analyst peak
   sales estimates).
4. Financing, balance sheet events and contingent liabilities not yet in the latest filing.
5. Upcoming catalysts with dates or windows.
6. Market data: the latest share price with its date, market capitalization, and sell-side price
   targets and ratings.
7. Regulatory, policy and competitive risks.

State numbers exactly as the source does. If sources disagree, give both. Do not speculate."""


def web_research_node(state: InvestmentAgentState) -> dict:
    plan = state["plan"]
    prompt = (
        f"Today is {date.today().isoformat()}.\n"
        f"Company: {state['company_name']} (ticker {state.get('ticker') or 'unknown'})\n"
        f"Investment question: {state['user_query']}\n\n"
        "Research tasks from the plan:\n" + "\n".join(f"- {t}" for t in plan.tasks)
    )
    try:
        notes, found = ask_with_web(SYSTEM, prompt)
    except anthropic.BadRequestError as exc:  # e.g. web search not enabled for the organization
        logger.warning("Web research unavailable: %s", exc)
        issue = f"Web research unavailable ({exc.status_code}); analysis uses filings only"
        return {"research_notes": "", "current_step": "web_research",
                "error": f"{state['error']}\n{issue}" if state.get("error") else issue}

    sources = list(state.get("sources") or []) + [
        Source(source_id=f"S{i}", title=s["title"], url=s["url"], kind="web", cited=s["cited"], snippet=s["snippet"])
        for i, s in enumerate(found, start=1)
    ]
    logger.info("Web research: %d sources (%d cited)", len(found), sum(s["cited"] for s in found))
    return {"research_notes": notes, "sources": sources, "current_step": "web_research"}


def format_sources(state: InvestmentAgentState, cited_only: bool = False) -> str:
    rows = [s for s in state.get("sources") or [] if not cited_only or s.cited or s.kind == "filing"]
    return "\n".join(f"[{s.source_id}] {s.title} — {s.url}" for s in rows) or "(none)"

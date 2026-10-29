"""Market data: the peer table, the target's financial history and macro series, all computed in code.

Also downloads each peer's latest 10-K and 10-Q so the competition section can search them.
Adds two sources the memo can cite: X1 (peer table and history) and M1 (FRED macro table).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List

from report_generator.state import InvestmentAgentState, Source
from core.tools import rag_search
from core.tools.fred import macro_snapshot
from core.tools.peer_comps import annual_history, peer_table
from core.tools.sec import FILINGS_DIR, SecAccessError, fetch_company_filings, filings_are_fresh, resolve_company

logger = logging.getLogger("uvicorn.error")
MAX_PEERS = 6


def _load_peer_filings(tickers: List[str]) -> List[str]:
    issues = []
    for t in tickers:
        try:
            company = resolve_company(t, t)
            rag_search.register_company(company["company"], company["ticker"])
            paths = sorted((FILINGS_DIR / company["ticker"]).glob("*.jsonl"))
            if not filings_are_fresh(company["ticker"]):
                paths = fetch_company_filings(company["ticker"], company["company"], company["cik"])
            rag_search.add_filing_files(paths)
        except (ValueError, SecAccessError, OSError) as exc:
            issues.append(f"{t}: {exc}")
    return issues


def market_data_node(state: InvestmentAgentState) -> dict:
    plan, ticker = state["plan"], state.get("ticker")
    peers = [t.upper() for t in plan.peer_tickers if t.upper() != (ticker or "").upper()][:MAX_PEERS]
    issues = _load_peer_filings(peers)
    rows = peer_table(([ticker] if ticker else []) + peers)
    history = []
    try:
        history = annual_history(ticker) if ticker else []
    except Exception as exc:  # history is a nice-to-have; the peer row has TTM figures
        issues.append(f"{ticker} history: {exc}")
    macro = macro_snapshot(plan.macro_series)
    today = date.today().isoformat()
    sources = [
        *state.get("sources", []),
        Source(source_id="X1", kind="computed", url="https://www.sec.gov/edgar/sec-api-documentation",
               title=f"Peer table and financial history computed from SEC XBRL company facts and closing prices on {today}"),
        Source(source_id="M1", kind="computed", url="https://fred.stlouisfed.org/",
               title=f"Macro series from FRED, Federal Reserve Bank of St. Louis, retrieved {today}"),
    ]
    logger.info("Market data: %d peer rows, %d macro series, %d history years", len(rows), len(macro), len(history))
    update = {"market_data": {"peers": rows, "history": history, "macro": macro, "as_of": today},
              "sources": sources, "current_step": "market_data"}
    if issues:
        issue = "Peer filings unavailable: " + "; ".join(issues)
        update["error"] = f"{state['error']}\n{issue}" if state.get("error") else issue
    return update

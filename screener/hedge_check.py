"""Hedge-fund check on one shortlisted company: the parts of the five questions the numbers can't answer.

hedge.py measures what the filings and prices show. This check adds what a fund's analyst would look up:
- bottleneck: is what it sells scarce (lead times, queues, permits), or can buyers switch?
- revenue quality: how much revenue is under multi-year contracts, for how long, and how much floats with
  spot prices (the 10-K pages on backlog, contracts and customers are searched first: RAG);
- what's priced in: are analysts raising or cutting estimates?
- catalysts in the next 6-12 months (earnings, contract signings, regulatory rulings) and crowding
  (short interest, how popular the trade is);
- what would prove a long thesis wrong.

One Sonnet call with web search reads the 10-K pages and the screener's numbers; a second extracts the
fields. Cached for 7 days in data/screener/hedge_checks/ (catalysts and estimates go stale fast).
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.llm import WRITER_MODEL, ask_json, ask_with_web
from screener import evidence
from screener.universe import DATA_DIR

logger = logging.getLogger("uvicorn.error")

CHECK_DIR = DATA_DIR / "hedge_checks"
CACHE_DAYS = 7
EST_USD_PER_COMPANY = 0.25  # measured ~$0.15 with 5 searches (NRG, GEV, WMB); 8 searches add search fees and tokens
TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 8},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 2},
]
QUERIES = {
    "backlog": "remaining performance obligations backlog contracted revenue expected to be recognized",
    "contracts": "long-term contracts term years take-or-pay fixed price power purchase agreement",
    "customers": "largest customers customer concentration percent of revenue",
    "pricing": "market prices commodity price exposure hedging spot",
}
MAX_PAGES = 5


class Catalyst(BaseModel):
    when: str = Field(description="Date or period, e.g. 2026-11-05 or 2027 上半年")
    event: str
    impact: Literal["正面", "負面", "不確定"]


class HedgeCheck(BaseModel):
    bottleneck: Literal["關鍵瓶頸", "部分瓶頸", "非瓶頸"]
    bottleneck_rationale: str
    contracted_revenue_pct: Optional[float] = Field(description="Share of revenue under multi-year contracts, percent; null if unknown")
    contract_years: Optional[float] = Field(description="Typical or weighted remaining contract length in years; null if unknown")
    spot_exposure: Literal["高", "中", "低"] = Field(description="How much revenue or margin floats with market prices")
    revenue_quality_rationale: str
    estimate_trend: Literal["上修", "持平", "下修", "不明"] = Field(description="Direction of analysts' EPS estimates and price targets over the last 3 months")
    estimate_rationale: str
    short_interest_pct: Optional[float] = Field(description="Short interest as percent of float, if found")
    crowding_notes: str
    catalysts: List[Catalyst] = Field(description="1-5 dated events in the next 12 months")
    kill_criteria: List[str] = Field(description="2-4 observable facts that would prove a long thesis wrong")
    verdict: str = Field(description="2-3 sentences: is the opportunity mispriced, and why or why not")


SYSTEM = """You are a fundamental analyst at a long/short equity hedge fund preparing a one-page check
on a company for a portfolio manager. Answer five questions: who holds the bottleneck in its value chain;
how much of its revenue is contracted, for how long, versus exposed to spot prices; whether analysts'
estimates are rising or falling and what the price implies (the screener's reverse DCF is given); whether
it earns its cost of capital at today's interest rates; and which dated catalysts are coming and how
crowded the stock is (short interest, popularity).

Use the 10-K pages for contracts and backlog, and web search for news, estimates, short interest and event
dates from the last 3 months. Cite every web fact with its URL in angle brackets, like <https://...>, and
10-K facts with the page id in square brackets. Say "not found" rather than guessing a number. Write in
Traditional Chinese (繁體中文), concise, in short sections for the five questions, then kill criteria and a verdict."""

EXTRACT = """Extract the fields from these analyst notes. Keep the notes' own judgments; use null for numbers
the notes didn't find. Rationales in Traditional Chinese, 1-2 sentences each, keeping the [S#] and page
citations that support them."""


def _pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _num(v: Optional[float], d: int = 1) -> str:
    return "n/a" if v is None else f"{v:,.{d}f}"


def company_brief(row: Dict[str, Any], profile: Optional[Dict[str, Any]], regime: Dict[str, Any]) -> str:
    p = profile or {}
    return "\n".join([
        f"Company: {row['name']} ({row['ticker']}), {row['sub_industry']}, market cap ${(row.get('market_cap') or 0) / 1e9:,.1f}bn, "
        f"price ${_num(row.get('price'), 2)}; today is {date.today().isoformat()}",
        f"Industry labels: {p.get('market_structure')}, bottleneck {p.get('bottleneck')}, AI role {p.get('ai_role')}",
        f"10-year Treasury {regime.get('ten_year')}%. WACC {_pct(row.get('wacc'))}, ROIC {_pct(row.get('roic'))} "
        f"(spread {_pct(row.get('return_spread'))}), interest cover {_num(row.get('interest_coverage'))}x, "
        f"net debt/EBITDA {_num(row.get('net_debt_to_ebitda'))}, buybacks+dividends funded with debt: {row.get('levered_payout')}",
        f"Reverse DCF on {row.get('dcf_base')}: the price implies {_pct(row.get('implied_growth'))} a year for 10 years vs "
        f"{_pct(row.get('delivered_growth'))} revenue growth delivered (3-5y) -> {row.get('priced_in')}",
        f"Backlog (RPO) {_num(row.get('rpo_coverage'), 2)} years of revenue, growing {_pct(row.get('rpo_growth'))}; "
        f"gross margin {_pct(row.get('gross_margin'))} vs 5-year median {_pct(row.get('gross_margin_5y_median'))}",
        f"P/E {_num(row.get('pe'))}, EV/EBITDA {_num(row.get('ev_ebitda'))}, 12-month momentum {_pct(row.get('momentum_12m'))}, "
        f"{_pct(row.get('drawdown_12m'))} from its 12-month high; latest quarter revenue growth {_pct(row.get('revenue_growth_latest_q'))}",
    ])


def _path(ticker: str):
    return CHECK_DIR / f"{ticker}.json"


def cached(ticker: str) -> Optional[Dict[str, Any]]:
    path = _path(ticker)
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_DAYS * 86400:
        return json.loads(path.read_text())
    return None


def check(row: Dict[str, Any], profile: Optional[Dict[str, Any]], regime: Dict[str, Any],
          refresh: bool = False) -> Dict[str, Any]:
    ticker = row["ticker"]
    if not refresh and (hit := cached(ticker)):
        return hit
    try:
        pages = evidence.filing_pages(ticker, row["name"], row["cik"], queries=QUERIES, max_pages=MAX_PAGES)
    except Exception as exc:  # the web half still works without the filing
        logger.warning("No 10-K pages for %s: %s", ticker, exc)
        pages = []
    docs = "\n\n".join(f'<document source_doc="{d["doc_id"]}">\n{d["content"]}\n</document>' for d in pages)
    prompt = f"{company_brief(row, profile, regime)}\n\n10-K pages:\n{docs or '(none found)'}"
    notes, sources = ask_with_web(SYSTEM, prompt, max_tokens=16000, model=WRITER_MODEL, effort="low", tools=TOOLS)
    fields = ask_json(EXTRACT, notes, HedgeCheck, model=WRITER_MODEL, effort="low")
    page_urls = {d["doc_id"]: d["metadata"].get("source_url") for d in pages}
    out = {**fields.model_dump(), "notes": notes, "sources": [{"n": i, **s} for i, s in enumerate(sources, 1) if s.get("cited")],  # [S#] numbers
           "pages": page_urls, "ticker": ticker, "model": WRITER_MODEL,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    CHECK_DIR.mkdir(parents=True, exist_ok=True)
    _path(ticker).write_text(json.dumps({**out, "prompt": prompt}, ensure_ascii=False, indent=1))
    return out

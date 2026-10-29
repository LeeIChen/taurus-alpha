"""Deep dive on a shortlisted company: check the industry-level labels against its own 10-K (RAG).

The latest 10-K is downloaded from EDGAR into data/filings/<TICKER>/ (shared with the report
generator), added to the hybrid search index, and searched for competition, customer concentration,
suppliers, pricing and market-size passages. One Sonnet call reads those pages and returns the
company's moat and bargaining power, each claim cited to a page. Cached for 30 days.

These cited labels are also the training examples for the LoRA labeler.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.llm import WRITER_MODEL, ask_json
from core.tools.rag_search import add_filing_files, default_index
from core.tools.sec import fetch_company_filings
from screener.universe import DATA_DIR

logger = logging.getLogger("uvicorn.error")

DEEP_DIR = DATA_DIR / "deep_dives"
CACHE_DAYS = 30
EST_USD_PER_COMPANY = 0.08
PAGES_PER_QUERY = 2
MAX_PAGES = 8

QUERIES = {
    "competition": "competition competitors compete market share principal competitive factors",
    "customers": "largest customers customer concentration percent of net sales revenue",
    "suppliers": "suppliers single source sole source raw materials supply agreements",
    "pricing": "pricing price increases pass through cost inflation pricing power",
    "market": "addressable market demand growth opportunity industry trends",
}


class Evidence(BaseModel):
    claim: str
    page_id: str = Field(description="source_doc id of the page that supports the claim")


class CompanyCheck(BaseModel):
    moat: Literal["寬", "窄", "無"]
    moat_rationale: str
    supplier_power: Literal["強", "中", "弱"] = Field(description="The company's power over its suppliers")
    supplier_power_rationale: str
    customer_power: Literal["強", "中", "弱"] = Field(description="The company's power over its customers")
    customer_power_rationale: str
    market_position: str = Field(description="Its rank or share in its main markets, if the filing says")
    red_flags: List[str] = Field(description="0-3 items: customer concentration, sole-source risk, price pressure...")
    evidence: List[Evidence] = Field(description="2-6 cited claims behind the labels")


SYSTEM = """You check a screener's industry-level labels against one company's own 10-K. Use only the
filing pages and figures given. Write rationales in Traditional Chinese (繁體中文), 1-2 sentences each.
- moat 寬/窄/無: a durable advantage (scale, network, switching costs, brand, patents, regulation) that the
  filing and the margins support.
- supplier_power / customer_power: this company's own bargaining position, which may differ from the
  industry's typical one. Customer concentration (e.g. one customer > 10% of revenue) weakens customer_power;
  sole-source inputs weaken supplier_power.
- Every evidence item cites the page id (source_doc) it comes from. If the pages don't address something,
  say so in the rationale rather than guessing."""


def _cache_path(ticker: str):
    return DEEP_DIR / f"{ticker}.json"


def cached(ticker: str) -> Optional[Dict[str, Any]]:
    path = _cache_path(ticker)
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_DAYS * 86400:
        return json.loads(path.read_text())
    return None


def filing_pages(ticker: str, company: str, cik: int, queries: Optional[Dict[str, str]] = None,
                 max_pages: int = MAX_PAGES) -> List[Dict[str, Any]]:
    paths = fetch_company_filings(ticker, company, cik, forms=["10-K"])
    add_filing_files(paths)
    seen, pages = set(), []
    for q in (queries or QUERIES).values():
        for d in default_index.search(q, filters={"ticker": [ticker], "doc_type": ["10-K"]}, top_k=PAGES_PER_QUERY):
            if d["doc_id"] not in seen:
                seen.add(d["doc_id"])
                pages.append(d)
    return pages[:max_pages]


def _fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None:
        return "n/a"
    return f"{v * 100:.1f}%" if pct else f"{v:,.1f}"


def deep_dive(row: Dict[str, Any], profile: Optional[Dict[str, Any]], refresh: bool = False) -> Dict[str, Any]:
    ticker = row["ticker"]
    if not refresh and (hit := cached(ticker)):
        return hit
    pages = filing_pages(ticker, row["name"], row["cik"])
    if not pages:
        raise ValueError(f"No 10-K pages found for {ticker}")
    docs = "\n\n".join(f'<document source_doc="{d["doc_id"]}" page_number="{d["metadata"].get("page_number")}">\n'
                       f'{d["content"]}\n</document>' for d in pages)
    p = profile or {}
    prompt = (
        f"Company: {row['name']} ({ticker}), {row['sub_industry']}\n"
        f"Industry labels from the screener: market structure {p.get('market_structure')}, typical supplier power "
        f"{p.get('supplier_power')}, typical customer power {p.get('customer_power')}\n"
        f"Company figures: gross margin {_fmt(row.get('gross_margin'))} (sub-industry percentile "
        f"{_fmt(row.get('pricing_power_pct'))}), operating margin {_fmt(row.get('operating_margin'))}, ROIC "
        f"{_fmt(row.get('roic'))}, receivable days {_fmt(row.get('dso'), False)}, payable days {_fmt(row.get('dpo'), False)}\n\n"
        f"10-K pages:\n{docs}"
    )
    check = ask_json(SYSTEM, prompt, CompanyCheck, model=WRITER_MODEL, effort="medium")
    valid = {d["doc_id"]: d["metadata"].get("source_url") for d in pages}
    out = check.model_dump()
    out["evidence"] = [{**e, "url": valid.get(e["page_id"])} for e in out["evidence"] if e["page_id"] in valid]
    out.update(ticker=ticker, filing=pages[0]["metadata"].get("period_of_report"), model=WRITER_MODEL,
               pages=[d["doc_id"] for d in pages], at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    DEEP_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(ticker).write_text(json.dumps({**out, "prompt": prompt}, ensure_ascii=False, indent=1))
    return out

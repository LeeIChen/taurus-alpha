"""Section writers: each memo section retrieves its own evidence and is written in parallel.

thesis -> sections --Send x8--> write_section -> assemble -> editor

Every section sees the same compact core (thesis, valuation, metrics, XBRL financials, the peer,
history and macro tables, the source list) plus the filing pages and research-note paragraphs
retrieved for that section's own queries, instead of the whole evidence block. assemble() adds the
title block, numbered headings and the source list in code, so the outline is always the same.
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List

from langgraph.types import Send

from app import reward
from app.agents.analysis import format_documents, market_tables
from app.agents.schemas import MEMO_READ_TIMEOUT
from app.agents.writer import valuation_payload
from app.llm import WRITER_EFFORT, WRITER_MODEL, ask_text
from app.state import InvestmentAgentState
from app.tools.rag_search import InMemoryIndex, canonical_company, rag_search

logger = logging.getLogger("uvicorn.error")

TARGET_PAGES = 12  # filing pages per section from the target company
PEER_PAGES = 2  # per peer and query, for sections that compare
NOTE_PARAGRAPHS = 14


@dataclass(frozen=True)
class Section:
    key: str
    title_en: str
    title_zh: str
    words: int
    brief: str
    queries: List[str]  # filing searches; {company} is the target, {peer} each peer
    peer_queries: List[str] = field(default_factory=list)


SECTIONS = [
    Section("summary", "Summary", "投資摘要", 180,
            "The title block above already gives rating, target, price and implied move; do not repeat it. "
            "Start with a key-data table (market cap, enterprise value, the valuation multiple that matters, "
            "52-week range, Street average target, next earnings date). Then 3-4 bullets with the core arguments, "
            "one sentence each with its key number. End with one sentence on where we could be wrong: the bull "
            "or bear case and its value per share.",
            ["{company} outlook guidance", "{company} results of operations overview"]),
    Section("company", "Company and latest results", "公司概況與最新財務表現", 300,
            "Open with where the business stands in one sentence. One paragraph on what the company does and how "
            "it makes money (brands, segments, business model). A table from financial_history (fiscal years and "
            "TTM). The latest quarter's operating highlights as 4-6 bullets with numbers (growth by segment, "
            "customers, orders, margins, cash flow). A short balance sheet table (cash, debt, net debt, equity, "
            "shares). Close with management's guidance.",
            ["{company} business overview segments brands", "{company} results of operations net revenue",
             "{company} key operating metrics customers orders", "{company} liquidity capital resources debt",
             "{company} guidance outlook"]),
    Section("industry", "Industry", "產業分析", 300,
            "Open with how attractive the industry is in one sentence. Market size and structure as a small table "
            "(size, growth, and the sector's key structural metric such as online penetration) with sources. "
            "3-5 bullets on the industry's defining traits. A Porter's five forces table (force, strength "
            "high/medium/low, one-line reason). End with one line on the implication for the company.",
            ["{company} industry market opportunity", "{company} competition market share trends",
             "{company} seasonality demand factors", "{company} suppliers sourcing tariffs supply chain"],
            ["{peer} industry market trends demand"]),
    Section("value_chain", "Value chain", "價值鏈分析", 250,
            "Open with where in the value chain the company captures value. Describe the chain from suppliers to "
            "the customer and which steps the company owns. A table of where each $100 of revenue went in the "
            "latest quarter (cost of goods and fulfillment, advertising, other operating costs, operating profit; "
            "adjusted EBITDA if reported), computed from the filings. 3-4 bullets: moats, the biggest costs, new "
            "profit pools, where risks sit. End with one line on the implication.",
            ["{company} suppliers sourcing manufacturing", "{company} logistics fulfillment delivery network",
             "{company} cost of goods sold shipping", "{company} advertising customer acquisition",
             "{company} revenue recognition marketplace platform"]),
    Section("competition", "Competition", "競爭對手分析", 200,
            "Open with one sentence comparing the company with its peers. One table: rows are the company, its "
            "listed peers (figures from peer_comps) and key unlisted rivals from the research notes; columns are "
            "positioning, revenue scale, latest growth, margin, valuation multiple and threat to the company. "
            "Then strengths and weaknesses as two short lists, and how the company differentiates. End with one "
            "line on what this means for the company's multiple.",
            ["{company} competition competitors", "{company} competitive strengths"],
            ["{peer} net revenues growth gross margin", "{peer} competitive strengths stores"]),
    Section("macro", "Macro", "總體經濟分析", 250,
            "Open with whether macro is a headwind or tailwind and for how long. An indicator table from "
            "macro_data and the research notes: indicator, latest value with date, trend, impact on the company. "
            "Then 3-4 short points interpreting the drivers that matter most for this company. End with the "
            "macro conclusion for the next 2-3 quarters and the medium term.",
            ["{company} macroeconomic conditions interest rates consumer spending",
             "{company} housing market inflation risk factors"]),
    Section("valuation", "Valuation", "合理股價估值", 350,
            "Open with the fair value, its range, the method blend and the implied move. The assumptions, DCF "
            "value, P/E value and blended target come from the valuation output: use them exactly. Subsections: "
            "DCF (assumptions table labelled as assumptions, then the sensitivity table from dcf_sensitivity); a "
            "peer-multiple cross-check from peer_comps (which multiple, which peers, implied value per share); "
            "scenarios (bull, base, bear with probability, key assumptions and value per share, and the "
            "probability-weighted value) where the inputs support them; what the current price implies. For a "
            "sum of the parts, add the program table.",
            ["{company} free cash flow capital expenditures", "{company} convertible notes debt maturity",
             "{company} stock-based compensation diluted shares"]),
    Section("catalysts_risks", "Catalysts, risks and conclusion", "催化劑、風險與結論", 250,
            "A dated catalyst table (date, event, what to watch, direction), chronological, at most 8 rows, "
            "including macro events (central bank meetings, data releases, policy dates) and company events. Then "
            "upside risks and downside risks as two lists of at most 4 bullets each. End with a 2-3 sentence "
            "conclusion restating the rating, the fair value and what would change the view.",
            ["{company} risk factors", "{company} legal proceedings contingencies", "{company} upcoming events outlook"]),
]
SECTION_KEYS = [s.key for s in SECTIONS]
_BY_KEY = {s.key: s for s in SECTIONS}

SECTION_SYSTEM = """You write one section of an equity research memo for a portfolio manager who
reads dozens a week. Clear and short beats complete. Write in {language}.

Rules:
- Open with the section's conclusion in one sentence, then the support.
- Stay inside this section's brief; other sections cover the rest, so do not restate the thesis.
- Write for the reader, not about the process: no "we searched", "not retrieved" or task lists. If a
  figure is missing, leave it out or mark the table cell "n/a".
- Plain sentences; bullets only for parallel items; tables for numbers. Stay within the word budget
  (tables excluded; for Chinese, about 1.6 characters per word).
- Ground every statement in the inputs. Cite once at the end of the sentence or short paragraph it
  supports, as [S#], [F#], [X1] (peer table and history), [M1] (macro table) or [source_doc p.N].
- The rating, target and valuation figures are fixed by the inputs; never change them.
- Return only the section body in Markdown, without the section heading; ### subheadings are fine."""


def is_chinese(text: str) -> bool:
    return bool(re.search(r"[㐀-鿿]", text or ""))


def _core(state: InvestmentAgentState) -> str:
    """What every section sees: thesis, valuation, metrics, financials, computed tables, sources."""
    thesis = state.get("thesis")
    sources = "\n".join(f"[{s.source_id}] {s.title}" for s in state.get("sources") or [])
    metrics = [m.model_dump(exclude={"code_executed"}) for m in state.get("financial_results") or []]
    return (
        f"<company>{state['company_name']} ({state.get('ticker') or ''})</company>\n"
        f"<question>{state['user_query']}</question>\n\n"
        f"<thesis>\n{thesis.model_dump_json(indent=1) if thesis else '(none)'}\n</thesis>\n\n"
        f"<valuation>\n{json.dumps(valuation_payload(state), indent=1, default=str)}\n</valuation>\n\n"
        f"<computed_metrics>\n{json.dumps(metrics, indent=1)}\n</computed_metrics>\n\n"
        f"<sec_reported_financials>\n{json.dumps(state.get('reported_financials') or {}, indent=1, sort_keys=True)}\n"
        f"</sec_reported_financials>\n\n{market_tables(state)}<sources>\n{sources}\n</sources>"
    )


def section_pages(state: InvestmentAgentState, section: Section) -> List[Dict[str, Any]]:
    """Filing pages for this section's queries: the target's first, then each peer's."""
    company = canonical_company(state["company_name"])
    peers = [r["company"] for r in (state.get("market_data") or {}).get("peers", [])
             if "error" not in r and r.get("ticker") != (state.get("ticker") or "").upper()]
    seen: Dict[str, Dict[str, Any]] = {}
    for q in section.queries:
        for doc in rag_search(q.format(company=company), company=company, top_k=4):
            if doc["doc_id"] not in seen or doc["score"] > seen[doc["doc_id"]]["score"]:
                seen[doc["doc_id"]] = doc
    target = sorted(seen.values(), key=lambda d: -d["score"])[:TARGET_PAGES]
    peer_docs: Dict[str, Dict[str, Any]] = {}
    for peer in peers:
        for q in section.peer_queries:
            for doc in rag_search(q.format(peer=peer), company=peer, top_k=PEER_PAGES):
                peer_docs.setdefault(doc["doc_id"], doc)
    return target + list(peer_docs.values())


def note_paragraphs(notes: str, section: Section, company: str) -> str:
    """The research-note paragraphs most relevant to this section, in their original order."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\n(?=[-*] |#)", notes or "") if p.strip()]
    if len(paragraphs) <= NOTE_PARAGRAPHS:
        return "\n\n".join(paragraphs) or "(none)"
    index = InMemoryIndex()  # BM25 only: notes are small and change every run
    for i, p in enumerate(paragraphs):
        index.add(str(i), p)
    query = " ".join(q.format(company=company, peer="") for q in section.queries + section.peer_queries)
    query += " " + section.title_en + " " + section.brief
    keep = sorted(int(d["doc_id"]) for d in index.search(query, top_k=NOTE_PARAGRAPHS))
    return "\n\n".join(paragraphs[i] for i in keep)


def _prompt(state: InvestmentAgentState, section: Section, pages: List[Dict[str, Any]]) -> str:
    zh = is_chinese(state["user_query"])
    return (
        f"{_core(state)}\n\n"
        f"<web_research_notes_for_this_section>\n"
        f"{note_paragraphs(state.get('research_notes') or '', section, state['company_name'])}\n"
        f"</web_research_notes_for_this_section>\n\n"
        f"<filing_pages_for_this_section>\n{format_documents(pages)}\n</filing_pages_for_this_section>\n\n"
        f"Write section \"{section.title_zh if zh else section.title_en}\" (about {section.words} words "
        f"plus tables).\nBrief: {section.brief}"
    )


def sections_node(state: InvestmentAgentState) -> dict:
    """Clears the previous sections so a resumed run starts them fresh."""
    return {"memo_sections": None, "section_docs": None, "section_candidates": None}


def fan_out(state: InvestmentAgentState) -> List[Send]:
    return [Send("write_section", {**state, "section": key}) for key in SECTION_KEYS]


def write_section(payload: Dict[str, Any]) -> dict:
    section = _BY_KEY[payload["section"]]
    pages = section_pages(payload, section)
    system = SECTION_SYSTEM.format(language="Traditional Chinese (Taiwan)" if is_chinese(payload["user_query"]) else
                                   "the language of the question")
    prompt = _prompt(payload, section, pages)
    n = reward.best_of()
    def draft() -> str:
        return ask_text(system, prompt, read_timeout=MEMO_READ_TIMEOUT, model=WRITER_MODEL, effort=WRITER_EFFORT).strip()

    # Each draft runs in a copy of this context, so the run's cost meter and step name still apply.
    with ThreadPoolExecutor(max_workers=n) as pool:
        drafts = [f.result() for f in [pool.submit(contextvars.copy_context().run, draft) for _ in range(n)]]
    if not any(drafts):
        raise ValueError(f"Section {section.key} came back empty")
    update: dict = {"section_docs": {section.key: pages}}
    best = 0
    if n > 1:
        try:
            scores = reward.score(f"Company: {payload['company_name']}\nSection: {section.title_en}\n"
                                  f"Question: {payload['user_query']}", drafts)
            best = max(range(n), key=scores.__getitem__)
        except (OSError, ValueError, KeyError) as exc:
            scores = [None] * n
            logger.warning("Reward model failed for %s (%s); keeping the first draft", section.key, exc)
        update["section_candidates"] = {section.key: [{"text": d, "score": s, "chosen": i == best}
                                                      for i, (d, s) in enumerate(zip(drafts, scores))]}
    logger.info("Section %s: %d pages, %d draft(s)", section.key, len(pages), n)
    update["memo_sections"] = {section.key: drafts[best]}
    return update


_CITE = re.compile(r"\[([^\[\]]+)\]")
_SOURCE_ID = re.compile(r"\b([SF]\d+|X1|M1)\b")
_RATING_ZH = {"buy": "買進", "hold": "持有", "sell": "賣出"}


def cited_ids(text: str) -> List[str]:
    ids = {m for c in _CITE.findall(text) for m in _SOURCE_ID.findall(c)}
    return sorted(ids, key=lambda s: (s[0] not in "F", s[0], int(re.sub(r"\D", "", s) or 0)))


_DOC_REF = re.compile(r"\b(F\d+|[A-Z]{1,5}-10-[KQ]-\d{4}-\d{2}-\d{2})\b")
_PAGE = re.compile(r"p\.?\s*(\d+)(?:\s*[-–]\s*(\d+))?")


def cited_pages(memo: str, sources: List[Any]) -> set:
    """Filing page ids cited in the memo, e.g. [F2 p.27, p.28] or [WSM-10-Q-2026-08-02 p.17]."""
    # Filing sources are titled like "W 10-Q for period 2026-06-30 (SEC EDGAR)"; page ids are "W-10-Q-2026-06-30-p28".
    prefix = {s.source_id: s.title.split(" (SEC")[0].replace(" for period ", "-").replace(" ", "-")
              for s in sources if s.kind == "filing"}
    pages = set()
    for cite in _CITE.findall(memo):
        for part in re.split(r";", cite):
            doc = _DOC_REF.search(part)
            if not doc:
                continue
            base = prefix.get(doc.group(1), doc.group(1))
            for a, b in _PAGE.findall(part[doc.end():]):
                pages.update(f"{base}-p{n}" for n in range(int(a), int(b or a) + 1) if int(b or a) - int(a) < 20)
    return pages


def title_block(state: InvestmentAgentState) -> str:
    t, zh = state.get("thesis"), is_chinese(state["user_query"])
    name, ticker = state["company_name"], state.get("ticker") or ""
    if not t:
        return f"# {name} ({ticker})"
    price = t.current_price or state.get("current_price")
    move = (t.price_target / price - 1) * 100 if price else None
    if zh:
        rating = _RATING_ZH.get(t.rating.lower(), t.rating)
        line = [f"評等：{rating}", f"12 個月目標價：{t.price_target:,.2f} 美元"]
        if price:
            line += [f"現價：{price:,.2f} 美元（{t.current_price_source}）",
                     f"潛在{'上漲' if move >= 0 else '下跌'}空間：約 {abs(move):.1f}%"]
        return f"# {name}（{ticker}）股票研究報告：{rating}\n\n" + "｜".join(line)
    line = [f"Rating: {t.rating}", f"12-month target: ${t.price_target:,.2f}"]
    if price:
        line += [f"Price: ${price:,.2f} ({t.current_price_source})", f"{'Upside' if move >= 0 else 'Downside'}: {abs(move):.1f}%"]
    return f"# {name} ({ticker}): {t.rating}, 12-month target ${t.price_target:,.2f}\n\n" + " | ".join(line)


def assemble_memo(state: InvestmentAgentState) -> str:
    zh = is_chinese(state["user_query"])
    parts = state.get("memo_sections") or {}
    missing = [k for k in SECTION_KEYS if not parts.get(k)]
    if missing:
        raise ValueError(f"Sections missing: {missing}")
    body = [title_block(state)]
    for i, s in enumerate(SECTIONS, 1):
        body.append(f"## {i}. {s.title_zh if zh else s.title_en}\n\n{parts[s.key].strip()}")
    memo = "\n\n".join(body)
    by_id = {s.source_id: s for s in state.get("sources") or []}
    refs = [f"{n}. [{sid}] {by_id[sid].title} — {by_id[sid].url}" for n, sid in
            enumerate((sid for sid in cited_ids(memo) if sid in by_id), 1)]
    memo += f"\n\n## {'來源' if zh else 'Sources'}\n\n" + ("\n".join(refs) or "-")
    memo += "\n\n" + ("*本報告僅供研究參考，不構成投資建議。*" if zh else
                      "*Research support for internal discussion, not investment advice.*")
    return memo


def assemble_node(state: InvestmentAgentState) -> dict:
    return {"draft_report": assemble_memo(state), "edit_notes": [], "revisions": 0, "draft_versions": [],
            "edit_history": [], "current_step": "assemble"}

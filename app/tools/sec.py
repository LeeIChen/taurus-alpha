"""SEC EDGAR access: resolve a company, fetch its latest 10-K/10-Q into page chunks,
and read key reported financials from XBRL company facts.

Filing documents (www.sec.gov) require a User-Agent with a contact email:
    SEC_USER_AGENT="taurus-alpha you@example.com"
XBRL facts and submissions (data.sec.gov) are read with the same header.

page_number is the page's position in the filing document (page breaks in the
HTML), which matches the printed page number only approximately.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from datetime import date
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("uvicorn.error")

ROOT = Path(__file__).resolve().parent.parent.parent
FILINGS_DIR = Path(os.environ.get("TAURUS_FILINGS_DIR", ROOT / "data" / "filings"))
SEC_CACHE_DIR = ROOT / "data" / "sec"
REQUEST_INTERVAL_S = 0.2  # SEC fair-access limit is 10 requests/second
MAX_CHUNK_WORDS = 1200
REFRESH_AFTER_DAYS = 30  # re-check EDGAR for newer filings after this long

# Short display names for well-known filers; others use a cleaned SEC title.
KNOWN_COMPANIES = {
    "AAPL": ("Apple", 320193),
    "MSFT": ("Microsoft", 789019),
    "GOOGL": ("Alphabet", 1652044),
    "AMZN": ("Amazon", 1018724),
    "META": ("Meta", 1326801),
    "NVDA": ("Nvidia", 1045810),
    "TSLA": ("Tesla", 1318605),
}

_lock = threading.Lock()


class SecAccessError(RuntimeError):
    pass


def _user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", "")


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": _user_agent() or "taurus-alpha"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 403 and "@" not in _user_agent():
            raise SecAccessError("SEC refused the request; set SEC_USER_AGENT to a name and contact email") from exc
        raise
    time.sleep(REQUEST_INTERVAL_S)
    return data


def _clean_company_title(title: str) -> str:
    title = re.sub(r"(?i)[,.]?\s+(inc|corp|corporation|co|company|ltd|plc|holdings|group|n\.v|s\.a|ag)\.?$", "", title.strip())
    return title.title() if title.isupper() else title


def _ticker_table() -> Dict[str, Dict[str, Any]]:
    """SEC's ticker -> CIK table, cached on disk for a week."""
    path = SEC_CACHE_DIR / "company_tickers.json"
    if not path.exists() or time.time() - path.stat().st_mtime > 7 * 86400:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_get("https://www.sec.gov/files/company_tickers.json"))
    rows = json.loads(path.read_text()).values()
    return {r["ticker"].upper(): {"cik": int(r["cik_str"]), "title": r["title"]} for r in rows}


def resolve_company(company_name: str, ticker: Optional[str] = None) -> Dict[str, Any]:
    """Return {"ticker", "cik", "company"} for a ticker or company name."""
    if ticker and ticker.upper() in KNOWN_COMPANIES:
        name, cik = KNOWN_COMPANIES[ticker.upper()]
        return {"ticker": ticker.upper(), "cik": cik, "company": name}
    for t, (name, cik) in KNOWN_COMPANIES.items():
        if company_name.strip().lower() in (name.lower(), t.lower()):
            return {"ticker": t, "cik": cik, "company": name}
    table = _ticker_table()
    if ticker:
        row = table.get(ticker.upper())
        if not row:
            raise ValueError(f"Ticker {ticker} not found at SEC")
        return {"ticker": ticker.upper(), "cik": row["cik"], "company": _clean_company_title(row["title"])}
    wanted = company_name.strip().lower()
    if wanted.upper() in table:
        row = table[wanted.upper()]
        return {"ticker": wanted.upper(), "cik": row["cik"], "company": _clean_company_title(row["title"])}
    matches = [(t, r) for t, r in table.items() if r["title"].lower().startswith(wanted)]
    if not matches:
        matches = [(t, r) for t, r in table.items() if wanted in r["title"].lower()]
    if not matches:
        raise ValueError(f"No SEC filer matches '{company_name}'; pass a ticker")
    t, row = min(matches, key=lambda m: (len(m[1]["title"]), len(m[0])))
    return {"ticker": t, "cik": row["cik"], "company": _clean_company_title(row["title"])}


def latest_filings(cik: int, forms: List[str]) -> Dict[str, dict]:
    """Return the most recent filing of each requested form type."""
    subs = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    recent = subs["filings"]["recent"]
    found: Dict[str, dict] = {}
    for i, form in enumerate(recent["form"]):  # newest first
        if form in forms and form not in found:
            accession = recent["accessionNumber"][i]
            found[form] = {
                "form": form,
                "filing_date": recent["filingDate"][i],
                "period_of_report": recent["reportDate"][i],
                "url": (
                    f"https://www.sec.gov/Archives/edgar/data/{cik}/"
                    f"{accession.replace('-', '')}/{recent['primaryDocument'][i]}"
                ),
            }
    return found


class _FilingText(HTMLParser):
    """HTML -> text, inserting \\f at CSS page breaks and skipping hidden XBRL headers."""

    _BLOCK = {"p", "div", "br", "tr", "li", "h1", "h2", "h3", "h4", "h5", "h6", "table"}
    _VOID = {"br", "hr", "img", "meta", "link", "input"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._stack: list[tuple[bool, bool]] = []  # per open element: (hidden, page break after)
        self._hidden_depth = 0

    def handle_starttag(self, tag, attrs):
        style = (dict(attrs).get("style") or "").replace(" ", "").lower()
        if "page-break-before:always" in style or "break-before:page" in style:
            self.parts.append("\f")
        if tag in self._BLOCK:
            self.parts.append("\n")
        if tag == "td":
            self.parts.append(" | ")
        if tag in self._VOID:
            if "page-break-after:always" in style or "break-after:page" in style:
                self.parts.append("\f")
            return
        hidden = "display:none" in style or tag == "ix:header"
        break_after = "page-break-after:always" in style or "break-after:page" in style
        self._stack.append((hidden, break_after))
        self._hidden_depth += hidden

    def handle_endtag(self, tag):
        if tag in self._VOID or not self._stack:
            return
        hidden, break_after = self._stack.pop()
        self._hidden_depth -= hidden
        if tag in self._BLOCK:
            self.parts.append("\n")
        if break_after:
            self.parts.append("\f")

    def handle_data(self, data):
        if not self._hidden_depth:
            self.parts.append(data)

    def pages(self) -> list[str]:
        text = "".join(self.parts).replace("\xa0", " ")
        pages = []
        for raw in text.split("\f"):
            lines = [_clean_line(line) for line in raw.splitlines()]
            page = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
            if page:
                pages.append(page)
        return pages


def _clean_line(line: str) -> str:
    line = re.sub(r"[ \t]+", " ", line)
    line = re.sub(r"\|(\s*\|)+", "|", line)          # drop empty table cells
    line = re.sub(r"\$\s*\|\s*", "$", line)          # "$ | 96,221" -> "$96,221"
    line = re.sub(r"\s*\|\s*([)%])", r"\1", line)    # "(1,234 | )" -> "(1,234)"
    return line.strip(" |")


def _split_long(page: str) -> list[str]:
    words = page.split(" ")
    if len(words) <= MAX_CHUNK_WORDS:
        return [page]
    return [" ".join(words[i : i + MAX_CHUNK_WORDS]) for i in range(0, len(words), MAX_CHUNK_WORDS)]


def build_chunks(ticker: str, company: str, filing: dict, html: bytes) -> list[dict]:
    parser = _FilingText()
    parser.feed(html.decode("utf-8", errors="replace"))
    chunks = []
    for page_number, page in enumerate(parser.pages(), start=1):
        for part_index, part in enumerate(_split_long(page)):
            suffix = f"-{part_index}" if part_index else ""
            chunks.append(
                {
                    "doc_id": f"{ticker}-{filing['form']}-{filing['period_of_report']}-p{page_number}{suffix}",
                    "content": part,
                    "metadata": {
                        "company": company,
                        "ticker": ticker,
                        "doc_type": filing["form"],
                        "period_of_report": filing["period_of_report"],
                        "filing_date": filing["filing_date"],
                        "page_number": page_number,
                        "source_url": filing["url"],
                    },
                }
            )
    return chunks


def fetch_company_filings(
    ticker: str, company: str, cik: int, forms: Optional[List[str]] = None, out: Path = FILINGS_DIR
) -> List[Path]:
    """Download the latest filing of each form for one company; returns the JSONL paths written."""
    written = []
    for form, filing in sorted(latest_filings(cik, forms or ["10-K", "10-Q"]).items()):
        path = out / ticker / f"{form}_{filing['period_of_report']}.jsonl"
        if path.exists():
            written.append(path)
            continue
        chunks = build_chunks(ticker, company, filing, _get(filing["url"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")
        _update_manifest(out, {"ticker": ticker, "company": company, **filing,
                               "pages": chunks[-1]["metadata"]["page_number"] if chunks else 0,
                               "chunks": len(chunks), "path": str(path.relative_to(out))})
        logger.info("Fetched %s %s (%s): %d chunks", ticker, form, filing["period_of_report"], len(chunks))
        written.append(path)
    return written


def _update_manifest(out: Path, entry: dict) -> None:
    path = out / "manifest.json"
    manifest = json.loads(path.read_text()) if path.exists() else []
    manifest = [m for m in manifest if not (m["ticker"] == entry["ticker"] and m["form"] == entry["form"])]
    manifest.append(entry)
    path.write_text(json.dumps(sorted(manifest, key=lambda m: (m["ticker"], m["form"])), indent=2) + "\n")


def filings_are_fresh(ticker: str, out: Path = FILINGS_DIR) -> bool:
    files = list((out / ticker).glob("*.jsonl"))
    if not files:
        return False
    newest = max(f.stat().st_mtime for f in files)
    return time.time() - newest < REFRESH_AFTER_DAYS * 86400


# ---- XBRL reported financials ----
_FACTS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "research_and_development": ["ResearchAndDevelopmentExpense"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "diluted_eps": ["EarningsPerShareDiluted"],
    "cash_and_equivalents": ["CashAndCashEquivalentsAtCarryingValue"],
    "short_term_investments": ["ShortTermInvestments", "MarketableSecuritiesCurrent", "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
    "long_term_investments": ["MarketableSecuritiesNoncurrent", "LongTermInvestments", "AvailableForSaleSecuritiesDebtSecuritiesNoncurrent"],
    "long_term_debt": ["LongTermDebt", "LongTermDebtNoncurrent", "ConvertibleNotesPayable"],
}


def reported_financials(cik: int) -> Dict[str, Any]:
    """Latest annual and latest-quarter values for key line items, plus shares outstanding."""
    facts = json.loads(_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"))["facts"]
    gaap, dei = facts.get("us-gaap", {}), facts.get("dei", {})
    out: Dict[str, Any] = {"source": f"SEC XBRL company facts, CIK {cik}", "values": {}}
    for key, tags in _FACTS.items():
        for tag in tags:
            units = gaap.get(tag, {}).get("units", {})
            vals = units.get("USD") or units.get("USD/shares")
            if not vals:
                continue
            annual = [v for v in vals if v.get("form") == "10-K" and v.get("fp") == "FY" and _days(v) > 300]
            latest = max(vals, key=lambda v: (v["end"], v.get("filed", "")))
            entry = {"tag": tag}
            if annual:
                a = max(annual, key=lambda v: (v["end"], v.get("filed", "")))
                entry["latest_annual"] = {"value": a["val"], "period_end": a["end"]}
            entry["latest_reported"] = {"value": latest["val"], "period_end": latest["end"],
                                        "period_start": latest.get("start"), "form": latest.get("form")}
            out["values"][key] = entry
            break
    shares = dei.get("EntityCommonStockSharesOutstanding", {}).get("units", {}).get("shares", [])
    if shares:
        s = max(shares, key=lambda v: v["end"])
        out["values"]["shares_outstanding"] = {"value": s["val"], "as_of": s["end"]}
    return out


def _days(v: dict) -> int:
    if not v.get("start"):
        return 0
    return (date.fromisoformat(v["end"]) - date.fromisoformat(v["start"])).days

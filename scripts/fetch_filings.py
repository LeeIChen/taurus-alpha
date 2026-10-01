"""Download the latest 10-K and 10-Q for the Magnificent 7 from SEC EDGAR.

Writes one JSONL file per filing to data/filings/<TICKER>/, one line per page:
    {"doc_id": ..., "content": ..., "metadata": {company, ticker, doc_type,
     period_of_report, filing_date, page_number, source_url}}

page_number is the page's position in the filing document (page breaks in the
HTML), which matches the printed page number only approximately.

SEC requires a User-Agent that identifies you with a contact email:
    export SEC_USER_AGENT="taurus-alpha you@example.com"

Usage: python scripts/fetch_filings.py [--forms 10-K 10-Q] [--out data/filings]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

COMPANIES = {
    "AAPL": ("Apple", 320193),
    "MSFT": ("Microsoft", 789019),
    "GOOGL": ("Alphabet", 1652044),
    "AMZN": ("Amazon", 1018724),
    "META": ("Meta", 1326801),
    "NVDA": ("Nvidia", 1045810),
    "TSLA": ("Tesla", 1318605),
}

USER_AGENT = os.environ.get("SEC_USER_AGENT", "")
REQUEST_INTERVAL_S = 0.2  # SEC fair-access limit is 10 requests/second
MAX_CHUNK_WORDS = 1200


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    time.sleep(REQUEST_INTERVAL_S)
    return data


def latest_filings(cik: int, forms: list[str]) -> dict[str, dict]:
    """Return the most recent filing of each requested form type."""
    subs = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
    recent = subs["filings"]["recent"]
    found: dict[str, dict] = {}
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--forms", nargs="+", default=["10-K", "10-Q"])
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "filings"))
    args = ap.parse_args()
    out = Path(args.out)
    if "@" not in USER_AGENT:
        raise SystemExit('Set SEC_USER_AGENT to a name and contact email, e.g. "taurus-alpha you@example.com"')

    manifest = []
    for ticker, (company, cik) in COMPANIES.items():
        for form, filing in sorted(latest_filings(cik, args.forms).items()):
            chunks = build_chunks(ticker, company, filing, _get(filing["url"]))
            path = out / ticker / f"{form}_{filing['period_of_report']}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk) + "\n")
            pages = chunks[-1]["metadata"]["page_number"] if chunks else 0
            manifest.append({"ticker": ticker, "company": company, **filing, "pages": pages,
                             "chunks": len(chunks), "path": str(path.relative_to(out))})
            print(f"{ticker:5} {form:5} period {filing['period_of_report']}  filed {filing['filing_date']}  "
                  f"{pages:3} pages  {len(chunks):4} chunks")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()

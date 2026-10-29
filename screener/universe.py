"""S&P 1500 constituents (S&P 500 + MidCap 400 + SmallCap 600) with GICS sector and sub-industry.

Source: the constituent tables on Wikipedia's "List of S&P 500/400/600 companies" pages (wikitext,
no API key), cached in data/screener/universe.json for a week. CIKs come from SEC's ticker table when
it lists the ticker (Wikipedia's otherwise). GICS sub-industry is the industry unit the screener classifies.
"""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List

from core.tools.sec import _ticker_table

logger = logging.getLogger("uvicorn.error")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "screener"
UNIVERSE_PATH = DATA_DIR / "universe.json"
CACHE_DAYS = 7

PAGES = {
    "S&P 500": "List_of_S%26P_500_companies",
    "S&P 400": "List_of_S%26P_400_companies",
    "S&P 600": "List_of_S%26P_600_companies",
}


def _wikitext(page: str) -> str:
    url = f"https://en.wikipedia.org/w/index.php?title={page}&action=raw"
    req = urllib.request.Request(url, headers={"User-Agent": "taurus-alpha research tool"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read().decode("utf-8")


_ATTR = re.compile(r'^\s*(?:[a-z-]+="[^"]*"\s*)+\|(?!\|)')


def _clean(cell: str) -> str:
    cell = _ATTR.sub("", cell)
    cell = re.sub(r"\{\{\s*\w*Symbol\s*\|\s*([^}|]+)[^}]*\}\}", r"\1", cell)
    cell = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", cell)  # [[target|text]] -> text
    cell = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", cell)  # [url text] -> text
    cell = re.sub(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>|\{\{[^}]*\}\}|<[^>]+>", "", cell)
    return cell.replace("&amp;", "&").strip()


def parse_constituents(wikitext: str) -> List[Dict[str, Any]]:
    """Rows of the table with id="constituents": ticker, name, sector, sub-industry, CIK if given."""
    start = wikitext.find('id="constituents"')
    if start < 0:
        raise ValueError("No constituents table found")
    table = wikitext[start: wikitext.find("\n|}", start)]
    rows = []
    for raw in table.split("\n|-")[1:]:
        cells: List[str] = []
        for line in raw.strip("\n").splitlines():
            if not line.startswith("|") or line.startswith(("|-", "|}", "|+")):
                continue
            cells += [c for c in line.lstrip("|").split("||")]
        if len(cells) < 4:
            continue
        # ticker: last upper-case token of the raw cell ({{NyseSymbol|ABT}}, {{BZX link|CBOE}}, [{{...|PRK|...}} PRK])
        symbols = re.findall(r"(?<![\w.])[A-Z][A-Z0-9]{0,5}(?:\.[A-Z])?(?![\w.])", re.sub(r"<!--.*?-->", "", cells[0]))
        cells = [_clean(c) for c in cells]
        # S&P 400 links EDGAR (CIK=...); S&P 500 and 600 have a CIK column, before "Founded" on the 500 page
        cik = re.search(r"CIK=0*(\d+)", raw) or next(
            (m for m in (re.match(r"^0*(\d{4,10})$", c) for c in cells[4:]) if m), None)
        rows.append({
            "ticker": (symbols[-1] if symbols else cells[0]).upper().replace(".", "-"),  # BRK.B -> BRK-B (SEC, Yahoo)
            "name": re.sub(r",? Inc\.?$", "", re.sub(r"\s*\(Class [A-Z]\)", "", cells[1])).strip(),
            "sector": cells[2],
            "sub_industry": cells[3],
            "cik": int(cik.group(1)) if cik else None,
        })
    return rows


def build_universe() -> List[Dict[str, Any]]:
    tickers = _ticker_table()
    out: Dict[str, Dict[str, Any]] = {}
    seen_ciks = set()
    for index, page in PAGES.items():
        rows = parse_constituents(_wikitext(page))
        logger.info("%s: %d constituents", index, len(rows))
        for r in rows:
            sec = tickers.get(r["ticker"])
            if sec:  # SEC's table wins: Wikipedia can list a new holding company whose XBRL history is elsewhere
                r["cik"] = sec["cik"]
            if r["ticker"] in out or (r["cik"] and r["cik"] in seen_ciks):  # one row per company (GOOGL, not GOOG)
                continue
            seen_ciks.add(r["cik"])
            out[r["ticker"]] = {**r, "index": index}
    out = {t: r for t, r in out.items() if re.fullmatch(r"[A-Z][A-Z0-9-]{0,7}", t) or logger.warning("Bad ticker %r", t)}
    missing = [t for t, r in out.items() if not r["cik"]]
    if missing:
        logger.warning("No CIK for %d tickers (dropped): %s", len(missing), ", ".join(missing[:20]))
    return [r for r in out.values() if r["cik"]]


def load_universe(refresh: bool = False) -> List[Dict[str, Any]]:
    if refresh or not UNIVERSE_PATH.exists() or time.time() - UNIVERSE_PATH.stat().st_mtime > CACHE_DAYS * 86400:
        try:
            rows = build_universe()
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            UNIVERSE_PATH.write_text(json.dumps(rows, indent=1))
        except (OSError, ValueError) as exc:
            if not UNIVERSE_PATH.exists():
                raise
            logger.warning("Universe refresh failed (%s); using cached copy", exc)
    return json.loads(UNIVERSE_PATH.read_text())

"""Download the latest 10-K and 10-Q from SEC EDGAR into data/filings/<TICKER>/.

Writes one JSONL file per filing, one line per page:
    {"doc_id": ..., "content": ..., "metadata": {company, ticker, doc_type,
     period_of_report, filing_date, page_number, source_url}}

SEC requires a User-Agent that identifies you with a contact email:
    export SEC_USER_AGENT="taurus-alpha you@example.com"

Usage:
    python scripts/fetch_filings.py                 # the Magnificent 7
    python scripts/fetch_filings.py MRNA PFE        # any tickers
    python scripts/fetch_filings.py --forms 10-K
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.tools.sec import KNOWN_COMPANIES, fetch_company_filings, resolve_company  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=list(KNOWN_COMPANIES))
    ap.add_argument("--forms", nargs="+", default=["10-K", "10-Q"])
    args = ap.parse_args()
    if "@" not in os.environ.get("SEC_USER_AGENT", ""):
        raise SystemExit('Set SEC_USER_AGENT to a name and contact email, e.g. "taurus-alpha you@example.com"')

    for ticker in args.tickers:
        company = resolve_company(ticker, ticker=ticker)
        for path in fetch_company_filings(company["ticker"], company["company"], company["cik"], args.forms):
            print(f"{company['ticker']:6} {company['company']:20} {path.name}")


if __name__ == "__main__":
    main()

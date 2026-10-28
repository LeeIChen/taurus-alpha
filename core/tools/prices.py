"""Daily closing prices for scoring runs against what actually happened.

Source: Yahoo Finance's public chart endpoint (unofficial; it can change without notice).
Responses are cached in data/prices/<TICKER>.json for a day. If a price is missing or the
endpoint fails, add rows to data/prices/manual.csv (ticker,date,close) and they take priority.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
PRICE_DIR = ROOT / "data" / "prices"
CACHE_HOURS = 20


def _manual() -> Dict[str, Dict[str, float]]:
    path = PRICE_DIR / "manual.csv"
    out: Dict[str, Dict[str, float]] = {}
    if path.exists():
        with path.open() as f:
            for row in csv.DictReader(f):
                out.setdefault(row["ticker"].upper(), {})[row["date"]] = float(row["close"])
    return out


def history(ticker: str) -> Dict[str, float]:
    """{ISO date: close} for the last 2 years, manual entries overriding downloaded ones."""
    ticker = ticker.upper()
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    cache = PRICE_DIR / f"{ticker}.json"
    closes: Dict[str, float] = {}
    if cache.exists() and time.time() - cache.stat().st_mtime < CACHE_HOURS * 3600:
        closes = json.loads(cache.read_text())
    else:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?range=2y&interval=1d"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())["chart"]["result"][0]
            stamps = result.get("timestamp") or []
            values = result["indicators"]["quote"][0].get("close") or []
            closes = {
                datetime.fromtimestamp(ts, timezone.utc).date().isoformat(): round(v, 4)
                for ts, v in zip(stamps, values) if v is not None
            }
            cache.write_text(json.dumps(closes))
        except (OSError, KeyError, IndexError, ValueError):
            if cache.exists():  # stale cache beats nothing
                closes = json.loads(cache.read_text())
    closes.update(_manual().get(ticker, {}))
    return dict(sorted(closes.items()))


def close_on_or_before(closes: Dict[str, float], day: date) -> Optional[Tuple[str, float]]:
    """Last close on or before `day` (handles weekends and holidays)."""
    target = day.isoformat()
    best = None
    for d, v in closes.items():
        if d <= target:
            best = (d, v)
        else:
            break
    return best


def latest(closes: Dict[str, float]) -> Optional[Tuple[str, float]]:
    if not closes:
        return None
    d = max(closes)
    return d, closes[d]

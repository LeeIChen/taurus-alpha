"""Share prices and price-based sensitivities for the screener, no model calls.

Monthly adjusted closes for 10 years from Yahoo Finance's public chart endpoint (unofficial; it can
change without notice), cached in data/screener/prices/<TICKER>.json for a day. From them:
- the current price (for market cap and multiples);
- market beta against SPY;
- rate beta: monthly return per 1 percentage point rise in the 10-year yield, holding the market fixed;
- cycle beta: monthly return per 1-unit move in the leading index, holding the market fixed.
"""

from __future__ import annotations

import json
import logging
import time
import http.client
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

logger = logging.getLogger("uvicorn.error")

ROOT = Path(__file__).resolve().parent.parent
PRICE_DIR = ROOT / "data" / "screener" / "prices"
CACHE_HOURS = 20
MARKET = "SPY"
MIN_MONTHS = 36


def _download(ticker: str) -> Optional[Dict[str, Any]]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?range=10y&interval=1mo"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())["chart"]["result"][0]
            stamps = result.get("timestamp") or []
            adj = (result["indicators"].get("adjclose") or [{}])[0].get("adjclose") or []
            closes = {datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m"): round(v, 4)
                      for ts, v in zip(stamps, adj) if v is not None}
            meta = result.get("meta", {})
            return {"price": meta.get("regularMarketPrice"), "currency": meta.get("currency"),
                    "price_time": meta.get("regularMarketTime"), "monthly": closes}
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            time.sleep(2 ** attempt)  # 429: back off
        except http.client.InvalidURL:
            return None
        except (OSError, KeyError, IndexError, ValueError, TypeError):
            time.sleep(2 ** attempt)
    return None


def prices(ticker: str) -> Optional[Dict[str, Any]]:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    path = PRICE_DIR / f"{ticker}.json"
    if path.exists() and time.time() - path.stat().st_mtime < CACHE_HOURS * 3600:
        return json.loads(path.read_text())
    data = _download(ticker)
    if data:
        path.write_text(json.dumps(data))
        return data
    if path.exists():  # stale beats nothing
        return json.loads(path.read_text())
    return None


def fetch_all(tickers: Iterable[str], workers: int = 6, progress=None) -> Dict[str, Optional[Dict[str, Any]]]:
    tickers = list(dict.fromkeys([MARKET, *tickers]))
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, (t, data) in enumerate(zip(tickers, pool.map(prices, tickers)), 1):
            out[t] = data
            if progress and i % 100 == 0:
                progress(i, len(tickers))
    return out


def _returns(monthly: Dict[str, float]) -> Dict[str, float]:
    keys = sorted(monthly)[:-1]  # the current month is incomplete
    return {k: monthly[k] / monthly[p] - 1 for p, k in zip(keys, keys[1:]) if monthly[p] > 0}


def sensitivities(monthly: Dict[str, float], market: Dict[str, float], factors: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """OLS of monthly returns on the market and on each macro factor (market held fixed).
    Returns betas with t-statistics (left out with under three years of data), momentum and drawdown."""
    r, m = _returns(monthly), _returns(market)
    out: Dict[str, Any] = {}
    # 12-month price momentum (skipping the latest month), a common screen input
    keys = sorted(monthly)
    if len(keys) >= 14:
        out["momentum_12m"] = monthly[keys[-2]] / monthly[keys[-14]] - 1
    # how far below its 12-month high (month-end closes, current month included) the stock trades now
    if len(keys) >= 12:
        out["drawdown_12m"] = monthly[keys[-1]] / max(monthly[k] for k in keys[-12:]) - 1
    months = [k for k in r if k in m]
    if len(months) < MIN_MONTHS:
        return out
    y, x_m = np.array([r[k] for k in months]), np.array([m[k] for k in months])
    X = np.column_stack([np.ones(len(months)), x_m])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    out["market_beta"] = float(beta[1])
    out["volatility"] = float(np.std(y) * np.sqrt(12))
    out["months"] = len(months)
    for name, series in factors.items():
        ks = [k for k in months if k in series]
        if len(ks) < MIN_MONTHS:
            continue
        yy = np.array([r[k] for k in ks])
        X = np.column_stack([np.ones(len(ks)), [m[k] for k in ks], [series[k] for k in ks]])
        coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
        resid = yy - X @ coef
        dof = max(len(ks) - X.shape[1], 1)
        cov = (resid @ resid / dof) * np.linalg.pinv(X.T @ X)
        se = float(np.sqrt(max(cov[2, 2], 1e-12)))
        out[f"{name}_beta"] = float(coef[2])
        out[f"{name}_t"] = float(coef[2] / se)
    return out

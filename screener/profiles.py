"""Industry profiles: the judgments numbers alone can't make, one model call per GICS sub-industry.

For each sub-industry the model gets its members and computed aggregates (concentration among index
members, margins, growth, working-capital days) and returns, in Traditional Chinese:
- market structure: 獨占 / 寡占 / 壟斷性競爭 / 接近完全競爭
- bargaining power of a typical company over suppliers and over customers: 強 / 中 / 弱
- addressable market (TAM) size and 5-year growth, and whether the industry is growth or value in nature
- which macro indicators drive it

Profiles are cached in data/screener/industry_profiles.json and reused; an investor's corrections
(label_overrides.json) take precedence and become training data for the LoRA labeler
(screener/training/train_labeler.py). With TAURUS_LABELER_URL set, the local labeler is tried
first and Claude is only called when its answer doesn't validate.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from core import reward
from core.llm import WRITER_MODEL, ask_json, ask_with_web
from screener.universe import DATA_DIR

logger = logging.getLogger("uvicorn.error")

PROFILES_PATH = DATA_DIR / "industry_profiles.json"
OVERRIDES_PATH = DATA_DIR / "label_overrides.json"
PROFILE_EFFORT = os.environ.get("TAURUS_PROFILE_EFFORT", "medium")
EST_USD_PER_INDUSTRY = 0.03  # Sonnet, ~2k tokens in and ~2k out including thinking; replaced by measured cost
EST_USD_PER_TAM_CHECK = 0.15  # Sonnet with up to 3 searches and 1 fetch

STRUCTURES = ["獨占", "寡占", "壟斷性競爭", "接近完全競爭"]
POWER = ["強", "中", "弱"]
TAM_BUCKETS = ["高成長", "中度成長", "低成長", "萎縮"]

_lock = threading.Lock()


class IndustryProfile(BaseModel):
    market_structure: Literal["獨占", "寡占", "壟斷性競爭", "接近完全競爭"]
    market_structure_rationale: str = Field(description="1-2 sentences: who competes, how concentrated globally, entry barriers")
    supplier_power: Literal["強", "中", "弱"] = Field(description="A typical company's bargaining power OVER ITS SUPPLIERS")
    supplier_power_rationale: str
    customer_power: Literal["強", "中", "弱"] = Field(description="A typical company's bargaining power OVER ITS CUSTOMERS (pricing power)")
    customer_power_rationale: str
    tam_usd_bn: Optional[float] = Field(description="Current global addressable market, USD billions; null if not meaningful")
    tam_cagr_5y_pct: float = Field(description="Expected annual growth of the addressable market over the next 5 years, percent")
    tam_rationale: str
    nature: Literal["成長", "均衡", "價值"] = Field(description="Is the industry growth or value in nature")
    nature_rationale: str
    macro_drivers: str = Field(description="Which macro indicators move this industry's revenue and why (rates, PMI/orders, housing, consumer, commodity prices...)")
    moat_sources: List[str] = Field(description="Typical sources of durable advantage, 0-4 short items")
    confidence: Literal["高", "中", "低"]


SYSTEM = """You are an industry analyst classifying one GICS sub-industry for a stock screener used by a
Taiwanese investor. Write every rationale in Traditional Chinese (繁體中文), 1-2 sentences, concrete
(name the main competitors, suppliers or customers), no filler.

Definitions:
- market_structure: 獨占 = one firm controls the market or prices are regulated monopolies (e.g. local utilities);
  寡占 = a few firms hold most of the GLOBAL market and watch each other's pricing;
  壟斷性競爭 = many firms with differentiated products or brands; 接近完全競爭 = many firms, commodity-like products, price takers.
  The concentration figures you get count only S&P 1500 members, so they overstate concentration when
  foreign, private or smaller rivals matter; judge the real market.
- supplier_power / customer_power are the TYPICAL COMPANY'S power over its suppliers / over its customers
  (強 = it sets terms; 弱 = it takes terms). Gross margin level and stability, and receivable (DSO) and
  payable (DPO) days versus other industries, are evidence.
- tam_cagr_5y_pct: your best estimate of nominal annual growth of the addressable market over the next 5 years.
  The member companies' historical revenue growth is given as a reference, not the answer.
- nature: 成長 if most value comes from future expansion, 價值 if from current cash flows and assets."""


def _fmt_pct(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def industry_prompt(ind: Dict[str, Any], companies: Dict[str, Dict[str, Any]]) -> str:
    top = ", ".join(f"{c['name']} ({c['ticker']}, {_fmt_pct(c['revenue_share'])} of members' revenue)"
                    for c in ind["top_companies"])
    others = [t for t in ind["tickers"][6:20]]
    lines = [
        f"Sub-industry: {ind['sub_industry']} (sector: {ind['sector']})",
        f"S&P 1500 members: {ind['n']}. Largest: {top}" + (f". Others: {', '.join(others)}" if others else ""),
        f"Members' combined revenue: ${ind['revenue_total'] / 1e9:,.1f}bn; HHI among members {ind['hhi'] or 0:,.0f}; "
        f"top-4 share {_fmt_pct(ind['cr4'])}",
        f"Median gross margin {_fmt_pct(ind['median_gross_margin'])} (5-year std {_fmt_pct(ind['median_gross_margin_stability'])}), "
        f"operating margin {_fmt_pct(ind['median_operating_margin'])}, ROIC {_fmt_pct(ind['median_roic'])}",
        f"Median receivable days {ind['median_dso'] or 0:.0f}, payable days {ind['median_dpo'] or 0:.0f}",
        f"Revenue growth: median 3-year CAGR {_fmt_pct(ind['median_revenue_cagr_3y'])}, "
        f"members' combined 5-year CAGR {_fmt_pct(ind['revenue_cagr_5y_aggregate'])}",
        f"Median market beta {ind['median_market_beta'] or 0:.2f}; median revenue elasticity to industrial production "
        f"{ind['median_revenue_elasticity'] if ind['median_revenue_elasticity'] is not None else 'n/a'}",
    ]
    return "\n".join(lines)


def tam_bucket(cagr_pct: Optional[float]) -> Optional[str]:
    if cagr_pct is None:
        return None
    return "高成長" if cagr_pct >= 10 else "中度成長" if cagr_pct >= 4 else "低成長" if cagr_pct >= 0 else "萎縮"


def _load(path) -> Dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    tmp.replace(path)


def _local_label(prompt: str) -> Optional[IndustryProfile]:
    """The LoRA labeler's answer, if one is served and it validates."""
    url = os.environ.get("TAURUS_LABELER_URL")
    if not url:
        return None
    try:
        text = reward._post(f"{url.rstrip('/')}/label", {"prompt": f"{SYSTEM}\n\n{prompt}"})["text"]
        return IndustryProfile.model_validate_json(text[text.find("{"): text.rfind("}") + 1])
    except (OSError, KeyError, ValueError, ValidationError) as exc:
        logger.warning("Local labeler failed (%s); asking Claude", exc)
        return None


def classify_industry(ind: Dict[str, Any], companies: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    prompt = industry_prompt(ind, companies)
    profile, source = _local_label(prompt), "lora"
    if profile is None:
        profile, source = ask_json(SYSTEM, prompt, IndustryProfile, model=WRITER_MODEL, effort=PROFILE_EFFORT), "claude"
    out = profile.model_dump()
    out.update(source=source, model=WRITER_MODEL if source == "claude" else "lora", prompt=prompt,
               at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    with _lock:
        data = _load(PROFILES_PATH)
        data[ind["sub_industry"]] = out
        _save(PROFILES_PATH, data)
    return out


def missing(industries: Dict[str, Dict[str, Any]]) -> List[str]:
    have = _load(PROFILES_PATH)
    return [k for k in industries if k not in have]


def classify_all(industries: Dict[str, Dict[str, Any]], companies: Dict[str, Dict[str, Any]],
                 names: Optional[List[str]] = None, workers: int = 4) -> Dict[str, Any]:
    todo = names if names is not None else missing(industries)
    done, failed = 0, []

    def run(name: str) -> None:
        nonlocal done
        try:
            classify_industry(industries[name], companies)
            done += 1
            logger.info("Profiled %s (%d/%d)", name, done, len(todo))
        except Exception as exc:  # one bad industry must not stop the rest
            failed.append(name)
            logger.warning("Profile %s failed: %s", name, exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:  # copy the context so the cost meter sees each call
        list(pool.map(lambda n: contextvars.copy_context().run(run, n), todo))
    return {"profiled": done, "failed": failed}


# ---- the merged view the screener uses ----

def overrides() -> Dict[str, Dict[str, Any]]:
    return _load(OVERRIDES_PATH)


def set_override(sub_industry: str, field: str, value: Any, note: str = "") -> None:
    """An investor's correction; it wins over the model's label from now on and is kept for training."""
    allowed = {"market_structure": STRUCTURES, "supplier_power": POWER, "customer_power": POWER,
               "nature": ["成長", "均衡", "價值"], "tam_bucket": TAM_BUCKETS}
    if field not in allowed or value not in allowed[field]:
        raise ValueError(f"{field} must be one of {allowed.get(field)}")
    with _lock:
        data = _load(OVERRIDES_PATH)
        data.setdefault(sub_industry, {})[field] = {"value": value, "note": note,
                                                     "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        _save(OVERRIDES_PATH, data)


def merged_profiles() -> Dict[str, Dict[str, Any]]:
    """Model profiles with investor overrides applied; `overridden` lists the corrected fields."""
    data, ovr = _load(PROFILES_PATH), overrides()
    out = {}
    for name, p in data.items():
        p = {k: v for k, v in p.items() if k != "prompt"}
        p["tam_bucket"] = tam_bucket(p.get("tam_cagr_5y_pct"))
        if p.get("tam_verified"):
            p["tam_bucket"] = tam_bucket(p["tam_verified"].get("tam_cagr_5y_pct"))
        p["overridden"] = []
        for field, o in ovr.get(name, {}).items():
            p[field] = o["value"]
            p["overridden"].append(field)
        out[name] = p
    return out


# ---- optional: check one industry's TAM on the web ----

class TamCheck(BaseModel):
    tam_usd_bn: Optional[float]
    tam_cagr_5y_pct: float
    summary: str = Field(description="2-3 sentences in Traditional Chinese with the figures and who estimated them")


TAM_SYSTEM = """You check the size and expected growth of one industry's addressable market. Search for
recent (last 2 years) estimates from research firms, industry bodies or company filings. Cite each figure
with its URL in angle brackets, like <https://...>. Prefer the median of credible estimates over the
most optimistic one."""
TAM_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 3},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 1},
]


def verify_tam(sub_industry: str) -> Dict[str, Any]:
    data = _load(PROFILES_PATH)
    if sub_industry not in data:
        raise KeyError(f"No profile for {sub_industry}")
    p = data[sub_industry]
    notes, sources = ask_with_web(
        TAM_SYSTEM,
        f"Industry: {sub_industry}. Current model estimate: TAM ${p.get('tam_usd_bn')}bn growing "
        f"{p.get('tam_cagr_5y_pct')}% a year. Find current estimates and report the market size and 5-year CAGR.",
        max_tokens=16000, model=WRITER_MODEL, effort="low", tools=TAM_TOOLS)
    check = ask_json("Extract the figures from these research notes. Use null when no size was found.",
                     notes, TamCheck, model=WRITER_MODEL, effort="low")
    out = {**check.model_dump(), "notes": notes, "sources": [s for s in sources if s.get("cited")],
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with _lock:
        data = _load(PROFILES_PATH)
        data[sub_industry]["tam_verified"] = out
        _save(PROFILES_PATH, data)
    return out

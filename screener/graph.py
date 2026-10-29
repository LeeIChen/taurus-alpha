"""The interactive screener as a LangGraph state machine.

Each step pauses with interrupt(): it sends the wizard a payload (the question, the options with how
many companies each keeps, the industry table, the macro context) and resumes with the investor's
answer. Answers can go back a step, so every step recomputes its candidates from the choices so far.
Sessions are checkpointed in SQLite (data/screener/sessions.sqlite) and survive a server restart.

overview -> [interpret_goals] -> style -> size -> cycle -> structure -> power -> tam -> quality
         -> edge -> valuation -> timing -> shortlist <-> deep_dive / hedge_check
(edge, valuation and timing are the hedge-fund questions in hedge.py; shortlist also takes ratings and
A/B choices for preferences.py)

Only interpret_goals, deep_dive and hedge_check call Claude; every other step is computed from the dataset.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
import contextvars
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from core.llm import WRITER_MODEL, ask_json, meter_rows
from screener import evidence, hedge, hedge_check, preferences, profiles
from screener.features import CYCLE_LABELS, RATE_LABELS, SIZE_LABELS, STYLE_LABELS, load_dataset
from screener.universe import DATA_DIR

logger = logging.getLogger("uvicorn.error")

SESSIONS_PATH = DATA_DIR / "sessions.sqlite"
SHORTLIST_SIZE = 25
MAX_DEEP_DIVES = 8

# step -> filters it asks about: (field, label, level, options). "company" fields are read from the
# company row; "industry" fields from the sub-industry's profile (model labels plus overrides).
STEPS: Dict[str, Dict[str, Any]] = {
    "style": {
        "title": "成長型 vs 價值型",
        "question": "你想找成長型、價值型，還是兩者之間的公司？",
        "explain": "成長分數 = 3/5 年營收 CAGR、最新一季營收年增率、3 年獲利 CAGR 的全市場百分位；價值分數 = 盈餘殖利率、"
                   "淨值/股價、營收/股價、自由現金流殖利率、股息殖利率的百分位。兩者差距大於 15 個百分點才歸為成長或價值。"
                   "產業層級另附模型判斷的產業屬性。",
        "filters": [("style", "風格", "company", STYLE_LABELS)],
    },
    "size": {
        "title": "公司規模",
        "question": "要大型、中型還是小型公司？",
        "explain": "市值 = 最新股價 × 最新流通股數。大型 ≥ 200 億美元、中型 50–200 億、小型 < 50 億（接近 S&P 500/400/600 的門檻）。",
        "filters": [("size", "規模", "company", SIZE_LABELS)],
    },
    "cycle": {
        "title": "景氣與利率敏感度",
        "question": "對景氣循環和利率，你要敏感還是遲鈍的公司？",
        "explain": "景氣敏感度 = 市場 Beta、對領先指標（工時、初領失業金、建照、資本財訂單、殖利率曲線、消費者信心的合成指數）"
                   "的 Beta、以及營收對工業生產成長的彈性，三者百分位平均後分三等分。利率敏感度 = 控制大盤後，十年期公債殖利率"
                   "每升 1 個百分點的月報酬變化；t 值 ≤ −1.65 為升息受害、≥ 1.65 為升息受惠。",
        "filters": [("cycle", "景氣敏感度", "company", CYCLE_LABELS), ("rate", "利率敏感度", "company", RATE_LABELS)],
    },
    "structure": {
        "title": "產業競爭結構",
        "question": "要選哪種市場結構的產業？",
        "explain": "由模型依全球競爭者、進入障礙判斷，並參考 S&P 1500 成員間的 HHI 與前四大市占（只算美股成員，會高估集中度）。"
                   "你可以直接改正任何產業的標籤，之後都會以你的判斷為準。",
        "filters": [("market_structure", "市場結構", "industry", profiles.STRUCTURES)],
    },
    "power": {
        "title": "對上下游的議價力",
        "question": "要求公司對供應商、對客戶的議價力至少多強？",
        "explain": "產業層級由模型判斷，證據包括毛利率高低與穩定度、應收帳款天數（DSO，越短代表對客戶越強）與應付帳款天數"
                   "（DPO，越長代表對供應商越強）。進入候選名單後可用 10-K 逐家驗證。",
        "filters": [("supplier_power", "對供應商（上游）", "industry", profiles.POWER),
                    ("customer_power", "對客戶（下游）", "industry", profiles.POWER)],
    },
    "tam": {
        "title": "未來市場空間（TAM）",
        "question": "產業的可服務市場（TAM）未來五年要成長多快？",
        "explain": "模型估計的 TAM 五年年複合成長率：高成長 ≥ 10%、中度 4–10%、低 0–4%、萎縮 < 0。並列成員公司過去五年的合計營收成長供"
                   "對照。可對單一產業用網路搜尋查證（約 0.15 美元）。",
        "filters": [("tam_bucket", "TAM 成長", "industry", profiles.TAM_BUCKETS)],
    },
    # ---- the hedge-fund questions (screener/hedge.py), asked after the quality bar ----
    "edge": {
        "title": "誰掌握瓶頸？營收品質如何？",
        "question": "AI 這類浪潮裡，稀缺的環節才有定價權。要求公司處在瓶頸位置、營收有合約保障嗎？",
        "explain": "產業瓶頸地位由模型判斷（客戶是否要排隊等它的產能、設備、許可或併網資格）。定價權趨勢 = 毛利率是否高於自己"
                   "過去 5 年中位數，以及未履約合約（RPO，即已簽約未認列的營收）是否成長得比營收快。營收品質 = RPO 可涵蓋幾年營收、"
                   "過去 5 年營收成長的波動、衰退年數、毛利率穩定度，全市場百分位後分三等分。約 340 家公司有申報 RPO；隨市場電價"
                   "或商品價格浮動的營收（例如德州發電商）多半沒有 RPO，營收品質會偏低。",
        "filters": [("bottleneck", "產業瓶頸地位", "industry", profiles.BOTTLENECKS),
                    ("pricing_trend", "定價權趨勢", "company", hedge.PRICING_LABELS),
                    ("revenue_quality", "營收品質", "company", hedge.QUALITY_LABELS)],
        "company_cols": ["pricing_trend", "gross_margin", "gross_margin_5y_median", "rpo_coverage", "rpo_growth",
                         "revenue_growth_volatility", "revenue_down_years", "revenue_quality"],
    },
    "valuation": {
        "title": "股價裡已經反映了多少？在現在的利率下還划算嗎？",
        "question": "用反向 DCF 看市場已經預期多少成長，再用目前的十年期殖利率檢驗資金成本。",
        "explain": "反向 DCF：找出讓「未來 10 年現金流折現 + 之後每年 2.5% 永續成長」剛好等於目前企業價值（市值 + 負債 − 現金）"
                   "的 10 年年成長率，再和公司過去 3、5 年營收 CAGR 的平均比較：低 3 個百分點以上 = 預期偏低；±3 = 大致反映；"
                   "高 3–10 = 預期偏高；高 10 以上 = 極度樂觀。資本支出超過折舊 1.5 倍的公司（管線、電廠、資料中心）改用「營業現金流 − "
                   "折舊」為基礎，因為報表上的自由現金流已扣掉了帶來成長的投資。銀行與保險改用股價淨值比反推隱含 ROE。"
                   "折現率 WACC = 十年期殖利率 + 調整後 Beta × 5% 風險溢酬（股權），債務成本由利息支出 / 負債估計。"
                   "資金成本檢驗：ROIC（金融業用 ROE）比 WACC 高 2 個百分點以上、利息保障倍數 ≥ 4、沒有借錢回購配息 = 通過；"
                   "低於 WACC 或利息保障倍數 < 2 = 不通過。",
        "filters": [("priced_in", "市場隱含預期", "company", hedge.PRICED_IN_LABELS),
                    ("rate_test", "資金成本檢驗", "company", hedge.RATE_TEST_LABELS)],
        "company_cols": ["dcf_base", "implied_growth", "delivered_growth", "priced_in", "wacc", "roic", "return_spread",
                         "interest_coverage", "levered_payout", "rate_test"],
    },
    "timing": {
        "title": "催化劑與擁擠度",
        "question": "熱門股遇到壞消息跌最兇。要避開擁擠的交易、找基本面正在加速的公司嗎？",
        "explain": "擁擠度 = 12 個月股價動能、相對同類股的估值溢價（EV/EBITDA、P/E）、反向 DCF 的隱含預期落差，三者百分位"
                   "平均後分三等分（缺放空比率與持股集中度，這兩項要用「避險基金查核」上網查）。基本面動能 = 最新一季營收年增率"
                   "比 3 年 CAGR 快或慢 3 個百分點以上。另列離 12 個月高點的跌幅。財報日、合約、監管等具體催化劑與分析師預估的"
                   "上修／下修，請在候選名單用「避險基金查核」逐家查。",
        "filters": [("crowding", "擁擠度", "company", hedge.CROWDING_LABELS),
                    ("fund_momentum", "基本面動能", "company", hedge.MOMENTUM_LABELS)],
        "company_cols": ["crowding", "momentum_12m", "drawdown_12m", "valuation_premium_pct", "implied_gap",
                         "revenue_growth_latest_q", "revenue_cagr_3y", "fund_momentum"],
    },
}
FILTER_STEPS = list(STEPS)
ORDER = ["overview", "style", "size", "cycle", "structure", "power", "tam", "quality", "edge", "valuation", "timing",
         "shortlist"]
COMPANY_PREVIEW = 60

QUALITY_DEFAULTS = {"min_quality": 0.6, "min_piotroski": 5, "max_net_debt_to_ebitda": 3.0, "positive_fcf": True}


class ScreenState(TypedDict, total=False):
    goals: str
    suggestions: Dict[str, Any]
    choices: Dict[str, Any]  # step -> {"select": {field: [values]}, "level": {field: "company"|"industry"}}
    quality: Dict[str, Any]
    deep_dives: Dict[str, Dict[str, Any]]
    pending_deep_dive: List[str]
    hedge_checks: Dict[str, Dict[str, Any]]
    pending_hedge_check: List[str]
    cost_usd: float
    finished: bool


# ---- candidates ----

def _value(row: Dict[str, Any], field: str, level: str, profs: Dict[str, Dict[str, Any]],
           industries: Dict[str, Dict[str, Any]]) -> Optional[str]:
    if level == "industry":
        prof = profs.get(row["sub_industry"], {})
        if field == "style":  # the industry's nature: model judgment, else the members' median scores
            return prof.get("nature") or industries.get(row["sub_industry"], {}).get("style")
        if field in ("cycle", "rate"):
            return industries.get(row["sub_industry"], {}).get(field)
        return prof.get(field)
    return row.get(field)


def candidates(state: ScreenState, upto: str) -> List[Dict[str, Any]]:
    """Companies that pass every step before `upto` in ORDER (filters and the quality bar)."""
    data = load_dataset()
    profs = profiles.merged_profiles()
    rows = data["companies"]
    choices = state.get("choices") or {}
    for step in ORDER[1: ORDER.index(upto)] if upto in ORDER else ORDER[1:]:
        if step == "quality":
            q = {**QUALITY_DEFAULTS, **(state.get("quality") or {})}
            rows = [r for r in rows if passes_quality(r, q)]
            continue
        if step not in STEPS:
            continue
        c = choices.get(step) or {}
        for field, _, default_level, _ in STEPS[step]["filters"]:
            wanted = (c.get("select") or {}).get(field) or []
            if not wanted:
                continue
            level = (c.get("level") or {}).get(field, default_level)
            rows = [r for r in rows if _value(r, field, level, profs, data["industries"]) in wanted]
    return rows


def passes_quality(r: Dict[str, Any], q: Dict[str, Any]) -> bool:
    if q.get("min_quality") is not None and (r.get("quality_score") is None or r["quality_score"] < q["min_quality"]):
        return False
    if q.get("min_piotroski") is not None and r.get("piotroski") is not None and r["piotroski"] < q["min_piotroski"]:
        return False
    nd = r.get("net_debt_to_ebitda")
    if q.get("max_net_debt_to_ebitda") is not None and nd is not None and nd > q["max_net_debt_to_ebitda"]:
        return False
    if q.get("positive_fcf") and r.get("fcf") is not None and r["fcf"] <= 0:
        return False
    return True


# ---- payloads ----

INDUSTRY_FIELDS = ["sub_industry", "sector", "n", "style", "cycle", "rate", "hhi", "cr4", "median_gross_margin",
                   "median_operating_margin", "median_roic", "median_revenue_cagr_3y", "revenue_cagr_5y_aggregate",
                   "median_dso", "median_dpo", "median_market_beta", "median_rate_t", "median_implied_growth",
                   "median_return_spread", "median_rpo_coverage"]
PROFILE_FIELDS = ["market_structure", "market_structure_rationale", "supplier_power", "supplier_power_rationale",
                  "customer_power", "customer_power_rationale", "tam_usd_bn", "tam_cagr_5y_pct", "tam_bucket",
                  "tam_rationale", "nature", "nature_rationale", "macro_drivers", "moat_sources", "confidence",
                  "overridden", "tam_verified", "bottleneck", "bottleneck_rationale", "ai_role"]
COMPANY_FIELDS = ["ticker", "name", "sector", "sub_industry", "index", "price", "market_cap", "size", "style", "cycle",
                  "rate", "growth_score", "value_score", "cycle_score", "quality_score", "market_beta", "rate_beta",
                  "rate_t", "cycle_beta", "revenue_elasticity", "revenue", "revenue_cagr_3y", "revenue_growth_latest_q",
                  "gross_margin", "operating_margin", "fcf_margin", "roic", "roe", "piotroski", "net_debt_to_ebitda",
                  "interest_coverage", "pe", "pb", "ev_ebitda", "fcf_yield", "dividend_yield", "dso", "dpo",
                  "pricing_power_pct", "momentum_12m", "fiscal_year",
                  # hedge-fund questions
                  "pricing_trend", "gross_margin_5y_median", "rpo_coverage", "rpo_growth", "revenue_growth_volatility",
                  "revenue_down_years", "revenue_quality", "revenue_quality_score", "dcf_base", "implied_growth",
                  "delivered_growth", "implied_roe", "implied_gap", "priced_in", "wacc", "cost_of_equity", "return_spread",
                  "levered_payout", "rate_test", "crowding", "crowding_score", "drawdown_12m", "valuation_premium_pct",
                  "fund_momentum", "revenue_acceleration"]


def industry_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    data = load_dataset()
    profs = profiles.merged_profiles()
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["sub_industry"]] = counts.get(r["sub_industry"], 0) + 1
    out = []
    for name, ind in data["industries"].items():
        row = {k: ind.get(k) for k in INDUSTRY_FIELDS}
        row.update({k: profs.get(name, {}).get(k) for k in PROFILE_FIELDS})
        row["candidates"] = counts.get(name, 0)
        out.append(row)
    return sorted(out, key=lambda r: (-r["candidates"], r["sector"], r["sub_industry"]))


def _options(rows: List[Dict[str, Any]], field: str, level: str, labels: List[str]) -> List[Dict[str, Any]]:
    data = load_dataset()
    profs = profiles.merged_profiles()
    out = []
    for label in labels:
        hit = [r for r in rows if _value(r, field, level, profs, data["industries"]) == label]
        out.append({"value": label, "companies": len(hit), "industries": len({r["sub_industry"] for r in hit})})
    unlabeled = sum(_value(r, field, level, profs, data["industries"]) is None for r in rows)
    if unlabeled:
        out.append({"value": None, "companies": unlabeled, "industries": 0, "note": "缺資料（只在不限時保留）"})
    return out


def filter_payload(step: str, state: ScreenState) -> Dict[str, Any]:
    spec = STEPS[step]
    rows = candidates(state, step)
    prev = (state.get("choices") or {}).get(step) or {}
    sugg = state.get("suggestions") or {}
    dims = []
    for field, label, level, opts in spec["filters"]:
        level = (prev.get("level") or {}).get(field, level)
        dims.append({"field": field, "label": label, "level": level,
                     "can_switch_level": field in ("style", "cycle", "rate"),
                     "options": _options(rows, field, level, opts),
                     "selected": (prev.get("select") or {}).get(field, []),
                     "suggested": sugg.get(field) or []})
    payload = {"step": step, "position": ORDER.index(step), "steps": ORDER, "title": spec["title"],
               "question": spec["question"], "explain": spec["explain"], "remaining": len(rows),
               "dimensions": dims, "industries": industry_rows(rows), "suggestion_note": sugg.get("summary")}
    if step in ("cycle", "valuation"):
        payload["regime"] = load_dataset()["regime"]
    if spec.get("company_cols"):  # company-level questions: show the companies themselves, best quality first
        ranked = sorted(rows, key=lambda r: r.get("quality_score") or 0, reverse=True)[:COMPANY_PREVIEW]
        payload["company_cols"] = spec["company_cols"]
        payload["companies"] = [{**{k: r.get(k) for k in ("ticker", "name", "sub_industry", "quality_score")},
                                 **{k: r.get(k) for k in spec["company_cols"]}} for r in ranked]
    return payload


def company_view(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: r.get(k) for k in COMPANY_FIELDS}


# ---- nodes ----

def _route(answer: Dict[str, Any], step: str) -> str:
    i = ORDER.index(step)
    action = answer.get("action", "next")
    if action == "back":
        return ORDER[max(i - 1, 0)]
    if action == "restart":
        return "overview"
    if action == "goto" and answer.get("step") in ORDER:
        return answer["step"]
    return ORDER[i + 1] if i + 1 < len(ORDER) else END


def _apply_overrides(answer: Dict[str, Any], session: str) -> None:
    for o in answer.get("overrides") or []:
        profiles.set_override(o["sub_industry"], o["field"], o["value"], o.get("note", ""))
        preferences.record({"type": "label", "session": session, **o})


def _session(config) -> str:
    return (config or {}).get("configurable", {}).get("thread_id", "")


def overview(state: ScreenState, config) -> Command:
    data = load_dataset()
    payload = {"step": "overview", "position": 0, "steps": ORDER, "title": "產業地圖與總經環境",
               "question": "先看各產業的分類。想找什麼樣的公司？可用一句話描述（選填），我會預先勾好建議選項。",
               "regime": data["regime"], "remaining": len(data["companies"]), "built_at": data["built_at"],
               "industries": industry_rows(data["companies"]), "goals": state.get("goals", ""),
               "missing_profiles": len(profiles.missing(data["industries"]))}
    answer = interrupt(payload)
    _apply_overrides(answer, _session(config))
    goals = (answer.get("goals") or "").strip()
    if answer.get("action") in ("goto", "restart"):
        return Command(goto=_route(answer, "overview"))
    if goals and goals != state.get("goals"):
        return Command(goto="interpret_goals", update={"goals": goals})
    return Command(goto="style", update={"goals": goals})


class Suggestions(BaseModel):
    style: List[str] = Field(default_factory=list, description=f"subset of {STYLE_LABELS}; empty = no preference")
    size: List[str] = Field(default_factory=list, description=f"subset of {SIZE_LABELS}")
    cycle: List[str] = Field(default_factory=list, description=f"subset of {CYCLE_LABELS} (sensitivity to the business cycle)")
    rate: List[str] = Field(default_factory=list, description=f"subset of {RATE_LABELS}")
    market_structure: List[str] = Field(default_factory=list, description=f"subset of {profiles.STRUCTURES}")
    supplier_power: List[str] = Field(default_factory=list, description=f"subset of {profiles.POWER}")
    customer_power: List[str] = Field(default_factory=list, description=f"subset of {profiles.POWER}")
    tam_bucket: List[str] = Field(default_factory=list, description=f"subset of {profiles.TAM_BUCKETS}")
    bottleneck: List[str] = Field(default_factory=list, description=f"subset of {profiles.BOTTLENECKS} (the industry controls a scarce input)")
    pricing_trend: List[str] = Field(default_factory=list, description=f"subset of {hedge.PRICING_LABELS} (gross margin and backlog trend)")
    revenue_quality: List[str] = Field(default_factory=list, description=f"subset of {hedge.QUALITY_LABELS} (contracted, steady revenue)")
    priced_in: List[str] = Field(default_factory=list, description=f"subset of {hedge.PRICED_IN_LABELS} (reverse-DCF implied growth vs delivered)")
    rate_test: List[str] = Field(default_factory=list, description=f"subset of {hedge.RATE_TEST_LABELS} (ROIC above WACC at today's rates)")
    crowding: List[str] = Field(default_factory=list, description=f"subset of {hedge.CROWDING_LABELS} (momentum + valuation premium)")
    fund_momentum: List[str] = Field(default_factory=list, description=f"subset of {hedge.MOMENTUM_LABELS} (revenue growth speeding up)")
    summary: str = Field(description="One or two sentences in Traditional Chinese explaining the suggestions")


GOALS_SYSTEM = """Translate an investor's description of what they want into screener selections. Only
select values the description implies; leave a list empty when it says nothing about that dimension.
Take the current macro regime into account only if the investor asks for it ("順勢", "抗跌")."""


def interpret_goals(state: ScreenState) -> Dict[str, Any]:
    before = len(meter_rows())
    regime = load_dataset()["regime"]
    s = ask_json(GOALS_SYSTEM, f"Investor: {state['goals']}\nMacro now: cycle {regime['phase']}, long rates {regime['rates']}.",
                 Suggestions, model=WRITER_MODEL, effort="low")
    cost = sum(r["usd"] for r in meter_rows()[before:])
    return {"suggestions": s.model_dump(), "cost_usd": (state.get("cost_usd") or 0) + cost}


def make_filter_node(step: str):
    def node(state: ScreenState, config) -> Command:
        answer = interrupt(filter_payload(step, state))
        _apply_overrides(answer, _session(config))
        choices = dict(state.get("choices") or {})
        if answer.get("action", "next") == "next" or "select" in answer:
            choices[step] = {"select": answer.get("select") or {}, "level": answer.get("level") or {}}
        return Command(goto=_route(answer, step), update={"choices": choices})
    node.__name__ = step
    return node


def quality(state: ScreenState, config) -> Command:
    rows = candidates(state, "quality")
    q = {**QUALITY_DEFAULTS, **(state.get("quality") or {})}
    presets = {name: sum(passes_quality(r, {**q, "min_quality": v}) for r in rows)
               for name, v in (("寬鬆 (≥40%)", 0.4), ("標準 (≥60%)", 0.6), ("嚴格 (≥75%)", 0.75))}
    payload = {"step": "quality", "position": ORDER.index("quality"), "steps": ORDER, "title": "基本面強健度",
               "question": "設定基本面門檻：只留下財務體質好的公司。",
               "explain": "品質分數 = 同產業類股（GICS sector）內 ROIC、ROE、毛利率、營業利益率、自由現金流率、Piotroski F 分數、"
                          "淨負債/EBITDA（越低越好）、利息保障倍數、毛利率穩定度的百分位平均。F 分數 0–9 衡量獲利、槓桿與效率是否改善。",
               "remaining": len(rows), "current": q, "passing": sum(passes_quality(r, q) for r in rows), "presets": presets,
               "industries": industry_rows(rows)}
    answer = interrupt(payload)
    update = {}
    if answer.get("quality") is not None:
        update["quality"] = {**q, **answer["quality"]}
    return Command(goto=_route(answer, "quality"), update=update)


def _pair_suggestions(ranked: List[Dict[str, Any]], session_pairs: set) -> List[List[str]]:
    """Two A/B questions between neighbours in the ranking that the investor hasn't compared yet."""
    out = []
    for a, b in zip(ranked, ranked[1:]):
        key = tuple(sorted((a["ticker"], b["ticker"])))
        if key not in session_pairs and a["sub_industry"] != b["sub_industry"]:
            out.append([a["ticker"], b["ticker"]])
        if len(out) == 2:
            break
    return out


def shortlist(state: ScreenState, config) -> Command:
    session = _session(config)
    data = load_dataset()
    profs = profiles.merged_profiles()
    q = {**QUALITY_DEFAULTS, **(state.get("quality") or {})}
    rows = [dict(r) for r in candidates(state, "shortlist")]
    universe_feats = [preferences.raw_features(r, profs.get(r["sub_industry"])) for r in data["companies"]]
    prompt = f"Investor goals: {state.get('goals') or '(none)'}; filters: {state.get('choices')}"
    ranked, info = preferences.rank(rows, profs, universe_feats, prompt)
    top = ranked[:SHORTLIST_SIZE]
    fb = [f for f in preferences.load_feedback() if f.get("session") == session]
    ratings = {f["chosen"]["ticker"]: f["rating"] for f in fb if f["type"] == "rating"}
    compared = {tuple(sorted((f["chosen"]["ticker"], f["rejected"]["ticker"]))) for f in fb if f["type"] == "pair"}
    dives = state.get("deep_dives") or {}
    checks = state.get("hedge_checks") or {}
    companies = []
    for r in top:
        v = company_view(r)
        v.update(score=r["score"], learned_score=r.get("learned_score"), reward_score=r.get("reward_score"),
                 rating=ratings.get(r["ticker"]), deep_dive=dives.get(r["ticker"]), hedge_check=checks.get(r["ticker"]),
                 profile={k: profs.get(r["sub_industry"], {}).get(k) for k in
                          ("market_structure", "supplier_power", "customer_power", "tam_bucket", "tam_cagr_5y_pct",
                           "bottleneck", "ai_role")})
        companies.append(v)
    payload = {"step": "shortlist", "position": ORDER.index("shortlist"), "steps": ORDER, "title": "候選名單",
               "question": "這些公司通過了你所有的條件，依基本面品質與你的偏好排序。按 👍/👎 或回答下方的二選一，排序會學習你的偏好。",
               "remaining": len(rows), "companies": companies, "ranking": info,
               "pairs": _pair_suggestions(top, compared), "choices": state.get("choices"), "quality": q,
               "deep_dive_estimate_usd": evidence.EST_USD_PER_COMPANY,
               "hedge_check_estimate_usd": hedge_check.EST_USD_PER_COMPANY, "cost_usd": state.get("cost_usd") or 0}
    answer = interrupt(payload)
    action = answer.get("action", "stay")
    by_ticker = {r["ticker"]: r for r in rows}

    def fb_row(t: str) -> Dict[str, Any]:
        r = by_ticker[t]
        p = profs.get(r["sub_industry"])
        return {"ticker": t, "features": preferences.raw_features(r, p), "card": preferences.card(r, p)}

    if action == "rate" and answer.get("ticker") in by_ticker:
        preferences.record({"type": "rating", "session": session, "prompt": prompt, "rating": int(answer["rating"]),
                            "chosen": fb_row(answer["ticker"])})
        return Command(goto="shortlist")
    if action == "compare" and answer.get("chosen") in by_ticker and answer.get("rejected") in by_ticker:
        preferences.record({"type": "pair", "session": session, "prompt": prompt,
                            "chosen": fb_row(answer["chosen"]), "rejected": fb_row(answer["rejected"])})
        return Command(goto="shortlist")
    if action == "deep_dive":
        tickers = [t for t in answer.get("tickers") or [] if t in by_ticker][:MAX_DEEP_DIVES]
        return Command(goto="deep_dive", update={"pending_deep_dive": tickers})
    if action == "hedge_check":
        tickers = [t for t in answer.get("tickers") or [] if t in by_ticker][:MAX_DEEP_DIVES]
        return Command(goto="hedge_check", update={"pending_hedge_check": tickers})
    if action == "finish":
        return Command(goto=END, update={"finished": True})
    if action in ("back", "goto", "restart"):
        return Command(goto=_route(answer, "shortlist"))
    return Command(goto="shortlist")


def _run_checks(state: ScreenState, pending: str, done: str, check) -> Dict[str, Any]:
    """Run `check(row, profile)` for each pending ticker in parallel and add the cost to the session."""
    data = load_dataset()
    profs = profiles.merged_profiles()
    by_ticker = {r["ticker"]: r for r in data["companies"]}
    before = len(meter_rows())
    results = dict(state.get(done) or {})

    def one(t: str):
        try:
            r = by_ticker[t]
            return t, {k: v for k, v in check(r, profs.get(r["sub_industry"])).items() if k != "prompt"}
        except Exception as exc:  # a missing 10-K must not lose the others
            logger.warning("%s %s failed: %s", done, t, exc)
            return t, {"error": f"{type(exc).__name__}: {exc}"}

    tickers = state.get(pending) or []
    with ThreadPoolExecutor(max_workers=4) as pool:  # contexts copied in this thread, so the meter sees each call
        for f in [pool.submit(contextvars.copy_context().run, one, t) for t in tickers]:
            t, res = f.result()
            results[t] = res
    cost = sum(r["usd"] for r in meter_rows()[before:])
    return {done: results, pending: [], "cost_usd": (state.get("cost_usd") or 0) + cost}


def deep_dive(state: ScreenState) -> Dict[str, Any]:
    return _run_checks(state, "pending_deep_dive", "deep_dives", lambda r, p: evidence.deep_dive(r, p))


def run_hedge_check(state: ScreenState) -> Dict[str, Any]:
    regime = load_dataset()["regime"]
    return _run_checks(state, "pending_hedge_check", "hedge_checks", lambda r, p: hedge_check.check(r, p, regime))


def build_graph(checkpointer=None):
    g = StateGraph(ScreenState)
    g.add_node("overview", overview, destinations=("interpret_goals", "style", *ORDER))
    g.add_node("interpret_goals", interpret_goals)
    for step in FILTER_STEPS:
        g.add_node(step, make_filter_node(step), destinations=tuple(ORDER))
    g.add_node("quality", quality, destinations=tuple(ORDER))
    g.add_node("shortlist", shortlist, destinations=("deep_dive", "hedge_check", END, *ORDER))
    g.add_node("deep_dive", deep_dive)
    g.add_node("hedge_check", run_hedge_check)
    g.add_edge(START, "overview")
    g.add_edge("interpret_goals", "style")
    g.add_edge("deep_dive", "shortlist")
    g.add_edge("hedge_check", "shortlist")
    return g.compile(checkpointer=checkpointer)


_graph = None
_graph_lock = threading.Lock()


def get_graph():
    """The compiled graph with a SQLite checkpointer (one connection shared across request threads)."""
    global _graph
    with _graph_lock:
        if _graph is None:
            from langgraph.checkpoint.sqlite import SqliteSaver
            SESSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(SESSIONS_PATH), check_same_thread=False)
            _graph = build_graph(SqliteSaver(conn))
        return _graph

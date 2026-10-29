"""Screener tests on a synthetic dataset: no network, no Claude calls.

  PYTHONPATH=. pytest -q tests
"""

from __future__ import annotations

import json
import random

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from screener import evidence, features, fundamentals, graph, hedge, hedge_check, preferences, profiles
from screener.universe import parse_constituents

SUBS = {"Semiconductors": "Information Technology", "Regional Banks": "Financials", "Restaurants": "Consumer Discretionary"}
PROFILES = {
    "Semiconductors": {"market_structure": "寡占", "supplier_power": "中", "customer_power": "強", "tam_cagr_5y_pct": 12.0, "nature": "成長", "bottleneck": "關鍵瓶頸"},
    "Regional Banks": {"market_structure": "接近完全競爭", "supplier_power": "中", "customer_power": "弱", "tam_cagr_5y_pct": 3.0, "nature": "價值", "bottleneck": "非瓶頸"},
    "Restaurants": {"market_structure": "壟斷性競爭", "supplier_power": "中", "customer_power": "中", "tam_cagr_5y_pct": 5.0, "nature": "均衡", "bottleneck": "非瓶頸"},
}


def _company(rng: random.Random, i: int, sub: str) -> dict:
    rev = rng.uniform(1e9, 80e9)
    return {
        "ticker": f"T{i:02d}", "name": f"Company {i}", "cik": 1000 + i, "sector": SUBS[sub], "sub_industry": sub,
        "index": "S&P 500", "revenue": rev, "revenue_ttm": rev, "net_income": rev * rng.uniform(-0.05, 0.3),
        "net_income_ttm": rev * rng.uniform(0.01, 0.3), "gross_margin": rng.uniform(0.2, 0.7), "operating_margin": rng.uniform(0, 0.4),
        "roic": rng.uniform(-0.05, 0.4), "roe": rng.uniform(0, 0.3), "roa": rng.uniform(0, 0.1), "equity_to_assets": rng.uniform(0.05, 0.6),
        "fcf": rev * rng.uniform(-0.05, 0.25), "fcf_margin": rng.uniform(-0.05, 0.25), "piotroski": rng.randint(2, 9),
        "net_debt_to_ebitda": rng.uniform(-1, 5), "interest_coverage": rng.uniform(1, 30), "gross_margin_stability": rng.uniform(0, 0.1),
        "net_income_stability": rng.uniform(0, 1), "revenue_cagr_3y": rng.uniform(-0.1, 0.4), "revenue_cagr_5y": rng.uniform(-0.1, 0.3),
        "revenue_growth_latest_q": rng.uniform(-0.1, 0.4), "net_income_cagr_3y": rng.uniform(-0.1, 0.4), "dso": rng.uniform(10, 90),
        "dpo": rng.uniform(10, 90), "market_beta": rng.uniform(0.3, 1.8), "cycle_beta": rng.uniform(-0.02, 0.03),
        "revenue_elasticity": rng.uniform(-1, 3), "rate_beta": rng.uniform(-0.08, 0.06), "rate_t": rng.uniform(-3, 3),
        "momentum_12m": rng.uniform(-0.3, 0.6), "price": rng.uniform(10, 500), "shares": rng.uniform(5e7, 3e9),
        "latest_debt": rng.uniform(0, 2e10), "latest_cash": rng.uniform(0, 1e10), "latest_equity": rng.uniform(1e9, 5e10),
        "dividends": rng.uniform(0, 2e9), "ebitda": rev * rng.uniform(0.05, 0.4),
        "revenue_history": {str(y): rev * (0.9 ** (2025 - y)) for y in range(2019, 2026)},
    }


@pytest.fixture()
def dataset(tmp_path, monkeypatch):
    rng = random.Random(7)
    rows = [_company(rng, i, sub) for i, sub in enumerate(list(SUBS) * 8)]
    for r in rows:
        features.add_valuation(r)
    features.label_companies(rows)
    hedge.add_metrics(rows, rf=0.05)
    data = {"built_at": "2026-10-28T00:00:00+00:00", "companies": rows, "industries": features.industry_table(rows),
            "regime": {"as_of": "2026-09", "phase": "擴張", "phase_hint": "x", "leading_index": 0.2, "leading_index_3m_ago": 0.1,
                       "ten_year": 5.0, "ten_year_6m_ago": 4.3, "rates": "上升", "rate_hint": "y", "fed_funds": 3.75,
                       "fed_funds_6m_ago": 3.64, "policy": "按兵不動"}}
    monkeypatch.setattr(features, "DATASET_PATH", tmp_path / "dataset.json")
    (tmp_path / "dataset.json").write_text(json.dumps(data))
    monkeypatch.setattr(profiles, "PROFILES_PATH", tmp_path / "profiles.json")
    monkeypatch.setattr(profiles, "OVERRIDES_PATH", tmp_path / "overrides.json")
    monkeypatch.setattr(preferences, "FEEDBACK_PATH", tmp_path / "feedback.jsonl")
    monkeypatch.setattr(evidence, "DEEP_DIR", tmp_path / "deep")
    monkeypatch.setattr(hedge_check, "CHECK_DIR", tmp_path / "hedge")
    (tmp_path / "profiles.json").write_text(json.dumps(
        {k: {**v, "source": "claude", "prompt": f"Sub-industry: {k}"} for k, v in PROFILES.items()}, ensure_ascii=False))
    features._cache.clear()
    return data


# ---- pure functions ----

def test_parse_constituents_handles_each_table_style():
    wikitext = '''{| class="wikitable" id="constituents"
|-
! Symbol !! Security !! Sector !! Sub-Industry
|-
|| {{NyseSymbol|BRK.B}} <!-- DO NOT CHANGE. YOU'VE BEEN WARNED! -->
|| [[Berkshire Hathaway]]
|| Financials
|| Multi-Sector Holdings
|| [[Omaha]]
|| 0001067983
|-
|| {{BZX link|CBOE}}
|| [[Cboe Global Markets]]
|| Financials
|| Financial Exchanges & Data
|-
| style="border-color:inherit;" | {{NasdaqSymbol|AAON}}
| style="border-color:inherit;" | [[AAON]]
| Industrials
| Building Products
| [https://www.sec.gov/cgi-bin/browse-edgar?CIK=824142&action=getcompany reports]
|-
|[{{#invoke:Stock tickers/NYSE|GetURL|PRK|exchange=AMEX}} PRK]
|[[Park National|Park National Corp.]]
|Financials
|Regional Banks
|}'''
    rows = {r["ticker"]: r for r in parse_constituents(wikitext)}
    assert set(rows) == {"BRK-B", "CBOE", "AAON", "PRK"}
    assert rows["BRK-B"]["cik"] == 1067983 and rows["AAON"]["cik"] == 824142
    assert rows["PRK"]["name"] == "Park National Corp." and rows["AAON"]["sub_industry"] == "Building Products"


def test_piotroski_scores_improvement_and_needs_six_tests():
    prev = {"net_income": 5, "assets": 100, "cfo": 6, "long_term_debt": 30, "current_assets": 40, "current_liabilities": 30,
            "diluted_shares": 10, "revenue": 50, "gross_profit": 20}
    better = {"net_income": 8, "assets": 100, "cfo": 10, "long_term_debt": 20, "current_assets": 45, "current_liabilities": 30,
              "diluted_shares": 10, "revenue": 60, "gross_profit": 27}
    assert fundamentals.piotroski(better, prev) == 9
    assert fundamentals.piotroski({"net_income": 1}, {"assets": 10}) is None


def test_revenue_elasticity_recovers_slope():
    driver = {y: 100 * (1.02 ** (y - 2014)) * (1.05 if y % 2 else 0.97) for y in range(2014, 2026)}
    series = {y: 0.0 for y in driver}
    series[2014] = 1000.0
    for y in range(2015, 2026):
        series[y] = series[y - 1] * (1 + 2.0 * (driver[y] / driver[y - 1] - 1))
    assert fundamentals.elasticity(series, driver) == pytest.approx(2.0, abs=0.05)


def test_percentiles_share_ties():
    p = features.percentiles({"a": 1, "b": 2, "c": 2, "d": 3})
    assert p["a"] == 0 and p["d"] == 1 and p["b"] == p["c"] == 0.5


def test_bradley_terry_learns_the_investors_direction():
    rng = random.Random(1)
    rows = [{"roic": rng.uniform(0, 0.4), "gross_margin": rng.uniform(0.2, 0.6)} for _ in range(60)]
    feats = [{k: r.get(k) for k in preferences.FEATURES} for r in rows]
    std = preferences.Standardizer(feats)
    pairs = []
    for _ in range(40):
        a, b = rng.sample(feats, 2)
        hi, lo = (a, b) if a["roic"] > b["roic"] else (b, a)
        pairs.append({"chosen": {"features": hi}, "rejected": {"features": lo}})
    w = preferences.fit(pairs, std)
    names = list(preferences.FEATURES)
    assert w[names.index("roic")] > 0.5 and abs(w[names.index("gross_margin")]) < w[names.index("roic")]
    assert preferences.explain(w)[0]["feature"] == "ROIC"


def test_reverse_dcf_recovers_the_growth_it_was_priced_with():
    ev = hedge.dcf_value(100.0, 0.08, 0.09)
    assert hedge.implied_growth(ev, 100.0, 0.09) == pytest.approx(0.08, abs=1e-6)
    assert hedge.implied_growth(ev, -5.0, 0.09) is None
    row = {"sector": "Industrials", "ev": ev, "fcf": 100.0, "market_cap": ev, "market_beta": 1.0,
           "revenue_cagr_3y": 0.02, "revenue_cagr_5y": 0.04}
    out = hedge.reverse_dcf(row, rf=0.04)
    assert out["wacc"] == pytest.approx(0.09)  # 4% + 1.0 x 5%, no debt
    assert out["implied_growth"] == pytest.approx(0.08, abs=1e-6) and out["priced_in"] == "預期偏高"  # 8% vs 3% delivered


def test_growth_capex_uses_maintenance_cash_flow_and_banks_use_book():
    builder = {"fcf": 10.0, "cfo": 100.0, "capex": 90.0, "dna": 40.0}
    assert hedge.base_cash_flow(builder) == (60.0, "維持性現金流")
    assert hedge.base_cash_flow({"fcf": 10.0, "cfo": 50.0, "capex": 40.0, "dna": 35.0}) == (10.0, "自由現金流")
    bank = {"sector": "Financials", "pb": 2.0, "roe": 0.12, "market_beta": 1.0, "market_cap": 1e10}
    out = hedge.reverse_dcf(bank, rf=0.05)
    assert out["implied_roe"] == pytest.approx(0.03 + 2.0 * (0.10 - 0.03))  # 17% implied vs 12% earned
    assert out["priced_in"] == "預期偏高"


def test_rate_test_needs_returns_above_the_cost_of_capital():
    coc = {"wacc": 0.09, "cost_of_equity": 0.10}
    ok = {"sector": "Industrials", "roic": 0.15, "interest_coverage": 10, "net_debt_to_ebitda": 1, "fcf": 5, "buybacks": 1}
    assert hedge.rate_test(ok, 0.05, coc)["rate_test"] == "通過"
    levered = {**ok, "net_debt_to_ebitda": 4, "buybacks": 10, "fcf": 5}
    assert hedge.rate_test(levered, 0.05, coc)["levered_payout"] and hedge.rate_test(levered, 0.05, coc)["rate_test"] == "勉強"
    assert hedge.rate_test({**ok, "roic": 0.07}, 0.05, coc)["rate_test"] == "不通過"
    assert hedge.rate_test({**ok, "interest_coverage": 1.5}, 0.05, coc)["rate_test"] == "不通過"


def test_industry_table_concentration():
    rows = [{"ticker": t, "name": t, "sector": "S", "sub_industry": "X", "revenue": r} for t, r in (("A", 60), ("B", 30), ("C", 10))]
    ind = features.industry_table(rows)["X"]
    assert ind["hhi"] == pytest.approx(60 ** 2 + 30 ** 2 + 10 ** 2)
    assert ind["cr4"] == pytest.approx(1.0)


# ---- the interactive graph ----

def _payload(g, cfg):
    return next(t.interrupts[0].value for t in g.get_state(cfg).tasks if t.interrupts)


def test_wizard_flow_filters_back_feedback_and_deep_dive(dataset, monkeypatch):
    calls, checks = [], []
    monkeypatch.setattr(evidence, "deep_dive", lambda row, prof, refresh=False: calls.append(row["ticker"]) or
                        {"ticker": row["ticker"], "moat": "窄", "evidence": []})
    monkeypatch.setattr(hedge_check, "check", lambda row, prof, regime, refresh=False: checks.append(row["ticker"]) or
                        {"ticker": row["ticker"], "estimate_trend": "下修", "catalysts": []})
    g = graph.build_graph(InMemorySaver())
    cfg = {"configurable": {"thread_id": "t1"}}
    g.invoke({"choices": {}, "cost_usd": 0.0}, cfg)
    p = _payload(g, cfg)
    assert p["step"] == "overview" and p["remaining"] == 24
    assert {r["sub_industry"]: r["market_structure"] for r in p["industries"]}["Semiconductors"] == "寡占"

    g.invoke(Command(resume={"action": "next"}), cfg)
    p = _payload(g, cfg)
    assert p["step"] == "style"
    growth = next(o for o in p["dimensions"][0]["options"] if o["value"] == "成長")
    g.invoke(Command(resume={"action": "next", "select": {"style": ["成長"]}}), cfg)
    p = _payload(g, cfg)
    assert p["step"] == "size" and p["remaining"] == growth["companies"]

    g.invoke(Command(resume={"action": "back"}), cfg)
    p = _payload(g, cfg)
    assert p["step"] == "style" and p["dimensions"][0]["selected"] == ["成長"]
    # industry level: the industry's nature instead of each company's style
    g.invoke(Command(resume={"action": "next", "select": {"style": ["成長"]}, "level": {"style": "industry"}}), cfg)
    assert _payload(g, cfg)["remaining"] == 8  # all eight semiconductor companies

    for step in ("size", "cycle", "structure"):
        g.invoke(Command(resume={"action": "next"}), cfg)
    p = _payload(g, cfg)
    assert p["step"] == "power"
    # an investor correction applies at once and is logged as feedback
    g.invoke(Command(resume={"action": "goto", "step": "power", "overrides": [
        {"sub_industry": "Semiconductors", "field": "customer_power", "value": "中", "note": "hyperscalers push back"}]}), cfg)
    p = _payload(g, cfg)
    semis = next(r for r in p["industries"] if r["sub_industry"] == "Semiconductors")
    assert semis["customer_power"] == "中" and "customer_power" in semis["overridden"]
    g.invoke(Command(resume={"action": "next"}), cfg)  # power -> tam
    g.invoke(Command(resume={"action": "next", "select": {"tam_bucket": ["高成長"]}}), cfg)
    p = _payload(g, cfg)
    assert p["step"] == "quality" and p["remaining"] == 8
    g.invoke(Command(resume={"action": "next", "quality": {"min_quality": 0.0, "min_piotroski": 0,
                                                           "max_net_debt_to_ebitda": None, "positive_fcf": False}}), cfg)
    p = _payload(g, cfg)
    assert p["step"] == "edge" and p["remaining"] == 8 and len(p["companies"]) == 8 and "rpo_coverage" in p["company_cols"]
    g.invoke(Command(resume={"action": "next"}), cfg)
    p = _payload(g, cfg)
    assert p["step"] == "valuation" and "regime" in p
    opts = {o["value"]: o["companies"] for o in p["dimensions"][1]["options"]}  # rate_test
    keep = [k for k in ("通過", "勉強") if opts.get(k)]
    g.invoke(Command(resume={"action": "next", "select": {"rate_test": keep}}), cfg)
    p = _payload(g, cfg)
    assert p["step"] == "timing" and p["remaining"] == sum(opts[k] for k in keep)
    g.invoke(Command(resume={"action": "back"}), cfg)
    g.invoke(Command(resume={"action": "next"}), cfg)  # drop the rate filter again
    g.invoke(Command(resume={"action": "next"}), cfg)  # timing -> shortlist
    p = _payload(g, cfg)
    assert p["step"] == "shortlist" and len(p["companies"]) == 8
    a, b = p["companies"][-1]["ticker"], p["companies"][0]["ticker"]
    g.invoke(Command(resume={"action": "compare", "chosen": a, "rejected": b}), cfg)
    g.invoke(Command(resume={"action": "rate", "ticker": a, "rating": 1}), cfg)
    g.invoke(Command(resume={"action": "rate", "ticker": b, "rating": -1}), cfg)
    p = _payload(g, cfg)
    assert p["ranking"]["pairs"] == 2  # one A/B answer + one up-vs-down pair
    assert {f["type"] for f in preferences.load_feedback()} == {"label", "pair", "rating"}

    g.invoke(Command(resume={"action": "deep_dive", "tickers": [a, "NOT-A-CANDIDATE"]}), cfg)
    p = _payload(g, cfg)
    assert calls == [a] and next(c for c in p["companies"] if c["ticker"] == a)["deep_dive"]["moat"] == "窄"
    g.invoke(Command(resume={"action": "hedge_check", "tickers": [b]}), cfg)
    p = _payload(g, cfg)
    assert checks == [b] and next(c for c in p["companies"] if c["ticker"] == b)["hedge_check"]["estimate_trend"] == "下修"
    g.invoke(Command(resume={"action": "finish"}), cfg)
    assert g.get_state(cfg).values["finished"] and not g.get_state(cfg).next


def test_api_requires_confirmation_for_paid_steps(dataset, monkeypatch):
    import main
    from screener import api

    g = graph.build_graph(InMemorySaver())
    monkeypatch.setattr(api, "get_graph", lambda: g)
    c = TestClient(main.app)
    start = c.post("/screen", json={}).json()
    assert start["step"] == "overview"
    sid = start["session_id"]
    assert c.post(f"/screen/{sid}/answer", json={"action": "next"}).json()["step"] == "style"
    assert c.get(f"/screen/{sid}").json()["step"] == "style"
    assert c.post(f"/screen/{sid}/answer", json={"action": "deep_dive", "tickers": ["T01"]}).status_code == 402
    assert c.post(f"/screen/{sid}/answer", json={"action": "hedge_check", "tickers": ["T01"]}).status_code == 402
    assert "estimate_usd" in c.post(f"/screen/{sid}/report", json={"ticker": "T01"}).json()
    assert c.get("/screen/nope").status_code == 404
    assert c.get("/screener").status_code == 200


def test_label_examples_apply_human_corrections(dataset):
    from screener.training.export import label_examples

    profiles.set_override("Restaurants", "market_structure", "寡占", "big chains")
    rows = {r["name"]: r for r in label_examples()}
    assert json.loads(rows["Restaurants"]["completion"])["market_structure"] == "寡占"
    assert rows["Restaurants"]["human_corrected"] == ["market_structure"]
    with pytest.raises(ValueError):
        profiles.set_override("Restaurants", "market_structure", "壟斷")


def test_report_generator_still_builds():
    from report_generator.workflow import build_graph

    assert len(build_graph("planner").get_graph().nodes) == 16

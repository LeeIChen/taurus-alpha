"""Learning the investor's taste from feedback (the RLHF part of the screener).

Feedback on the shortlist is saved to data/screener/feedback.jsonl:
- rating: thumbs up or down on one company;
- pair: "of these two, I prefer A" (the wizard asks a few of these);
- label: a corrected industry label (also saved by profiles.set_override).

Two learners use it:
1. A Bradley-Terry model on standardized company features, fit here with numpy in milliseconds. Its
   weights rerank the shortlist and are shown to the investor ("you favor high ROIC, low leverage...").
2. A LoRA reward model on the company cards (screener/training/train_pref_reward.py, GPU). When served
   (TAURUS_SCREEN_REWARD_URL), its scores are blended in too.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from datetime import datetime, timezone
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core import reward
from screener.universe import DATA_DIR

logger = logging.getLogger("uvicorn.error")

FEEDBACK_PATH = DATA_DIR / "feedback.jsonl"
L2 = 1.0
PRIOR_PAIRS = 10  # the learned score's weight is n / (n + PRIOR_PAIRS)

ORDINALS = {
    "market_structure": {"獨占": 3, "寡占": 2, "壟斷性競爭": 1, "接近完全競爭": 0},
    "supplier_power": {"強": 2, "中": 1, "弱": 0},
    "customer_power": {"強": 2, "中": 1, "弱": 0},
}
# feature -> label shown to the investor
FEATURES = {
    "quality_score": "基本面品質", "growth_score": "成長性", "value_score": "便宜程度",
    "cycle_score": "景氣敏感度", "rate_beta": "升息受惠程度", "market_beta": "市場 Beta",
    "roic": "ROIC", "gross_margin": "毛利率", "fcf_margin": "自由現金流率", "net_debt_to_ebitda": "淨負債/EBITDA",
    "revenue_cagr_3y": "3 年營收成長", "momentum_12m": "12 個月股價動能", "log_market_cap": "市值規模",
    "piotroski": "Piotroski F 分數", "dividend_yield": "股息殖利率", "tam_cagr": "TAM 成長",
    "market_structure": "市場集中度", "customer_power": "對客戶議價力", "supplier_power": "對供應商議價力",
}
CLIP = {"net_debt_to_ebitda": (-5, 10), "roic": (-0.5, 1.0), "revenue_cagr_3y": (-0.5, 1.0), "momentum_12m": (-0.9, 2.0)}

_lock = threading.Lock()


def raw_features(row: Dict[str, Any], profile: Optional[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    p = profile or {}
    out: Dict[str, Optional[float]] = {}
    for k in FEATURES:
        if k == "log_market_cap":
            v = math.log10(row["market_cap"]) if row.get("market_cap") else None
        elif k == "tam_cagr":
            v = p.get("tam_cagr_5y_pct")
        elif k in ORDINALS:
            v = ORDINALS[k].get(p.get(k))
        else:
            v = row.get(k)
        if v is not None and k in CLIP:
            v = min(max(v, CLIP[k][0]), CLIP[k][1])
        out[k] = v
    return out


class Standardizer:
    """z-scores against the whole universe; a missing value becomes 0 (the mean)."""

    def __init__(self, rows: List[Dict[str, Optional[float]]]):
        self.stats = {}
        for k in FEATURES:
            vals = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
            self.stats[k] = (float(vals.mean()), float(vals.std() or 1.0)) if len(vals) else (0.0, 1.0)

    def vector(self, feats: Dict[str, Optional[float]]) -> np.ndarray:
        return np.array([0.0 if feats.get(k) is None else (feats[k] - m) / s for k, (m, s) in self.stats.items()])


def card(row: Dict[str, Any], profile: Optional[Dict[str, Any]]) -> str:
    """One-line company summary: what the investor saw, and what the LoRA reward model reads."""
    p = profile or {}

    def pct(v: Optional[float]) -> str:
        return "n/a" if v is None else f"{v * 100:.0f}%"

    nd = row.get("net_debt_to_ebitda")
    pe = row.get("pe")
    return (f"{row['ticker']} {row['name']}｜{row['sub_industry']}｜{row.get('size')} {row.get('style')} "
            f"景氣敏感度{row.get('cycle')} 利率{row.get('rate')}｜{p.get('market_structure')} 對供應商{p.get('supplier_power')} "
            f"對客戶{p.get('customer_power')} TAM {p.get('tam_bucket')}｜ROIC {pct(row.get('roic'))} 毛利率 {pct(row.get('gross_margin'))} "
            f"FCF率 {pct(row.get('fcf_margin'))} F分數 {row.get('piotroski')} 淨負債/EBITDA "
            f"{'n/a' if nd is None else format(nd, '.1f')} 3年營收CAGR {pct(row.get('revenue_cagr_3y'))} "
            f"P/E {'n/a' if not pe else format(pe, '.0f')}")


def record(entry: Dict[str, Any]) -> None:
    entry = {**entry, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with _lock:
        FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_PATH.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def load_feedback() -> List[Dict[str, Any]]:
    if not FEEDBACK_PATH.exists():
        return []
    return [json.loads(line) for line in FEEDBACK_PATH.read_text().splitlines() if line.strip()]


def pairs(feedback: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Preference pairs: explicit comparisons, plus every thumbs-up against every thumbs-down in a session.
    A later rating of the same company in a session replaces the earlier one."""
    feedback = load_feedback() if feedback is None else feedback
    out = [{"chosen": f["chosen"], "rejected": f["rejected"], "prompt": f.get("prompt", ""), "session": f.get("session")}
           for f in feedback if f["type"] == "pair"]
    ratings: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for f in feedback:
        if f["type"] == "rating":
            ratings.setdefault(f.get("session", ""), {})[f["chosen"]["ticker"]] = f
    for session, by_ticker in ratings.items():
        ups = [f for f in by_ticker.values() if f["rating"] > 0]
        downs = [f for f in by_ticker.values() if f["rating"] < 0]
        for u, d in product(ups, downs):
            out.append({"chosen": u["chosen"], "rejected": d["chosen"], "prompt": u.get("prompt", ""), "session": session})
    return out


def fit(pair_rows: List[Dict[str, Any]], std: Standardizer, iters: int = 400, lr: float = 0.1) -> Optional[np.ndarray]:
    """Bradley-Terry: P(chosen > rejected) = sigmoid(w . (x_chosen - x_rejected)), L2-regularized."""
    if not pair_rows:
        return None
    X = np.array([std.vector(p["chosen"]["features"]) - std.vector(p["rejected"]["features"]) for p in pair_rows])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        z = X @ w
        grad = -(X * (1 / (1 + np.exp(z)))[:, None]).sum(0) + L2 * w
        w -= lr * grad / len(X)
    return w


def explain(w: Optional[np.ndarray], top: int = 4) -> List[Dict[str, Any]]:
    if w is None:
        return []
    named = sorted(zip(FEATURES.values(), w.tolist()), key=lambda t: abs(t[1]), reverse=True)
    return [{"feature": n, "weight": round(v, 3), "direction": "偏好高" if v > 0 else "偏好低"} for n, v in named[:top] if abs(v) > 0.02]


def rank(rows: List[Dict[str, Any]], profiles: Dict[str, Dict[str, Any]], universe_feats: List[Dict[str, Optional[float]]],
         prompt: str = "") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Order candidates by quality, adjusted by what the investor has preferred so far.

    Returns (rows with score fields, info about the learners used)."""
    std = Standardizer(universe_feats)
    pr = pairs()
    w = fit(pr, std)
    alpha = len(pr) / (len(pr) + PRIOR_PAIRS)
    info: Dict[str, Any] = {"pairs": len(pr), "learned_weight": round(alpha, 2), "preferences": explain(w)}
    for r in rows:
        feats = raw_features(r, profiles.get(r["sub_industry"]))
        r["base_score"] = r.get("quality_score") or 0.0
        r["learned_score"] = float(std.vector(feats) @ w) if w is not None else 0.0
        # learned score is a log-odds; squash to [0, 1] so it mixes with the percentile-based quality score
        r["score"] = (1 - alpha) * r["base_score"] + alpha * (1 / (1 + math.exp(-r["learned_score"])))
    url = os.environ.get("TAURUS_SCREEN_REWARD_URL")
    if url and rows:
        try:
            scores = reward.score(prompt, [card(r, profiles.get(r["sub_industry"])) for r in rows], url=url)
            lo, hi = min(scores), max(scores)
            for r, s in zip(rows, scores):
                r["reward_score"] = (s - lo) / (hi - lo) if hi > lo else 0.5
                r["score"] = 0.5 * r["score"] + 0.5 * r["reward_score"]
            info["reward_model"] = True
        except (OSError, KeyError, ValueError) as exc:
            logger.warning("Screen reward model failed (%s); ranking without it", exc)
    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows, info


def export_pairs() -> List[Dict[str, str]]:
    """{prompt, chosen, rejected} text rows for the LoRA reward model."""
    return [{"prompt": p["prompt"], "chosen": p["chosen"]["card"], "rejected": p["rejected"]["card"]} for p in pairs()]

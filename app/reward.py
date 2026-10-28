"""Optional local models trained on your preferences (see scripts/train_*.py), served over HTTP.

- TAURUS_REWARD_URL: a reward model (LoRA on an open-weight model) that scores candidate texts.
  Section writers then draft TAURUS_BEST_OF candidates and keep the highest-scoring one.
- TAURUS_REWRITER_URL: a DPO-tuned rewriter that tightens prose. Its output is used only when it
  keeps every number of the original (numbers_preserved), so it can never change a figure.

Both are off unless set; the pipeline then runs on Claude alone. Claude itself is not trained.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import List, Optional

logger = logging.getLogger("uvicorn.error")

_NUMBER = re.compile(r"(?<![\w.])[-−]?\$?\d[\d,]*(?:\.\d+)?%?")


def reward_url() -> Optional[str]:
    return os.environ.get("TAURUS_REWARD_URL") or None


def rewriter_url() -> Optional[str]:
    return os.environ.get("TAURUS_REWRITER_URL") or None


def best_of() -> int:
    """Candidates per section; more than one only makes sense with a reward model to pick."""
    n = int(os.environ.get("TAURUS_BEST_OF", "1"))
    return max(1, min(n, 4)) if reward_url() else 1


def _post(url: str, payload: dict, timeout: float = 300.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def score(prompt: str, texts: List[str]) -> List[float]:
    """Reward-model scores for each text (higher is preferred)."""
    scores = _post(f"{reward_url().rstrip('/')}/score", {"prompt": prompt, "texts": texts})["scores"]
    if len(scores) != len(texts):
        raise ValueError(f"reward model returned {len(scores)} scores for {len(texts)} texts")
    return [float(s) for s in scores]


def numbers(text: str) -> List[str]:
    """Numeric tokens, normalized (thousands separators and the Unicode minus removed)."""
    return [n.replace(",", "").replace("−", "-") for n in _NUMBER.findall(text)]


def numbers_preserved(original: str, rewritten: str) -> bool:
    """True if the rewrite introduces no number that is not in the original."""
    return set(numbers(rewritten)) <= set(numbers(original))


def rewrite(prompt: str, text: str) -> Optional[str]:
    """The local rewriter's version of `text`, or None if it is off, fails, or changes a number."""
    if not rewriter_url():
        return None
    try:
        out = _post(f"{rewriter_url().rstrip('/')}/rewrite", {"prompt": prompt, "text": text})["text"].strip()
    except (OSError, KeyError, ValueError) as exc:
        logger.warning("Local rewriter failed (%s); keeping Claude's revision", exc)
        return None
    if not out or not numbers_preserved(text, out):
        logger.warning("Local rewriter changed or added numbers; keeping Claude's revision")
        return None
    return out

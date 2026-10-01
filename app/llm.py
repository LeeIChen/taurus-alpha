"""Shared Claude client and call helpers used by every agent."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel

MODEL = os.environ.get("TAURUS_MODEL", "claude-opus-5-5")
# Opus 5.5 defaults to "medium"; set it explicitly.
EFFORT = os.environ.get("TAURUS_EFFORT", "high")

# Server-side refusal fallback: if a safety classifier declines, the API retries
# on Anthropic's recommended fallback model inside the same call.
_FALLBACK_KWARGS = {
    "betas": ["server-side-fallback-2026-07-01"],
    "fallbacks": "default",
}

T = TypeVar("T", bound=BaseModel)


class ClaudeRefusal(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


def _check_refusal(response) -> None:
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) if response.stop_details else None
        raise ClaudeRefusal(f"Request declined (category={category})")


def ask_structured(system: str, prompt: str, schema: type[T], max_tokens: int = 16000) -> T:
    """Call Claude and return a validated instance of `schema`."""
    response = get_client().beta.messages.parse(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
        output_config={"effort": EFFORT},
        **_FALLBACK_KWARGS,
    )
    _check_refusal(response)
    if response.parsed_output is None:
        raise ValueError(f"No structured output (stop_reason={response.stop_reason})")
    return response.parsed_output


def ask_text(system: str, prompt: str, max_tokens: int = 64000) -> str:
    """Call Claude and return the concatenated text. Streams to avoid timeouts on long output."""
    with get_client().beta.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": EFFORT},
        **_FALLBACK_KWARGS,
    ) as stream:
        response = stream.get_final_message()
    _check_refusal(response)
    return "".join(block.text for block in response.content if block.type == "text")

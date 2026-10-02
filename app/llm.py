"""Shared Claude client and call helpers used by every agent."""

from __future__ import annotations

import logging
import os
import time
from functools import lru_cache
from typing import TypeVar

import anthropic
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

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


# Summarized thinking keeps data flowing on the stream while Claude reasons, so a
# silent connection really is stalled; the read timeout then drops and retries it
# instead of hanging (seen in practice on long analysis calls).
_THINKING = {"type": "adaptive", "display": "summarized"}
_TIMEOUT = anthropic.Timeout(600.0, read=180.0, connect=10.0)


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(timeout=_TIMEOUT)


def _check_response(response, label: str, started: float) -> None:
    usage = response.usage
    logger.info(
        "Claude %s: %.0fs, %d input / %d output tokens, stop_reason=%s",
        label, time.monotonic() - started, usage.input_tokens, usage.output_tokens, response.stop_reason,
    )
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) if response.stop_details else None
        raise ClaudeRefusal(f"Request declined (category={category})")
    if response.stop_reason == "max_tokens":
        raise ValueError(f"Claude {label} hit max_tokens before finishing")


def ask_structured(system: str, prompt: str, schema: type[T], max_tokens: int = 64000) -> T:
    """Call Claude and return a validated instance of `schema`.

    Streams so that long outputs (thinking plus a large structured answer) don't hit
    the SDK's non-streaming request timeout and get silently retried.
    """
    started = time.monotonic()
    with get_client().beta.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_format=schema,
        output_config={"effort": EFFORT},
        thinking=_THINKING,
        **_FALLBACK_KWARGS,
    ) as stream:
        response = stream.get_final_message()
    _check_response(response, schema.__name__, started)
    if response.parsed_output is None:
        raise ValueError(f"No structured output (stop_reason={response.stop_reason})")
    return response.parsed_output


def ask_text(system: str, prompt: str, max_tokens: int = 64000) -> str:
    """Call Claude and return the concatenated text. Streams to avoid timeouts on long output."""
    started = time.monotonic()
    with get_client().beta.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": EFFORT},
        thinking=_THINKING,
        **_FALLBACK_KWARGS,
    ) as stream:
        response = stream.get_final_message()
    _check_response(response, "text", started)
    return "".join(block.text for block in response.content if block.type == "text")

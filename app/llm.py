"""Shared Claude client and call helpers used by every agent."""

from __future__ import annotations

import logging
import os
import re
import time
from functools import lru_cache
from typing import Optional, TypeVar

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
        "Claude %s: %.0fs, %d input (+%d cache read, +%d cache write) / %d output tokens, stop_reason=%s",
        label, time.monotonic() - started, usage.input_tokens,
        getattr(usage, "cache_read_input_tokens", 0) or 0, getattr(usage, "cache_creation_input_tokens", 0) or 0,
        usage.output_tokens, response.stop_reason,
    )
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) if response.stop_details else None
        raise ClaudeRefusal(f"Request declined (category={category})")
    if response.stop_reason == "max_tokens":
        raise ValueError(f"Claude {label} hit max_tokens before finishing")


# ---- Prompt caching ----
# Steps after retrieval (analysis, valuation, thesis, memo, verifier) read the same large
# evidence block. A cache hit needs a byte-identical prefix, and changing the system prompt
# invalidates the messages cache, so those steps share one frozen system prompt and put the
# evidence first (cached); each step's own instructions and data come after the breakpoint.
# 1-hour TTL: consecutive steps can start more than 5 minutes apart (the memo alone can take ~4).
SHARED_SYSTEM = """You are a member of an equity research team. The first part of the user message is
the shared evidence for this company. Follow the task instructions that come after it exactly."""
_EVIDENCE_CACHE = {"type": "ephemeral", "ttl": "1h"}


def _request(system: str, prompt: str, context: Optional[str]) -> dict:
    if context is None:
        return {"system": system, "messages": [{"role": "user", "content": prompt}]}
    return {
        "system": SHARED_SYSTEM,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": context, "cache_control": _EVIDENCE_CACHE},
            {"type": "text", "text": f"<task_instructions>\n{system}\n</task_instructions>\n\n{prompt}"},
        ]}],
    }


def ask_structured(system: str, prompt: str, schema: type[T], max_tokens: int = 64000,
                   context: Optional[str] = None) -> T:
    """Call Claude and return a validated instance of `schema`.

    Pass `context` (shared evidence) to put it first and cache it across steps.
    Streams so that long outputs (thinking plus a large structured answer) don't hit
    the SDK's non-streaming request timeout and get silently retried.
    """
    started = time.monotonic()
    with get_client().beta.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        **_request(system, prompt, context),
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


def ask_text(system: str, prompt: str, max_tokens: int = 64000, context: Optional[str] = None) -> str:
    """Call Claude and return the concatenated text. Streams to avoid timeouts on long output.
    Pass `context` to cache shared evidence (see ask_structured)."""
    started = time.monotonic()
    with get_client().beta.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        **_request(system, prompt, context),
        output_config={"effort": EFFORT},
        thinking=_THINKING,
        **_FALLBACK_KWARGS,
    ) as stream:
        response = stream.get_final_message()
    _check_response(response, "text", started)
    return "".join(block.text for block in response.content if block.type == "text")


# ---- Web research (server-side tools: Anthropic runs the searches) ----
WEB_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 12},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 6},
]
MAX_CONTINUATIONS = 5


def ask_with_web(system: str, prompt: str, max_tokens: int = 64000) -> tuple:
    """Research with web search/fetch. Returns (notes_text, sources).

    `sources` is a list of dicts {url, title, cited, snippet} in first-seen order.
    The model cites each fact with its URL in angle brackets; those are replaced with source
    numbers such as "[S3]" and the sources marked cited. (The dynamic-filtering web search tool
    returns text without structured citations, so inline URLs are the reliable link.) Structured
    citations, when present, are used too.
    A turn that hits the server-side tool-iteration limit (stop_reason "pause_turn")
    is resumed by re-sending the conversation, as the API expects.
    """
    started = time.monotonic()
    messages = [{"role": "user", "content": prompt}]
    sources: dict = {}  # url -> {url, title, cited, snippet, n}
    parts = []

    def source_number(url: str, title: str = "") -> int:
        if url not in sources:
            sources[url] = {"url": url, "title": title or url, "cited": False, "snippet": "", "n": len(sources) + 1}
        elif title and sources[url]["title"] == url:
            sources[url]["title"] = title
        return sources[url]["n"]

    response = None
    assistant_blocks: list = []  # everything produced so far, for resuming paused turns
    for _ in range(MAX_CONTINUATIONS + 1):
        with get_client().beta.messages.stream(
            model=MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=WEB_TOOLS,
            # Automatic caching: with it on, server-side search writes cache entries after tool
            # results, so each search iteration re-reads the growing context at the cache rate.
            cache_control={"type": "ephemeral"},
            output_config={"effort": EFFORT},
            thinking=_THINKING,
            **_FALLBACK_KWARGS,
        ) as stream:
            response = stream.get_final_message()
        for block in response.content:
            if block.type == "web_search_tool_result" and isinstance(block.content, list):
                for result in block.content:
                    if getattr(result, "type", "") == "web_search_result":
                        source_number(result.url, getattr(result, "title", ""))
            elif block.type == "web_fetch_tool_result":
                url = getattr(block.content, "url", None)
                if url:
                    doc = getattr(block.content, "content", None)
                    source_number(url, getattr(doc, "title", "") or "")
            elif block.type == "text":
                refs = []
                for cit in getattr(block, "citations", None) or []:
                    url = getattr(cit, "url", None)
                    if url:
                        n = source_number(url, getattr(cit, "title", "") or "")
                        sources[url]["cited"] = True
                        if not sources[url]["snippet"]:
                            sources[url]["snippet"] = (getattr(cit, "cited_text", "") or "")[:300]
                        refs.append(n)
                parts.append(block.text + ("".join(f" [S{n}]" for n in dict.fromkeys(refs)) if refs else ""))
        assistant_blocks.extend(response.content)
        if response.stop_reason != "pause_turn":
            break
        # Resume: original request plus everything the assistant produced so far.
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": assistant_blocks}]
    _check_response(response, "web research", started)

    def cite(match: "re.Match") -> str:
        url = match.group(1).rstrip(".,;)")
        n = source_number(url)
        sources[url]["cited"] = True
        return f"[S{n}]"

    notes = _INLINE_URL.sub(cite, "".join(parts)).strip()
    ordered = sorted(sources.values(), key=lambda s: s["n"])
    return notes, [{k: v for k, v in s.items() if k != "n"} for s in ordered]


_INLINE_URL = re.compile(r"<(https?://[^>\s]+)>")

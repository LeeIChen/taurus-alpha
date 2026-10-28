"""Shared Claude client and call helpers used by every agent."""

from __future__ import annotations

import contextvars
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from functools import lru_cache
from typing import List, Optional, TypeVar

import anthropic
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

MODEL = os.environ.get("TAURUS_MODEL", "claude-opus-5-5")
# Opus 5.5 defaults to "medium"; set it explicitly.
EFFORT = os.environ.get("TAURUS_EFFORT", "high")
# Writing steps that don't read the shared evidence cache (section writers, editor, reviser) can use
# a cheaper model and effort without breaking the cache, which is scoped to MODEL and EFFORT.
WRITER_MODEL = os.environ.get("TAURUS_WRITER_MODEL", "claude-sonnet-5-5")
WRITER_EFFORT = os.environ.get("TAURUS_WRITER_EFFORT", "medium")

# USD per million tokens. Cache writes cost 1.25x input (5-minute TTL) or 2x (1-hour TTL).
# Check https://www.anthropic.com/pricing when models or prices change.
PRICES = {
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.0, "output": 10.0, "cache_read": 0.20},
}
WEB_SEARCH_USD = 10.0 / 1000

# Server-side refusal fallback: if a safety classifier declines, the API retries
# on Anthropic's recommended fallback model inside the same call.
_FALLBACK_KWARGS = {
    "betas": ["server-side-fallback-2026-07-01"],
    "fallbacks": "default",
}

T = TypeVar("T", bound=BaseModel)


class ClaudeRefusal(RuntimeError):
    pass


class CallTimeout(TimeoutError):
    """A Claude call ran past its wall-clock deadline and was stopped."""


# Wall-clock limits. The read timeout below only fires on a silent connection; a stream that keeps
# sending something without finishing (seen once in web research: no result after 90 minutes) needs
# a deadline too. A failed run saves a checkpoint, so it can resume from the step that timed out.
CALL_DEADLINE_S = float(os.environ.get("TAURUS_CALL_DEADLINE_S", "900"))
WEB_TURN_DEADLINE_S = float(os.environ.get("TAURUS_WEB_TURN_DEADLINE_S", "720"))


# Summarized thinking keeps data flowing on the stream while Claude reasons, so a
# silent connection really is stalled; the read timeout then drops and retries it
# instead of hanging (seen in practice on long analysis calls).
_THINKING = {"type": "adaptive", "display": "summarized"}
_TIMEOUT = anthropic.Timeout(600.0, read=180.0, connect=10.0)


@lru_cache(maxsize=1)
def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(timeout=_TIMEOUT)


# Streams are read on worker threads so the caller can stop waiting at the deadline: closing a
# stream from another thread does not interrupt a blocked socket read (tested). A stopped worker
# exits on its own once its read timeout fires.
_STREAM_POOL = ThreadPoolExecutor(max_workers=32, thread_name_prefix="claude-stream")


def _final_message(stream, label: str, deadline_s: float, on_event=None):
    """Read a stream to its final message, giving up with CallTimeout after `deadline_s`."""
    def consume():
        for event in stream:
            if on_event:
                on_event(event)
        return stream.get_final_message()

    future = _STREAM_POOL.submit(contextvars.copy_context().run, consume)
    try:
        return future.result(timeout=deadline_s)
    except FutureTimeout:
        stream.close()
        raise CallTimeout(f"Claude {label} ran past {deadline_s:.0f}s and was stopped") from None


# ---- Cost meter ----
# main.py starts a meter per run; every call made inside it (including from the parallel section
# writers, which run in context-copying worker threads) appends one row.
_METER: contextvars.ContextVar[Optional[List[dict]]] = contextvars.ContextVar("taurus_meter", default=None)


def start_meter() -> List[dict]:
    rows: List[dict] = []
    _METER.set(rows)
    return rows


def meter_rows() -> List[dict]:
    """The current run's cost rows so far (empty outside a metered run)."""
    rows = _METER.get()
    return rows if rows is not None else []


def _current_step() -> str:
    try:
        from langgraph.config import get_config
        return get_config().get("metadata", {}).get("langgraph_node", "")
    except (ImportError, RuntimeError):  # called outside a graph run
        return ""


def call_cost(model: str, usage) -> dict:
    """Token counts and USD cost of one response."""
    price = PRICES.get(model) or PRICES.get(re.sub(r"-\d{8}$", "", model)) or PRICES[MODEL]
    read = getattr(usage, "cache_read_input_tokens", 0) or 0
    write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    split = getattr(usage, "cache_creation", None)
    write_1h = (getattr(split, "ephemeral_1h_input_tokens", 0) or 0) if split else 0
    searches = getattr(getattr(usage, "server_tool_use", None), "web_search_requests", 0) or 0
    usd = (usage.input_tokens * price["input"] + read * price["cache_read"]
           + (write - write_1h) * price["input"] * 1.25 + write_1h * price["input"] * 2
           + usage.output_tokens * price["output"]) / 1e6 + searches * WEB_SEARCH_USD
    return {"model": model, "input": usage.input_tokens, "cache_read": read, "cache_write": write,
            "output": usage.output_tokens, "web_searches": searches, "usd": round(usd, 4)}


def summarize_costs(rows: List[dict]) -> dict:
    """Total and per-step cost for a run record."""
    by_step: dict = {}
    for r in rows:
        s = by_step.setdefault(r["step"] or "other", {"calls": 0, "usd": 0.0, "input": 0, "cache_read": 0,
                                                       "cache_write": 0, "output": 0, "web_searches": 0})
        s["calls"] += 1
        for k in ("usd", "input", "cache_read", "cache_write", "output", "web_searches"):
            s[k] += r[k]
    for s in by_step.values():
        s["usd"] = round(s["usd"], 4)
    return {"total_usd": round(sum(r["usd"] for r in rows), 4), "by_step": by_step, "calls": rows}


def _record(response, label: str, started: float) -> None:
    """Log one response's tokens and cost, and add it to the run's meter."""
    usage = response.usage
    cost = call_cost(getattr(response, "model", None) or MODEL, usage)
    row = {"step": _current_step(), "label": label, "seconds": round(time.monotonic() - started, 1), **cost}
    meter = _METER.get()
    if meter is not None:
        meter.append(row)
    logger.info(
        "Claude %s %s (%s): %.0fs, %d input (+%d cache read, +%d cache write) / %d output tokens, "
        "%d searches, $%.3f, stop_reason=%s",
        row["step"] or "-", label, cost["model"], row["seconds"], usage.input_tokens, cost["cache_read"],
        cost["cache_write"], usage.output_tokens, cost["web_searches"], cost["usd"], response.stop_reason,
    )


def _check_response(response, label: str, started: float) -> None:
    _record(response, label, started)
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None) if response.stop_details else None
        raise ClaudeRefusal(f"Request declined (category={category})")
    if response.stop_reason == "max_tokens":
        raise ValueError(f"Claude {label} hit max_tokens before finishing")


# ---- Prompt caching ----
# Steps after retrieval (analysis, valuation, thesis, verifier) read the same large
# evidence block; the section writers, editor and reviser don't, so they can use WRITER_MODEL. A cache hit needs a byte-identical prefix, and changing the system prompt
# invalidates the messages cache, so those steps share one frozen system prompt and put the
# evidence first (cached); each step's own instructions and data come after the breakpoint.
# 5-minute TTL: each read restarts the timer, and steps run back to back (each took 45-140 s in the
# Wayfair run). The 1-hour TTL doubles the write price; a miss after a memo over 5 minutes costs one re-write.
SHARED_SYSTEM = """You are a member of an equity research team. The first part of the user message is
the shared evidence for this company. Follow the task instructions that come after it exactly."""
_EVIDENCE_CACHE = {"type": "ephemeral", "ttl": "5m"}


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
        response = _final_message(stream, schema.__name__, CALL_DEADLINE_S)
    _check_response(response, schema.__name__, started)
    if response.parsed_output is None:
        raise ValueError(f"No structured output (stop_reason={response.stop_reason})")
    return response.parsed_output


def ask_text(system: str, prompt: str, max_tokens: int = 64000, context: Optional[str] = None,
             read_timeout: Optional[float] = None, model: Optional[str] = None, effort: Optional[str] = None) -> str:
    """Call Claude and return the concatenated text. Streams to avoid timeouts on long output.
    Pass `context` to cache shared evidence (see ask_structured).

    `read_timeout` allows longer silences on the stream and turns off the SDK's automatic
    retry, which would start the whole answer again: used for the long memo, where a failed
    run resumes from its checkpoint instead. `model` and `effort` override the defaults; only
    calls without `context` should do that, since the evidence cache is scoped to both."""
    started = time.monotonic()
    client = get_client()
    if read_timeout:
        client = client.with_options(timeout=anthropic.Timeout(600.0, read=read_timeout, connect=10.0), max_retries=0)
    with client.beta.messages.stream(
        model=model or MODEL,
        max_tokens=max_tokens,
        **_request(system, prompt, context),
        output_config={"effort": effort or EFFORT},
        thinking=_THINKING,
        **_FALLBACK_KWARGS,
    ) as stream:
        response = _final_message(stream, "text", CALL_DEADLINE_S)
    _check_response(response, "text", started)
    return "".join(block.text for block in response.content if block.type == "text")


_JSON_BLOCK = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)


def ask_json(system: str, prompt: str, schema: type[T], context: Optional[str] = None,
             model: Optional[str] = None, effort: Optional[str] = None) -> T:
    """Like ask_structured, but the schema goes in the prompt and the reply is validated here.

    Used for the cached steps: a structured-output schema renders ahead of the messages, so
    it must be identical across steps, and one schema covering every step is too large for
    the API to compile. Retries once with the validation error if the reply doesn't parse.
    """
    import json

    ask = (f"{prompt}\n\nReply with one JSON object matching this JSON Schema, in a ```json block:\n"
           f"{json.dumps(schema.model_json_schema())}")
    text = ask_text(system, ask, context=context, model=model, effort=effort)
    for attempt in range(2):
        match = _JSON_BLOCK.search(text)
        raw = match.group(1) if match else text[text.find("{"): text.rfind("}") + 1]
        try:
            return schema.model_validate_json(raw)
        except ValueError as exc:
            if attempt:
                raise
            logger.warning("Claude %s reply did not validate; retrying once: %s", schema.__name__, str(exc)[:300])
            text = ask_text(system, f"{ask}\n\nYour previous reply failed validation:\n{str(exc)[:2000]}\n"
                            f"Previous reply:\n{text}", context=context, model=model, effort=effort)


# ---- Web research (server-side tools: Anthropic runs the searches) ----
WEB_TOOLS = [
    {"type": "web_search_20260209", "name": "web_search", "max_uses": 8},
    {"type": "web_fetch_20260209", "name": "web_fetch", "max_uses": 4},
]
MAX_CONTINUATIONS = 5


def ask_with_web(system: str, prompt: str, max_tokens: int = 64000, model: Optional[str] = None,
                 effort: Optional[str] = None, tools: Optional[list] = None) -> tuple:
    """Research with web search/fetch. Returns (notes_text, sources).

    `sources` is a list of dicts {url, title, cited, snippet} in first-seen order.
    The model cites each fact with its URL in angle brackets; those are replaced with source
    numbers such as "[S3]" and the sources marked cited. (The dynamic-filtering web search tool
    returns text without structured citations, so inline URLs are the reliable link.) Structured
    citations, when present, are used too. `model`, `effort` and `tools` override the defaults
    (the screener checks one industry's market size with fewer searches on a cheaper model).
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

    counts = {"web_search_tool_result": 0, "web_fetch_tool_result": 0}

    def progress(event) -> None:
        """One log line per finished search or fetch, so a slow or stuck turn is visible."""
        kind = getattr(getattr(event, "content_block", None), "type", "")
        if event.type == "content_block_start" and kind in counts:
            counts[kind] += 1
            logger.info("Web research progress: %d searches, %d fetches, %.0fs", counts["web_search_tool_result"],
                        counts["web_fetch_tool_result"], time.monotonic() - started)

    response = None
    assistant_blocks: list = []  # everything produced so far, for resuming paused turns
    for _ in range(MAX_CONTINUATIONS + 1):
        with get_client().beta.messages.stream(
            model=model or MODEL,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            tools=tools or WEB_TOOLS,
            # Automatic caching: with it on, server-side search writes cache entries after tool
            # results, so each search iteration re-reads the growing context at the cache rate.
            cache_control={"type": "ephemeral"},
            output_config={"effort": effort or EFFORT},
            thinking=_THINKING,
            **_FALLBACK_KWARGS,
        ) as stream:
            response = _final_message(stream, "web research", WEB_TURN_DEADLINE_S, on_event=progress)
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
        _record(response, "web research (paused turn)", started)  # every billed turn counts
        if time.monotonic() - started > 2 * WEB_TURN_DEADLINE_S:
            logger.warning("Web research stopped after %.0fs; keeping the notes so far", time.monotonic() - started)
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

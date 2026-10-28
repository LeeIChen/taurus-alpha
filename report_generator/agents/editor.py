"""Editor: reviews the memo for clarity and concision; the reviser applies its notes.

assemble -> editor -> (reviser -> editor)* -> verifier, at most MAX_REVISIONS rewrites.
The editor judges readability only; the verifier still checks every fact against the evidence.
Both run on WRITER_MODEL with the memo alone (no shared evidence): the reviser may not add facts,
so it needs only the memo and the notes.
"""

from __future__ import annotations

import re
from typing import List

from pydantic import BaseModel, Field

from core import reward
from report_generator.agents.schemas import MEMO_READ_TIMEOUT
from core.llm import WRITER_EFFORT, WRITER_MODEL, ask_json, ask_text
from report_generator.state import InvestmentAgentState

MAX_REVISIONS = 1
# Prose budget: the section budgets in report_generator/agents/sections.py add up to about 2,100 words.
MAX_WORDS = 2400
REQUIRED_SECTIONS = 8
_PROCESS_TALK = re.compile(r"not retrieved|were not available|research tasks?|we searched|未取得|研究任務", re.I)

EDIT_SYSTEM = """You are the editor of an equity research team. A portfolio manager should get the
call, the reasons and the numbers that matter in a few minutes of reading. Review the memo against
this rubric and list what must change:
- The fixed outline: title line, sections 1-8 in order, then Sources. Nothing extra.
- Each section opens with its conclusion in one sentence.
- No fact or figure repeated across sections; later sections refer back instead.
- No process commentary (task lists, "we searched", "not retrieved"); a data gap appears only where it
  changes a conclusion, in one clause.
- Prose over fragmented bullets; tables for numbers; no section over its word budget.
- Sections 3-5 each end with the implication for the target price or catalyst path.
- Citations at the end of sentences or short paragraphs, not after every clause.
Do not judge whether facts or numbers are right; a separate verifier does that. Approve the memo
when nothing material is left to fix. Otherwise give at most 8 issues, each quoting or naming the
passage and saying exactly how to fix it."""

REVISE_SYSTEM = """You are revising an equity research memo to apply the editor's notes. Apply every
note. Keep the rating, price target, every number and every citation as they are, and add no fact
that is not already in the memo or the evidence. Keep the outline and headings. Return the full
revised memo."""
# The rewriter's training prompt (report_generator/training/train_dpo_rewriter.py) must match REVISE_SYSTEM.


class EditReview(BaseModel):
    approved: bool
    issues: List[str] = Field(description="What to change, each naming the passage and the fix; empty if approved")


def prose_words(memo: str) -> int:
    """Word count outside tables; each CJK character counts as half a word."""
    prose = "\n".join(line for line in memo.splitlines() if not line.lstrip().startswith("|"))
    prose = re.sub(r"\[[^\]]*\]|https?://\S+", "", prose)  # citations and URLs
    cjk = len(re.findall(r"[㐀-鿿]", prose))
    return len(re.findall(r"[A-Za-z0-9$%][\w$%.,'’-]*", prose)) + cjk // 2


def mechanical_issues(memo: str) -> List[str]:
    """Checks that need no model: outline, length and process talk."""
    body = re.split(r"^## (?!\d+\.)", memo, flags=re.M)[0]  # up to the unnumbered Sources heading
    issues = []
    numbered = re.findall(r"^## (\d+)\.", body, re.M)
    if numbered != [str(i) for i in range(1, REQUIRED_SECTIONS + 1)]:
        issues.append(f"Use the fixed outline: sections 1-{REQUIRED_SECTIONS} in order (found {numbered or 'none'}).")
    words = prose_words(body)
    if words > MAX_WORDS:
        issues.append(f"The memo is {words} words of prose; cut it to under {MAX_WORDS} by removing repetition and detail.")
    talk = sorted({m.group(0) for m in _PROCESS_TALK.finditer(body)})
    if talk:
        issues.append("Remove process commentary from the memo: " + ", ".join(f'"{t}"' for t in talk) + ".")
    return issues


def editor_node(state: InvestmentAgentState) -> dict:
    memo = state["draft_report"]
    checks = mechanical_issues(memo)
    review = ask_json(EDIT_SYSTEM, f"Memo:\n{memo}\n\nAutomatic checks found:\n" +
                      ("\n".join(f"- {c}" for c in checks) or "- nothing"), EditReview,
                      model=WRITER_MODEL, effort=WRITER_EFFORT)
    issues = checks + ([] if review.approved else review.issues)
    return {"edit_notes": issues, "current_step": "editor"}


def reviser_node(state: InvestmentAgentState) -> dict:
    notes = "\n".join(f"- {n}" for n in state.get("edit_notes") or [])
    memo = reward.rewrite(f"{REVISE_SYSTEM}\n\nEditor's notes:\n{notes}", state["draft_report"]) or ask_text(
        REVISE_SYSTEM, f"Memo to revise:\n{state['draft_report']}\n\nEditor's notes:\n{notes}\n\n"
        "Reply with the full memo in Markdown only.", read_timeout=MEMO_READ_TIMEOUT,
        model=WRITER_MODEL, effort=WRITER_EFFORT).strip()
    if not memo:
        raise ValueError("The reviser returned no text")
    return {
        "draft_report": memo,
        "draft_versions": [*(state.get("draft_versions") or []), state["draft_report"]],
        "edit_history": [*(state.get("edit_history") or []), list(state.get("edit_notes") or [])],
        "revisions": (state.get("revisions") or 0) + 1,
        "current_step": "reviser",
    }


def after_editor(state: InvestmentAgentState) -> str:
    """Revise while the editor has notes and the revision budget lasts, then verify."""
    if state.get("edit_notes") and (state.get("revisions") or 0) < MAX_REVISIONS:
        return "reviser"
    return "verifier"

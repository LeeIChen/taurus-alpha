"""Output schemas for the steps after retrieval, and the shared call that keeps them cached.

The structured-output schema is rendered ahead of the messages, so a different schema per
step would change the prompt prefix and miss the evidence cache. Every step therefore uses
the same StepOutput schema and fills only its own section.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.llm import ask_structured
from app.state import Citation, InvestmentAgentState, InvestmentThesis, PipelineAssumptions, ValuationAssumptions


class MetricDraft(BaseModel):
    metric_name: str
    formula_used: str = Field(description="Formula in words or symbols, e.g. 'EBITDA / Revenue'")
    code: str = Field(description="Python script that prints only the metric's numeric value")


class AnalysisDraft(BaseModel):
    metrics: List[MetricDraft]
    citations: List[Citation]


class ValuationDraft(BaseModel):
    method: str = Field(description="'pipeline_sotp' or 'company_dcf'")
    method_reason: str
    current_price: Optional[float] = Field(description="Latest share price in USD, or null if unknown")
    current_price_source: str = Field(description="Source id and date for the price, or 'user-provided'")
    pipeline: Optional[PipelineAssumptions] = Field(description="Required when method is pipeline_sotp, else null")
    company_dcf: Optional[ValuationAssumptions] = Field(description="Required when method is company_dcf, else null")


class FaithfulnessVerdict(BaseModel):
    supported_claims: int
    total_claims: int
    unsupported_claims: List[str] = Field(description="Quote each claim the sources do not support")


class StepOutput(BaseModel):
    analysis: Optional[AnalysisDraft] = Field(description="Filled by the analysis step only")
    valuation: Optional[ValuationDraft] = Field(description="Filled by the valuation step only")
    thesis: Optional[InvestmentThesis] = Field(description="Filled by the thesis step only")
    memo_markdown: Optional[str] = Field(description="Filled by the memo step only: the full memo in Markdown")
    verdict: Optional[FaithfulnessVerdict] = Field(description="Filled by the verifier step only")


def run_step(section: str, instructions: str, prompt: str, state: InvestmentAgentState, evidence: str):
    """Run one step against the cached evidence and return its section of StepOutput."""
    out = ask_structured(
        f"{instructions}\n\nFill only the `{section}` section of the output; set every other section to null.",
        prompt,
        StepOutput,
        context=evidence,
    )
    value = getattr(out, section)
    if value is None:
        raise ValueError(f"The {section} step returned no {section} section")
    return value

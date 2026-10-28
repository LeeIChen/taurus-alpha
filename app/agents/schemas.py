"""Output schemas for the steps after retrieval, and the shared call that keeps them cached.

A structured-output schema renders ahead of the messages, so a different schema per step
would change the prompt prefix and miss the evidence cache, and one schema covering every
step is too large for the API to compile. These steps therefore ask for JSON in the prompt
(after the cached evidence) and validate it locally.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from app.llm import ask_json
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


# Long writing calls can think silently for minutes (a 180 s read timeout failed a real run).
MEMO_READ_TIMEOUT = 600.0

STEP_SCHEMAS = {
    "analysis": AnalysisDraft,
    "valuation": ValuationDraft,
    "thesis": InvestmentThesis,
    "verdict": FaithfulnessVerdict,
}


def run_step(section: str, instructions: str, prompt: str, state: InvestmentAgentState, evidence: str):
    """Run one step against the cached evidence and return its validated model."""
    return ask_json(instructions, prompt, STEP_SCHEMAS[section], context=evidence)

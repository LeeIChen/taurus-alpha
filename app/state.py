from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field

# ---- Structured schemas for agent outputs ----
class TaskPlan(BaseModel):
    tasks: List[str] = Field(description="List of decomposed sub-tasks")
    required_metrics: List[str] = Field(description="Financial metrics to compute, e.g. EBITDA, CAGR")

class FinancialMetricResult(BaseModel):
    metric_name: str
    value: float
    formula_used: str
    code_executed: str

class Citation(BaseModel):
    source_doc: str
    page_number: int
    content_snippet: str

class ValuationAssumptions(BaseModel):
    base_fcf: float = Field(description="Latest annual or trailing-four-quarter free cash flow (operating cash flow minus capex), USD millions")
    fcf_growth_rates: List[float] = Field(description="Projected FCF growth for each of the next 5 years, as decimals (0.25 = 25%)")
    terminal_growth: float = Field(description="Perpetual FCF growth after the projection period, as a decimal")
    discount_rate: float = Field(description="Discount rate (WACC), as a decimal")
    net_cash: float = Field(description="Cash and marketable securities minus total debt, USD millions (negative for net debt)")
    diluted_shares: float = Field(description="Diluted shares outstanding, millions")
    forward_eps: float = Field(description="Projected diluted EPS for the next 12 months, USD")
    target_pe: float = Field(description="P/E multiple applied to forward EPS")
    dcf_weight: float = Field(description="Weight of the DCF value in the blended target, 0-1; the P/E value gets the rest")
    rationale: str = Field(description="Why these assumptions; cite the source page for every reported figure")

class ValuationResult(BaseModel):
    assumptions: ValuationAssumptions
    dcf_value_per_share: float
    pe_target_price: float
    target_price: float
    current_price: Optional[float] = None
    upside_pct: Optional[float] = None
    # Rows of {"discount_rate", "terminal_growth", "value_per_share"}
    dcf_sensitivity: List[Dict[str, float]] = Field(default_factory=list)

# ---- LangGraph global state ----
class InvestmentAgentState(TypedDict):
    company_name: str
    user_query: str
    plan: Optional[TaskPlan]
    retrieved_docs: List[Dict[str, Any]]
    financial_results: List[FinancialMetricResult]
    citations: List[Citation]
    current_price: Optional[float]
    valuation: Optional[ValuationResult]
    draft_report: str
    faithfulness_score: float
    current_step: str
    error: Optional[str]

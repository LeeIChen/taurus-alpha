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

class Source(BaseModel):
    source_id: str  # "S1", "S2", ... referenced in notes, thesis and memo
    title: str
    url: str
    kind: str = "web"  # "web" or "filing"
    cited: bool = False  # True if a statement in the research notes cites it
    snippet: str = ""

# ---- Pipeline (sum-of-the-parts, risk-adjusted NPV) valuation ----
class ProgramAssumption(BaseModel):
    name: str = Field(description="Product or program and indication, e.g. 'Intismeran: adjuvant melanoma'")
    group: str = Field(description="Franchise grouping, e.g. 'COVID', 'Flu', 'Oncology'")
    stage: str = Field(description="Marketed, Approved, Filed, Phase 3, Phase 2, Phase 1")
    probability_of_success: float = Field(description="0-1; 1.0 for marketed products")
    pos_basis: str = Field(description="Benchmark used (e.g. BIO 2011-2020 phase success rate) and the program-specific adjustment, with source ids like [S4]")
    launch_year: int
    current_sales: float = Field(description="Latest annual net sales for marketed products, USD millions; 0 if not marketed")
    peak_sales: float = Field(description="Peak annual global net sales for the indication, USD millions")
    peak_year: int
    decline_start_year: int = Field(description="First year sales decline (loss of exclusivity or market erosion)")
    annual_decline: float = Field(description="Yearly decline after decline_start_year, as a decimal (0.10 = 10%)")
    contribution_margin: float = Field(description="Margin at peak after COGS, selling costs and lifecycle R&D, as a decimal")
    economic_share: float = Field(description="Company's share of program economics, 0-1 (e.g. 0.5 for a 50/50 partnership)")
    remaining_dev_cost_per_year: float = Field(description="Company-share development cost per year until readout, USD millions")
    readout_year: int = Field(description="Year the key risk resolves; later development costs are probability-weighted")

class PipelineAssumptions(BaseModel):
    programs: List[ProgramAssumption]
    corporate_overhead_per_year: float = Field(description="Unallocated platform R&D and corporate costs, USD millions per year")
    net_cash: float = Field(description="Cash and investments minus debt (including convertibles), USD millions")
    other_adjustments: float = Field(description="Probability-weighted contingent liabilities (negative) or assets (positive), USD millions")
    other_adjustments_note: str = Field(description="What other_adjustments covers, with source ids")
    diluted_shares: float = Field(description="Diluted shares outstanding, millions")
    discount_rate: float = Field(description="Discount rate, as a decimal")
    tax_free_until_year: int = Field(description="Last year profits are shielded by NOLs (0% tax)")
    long_run_tax_rate: float = Field(description="Tax rate after NOLs run out, as a decimal")
    pe_year: int = Field(description="Year whose risk-adjusted EPS the P/E method uses (pick a year near steady state)")
    target_pe: float
    dcf_weight: float = Field(description="Weight of the DCF value in the blended target, 0-1; P/E gets the rest")
    rationale: str = Field(description="Why these assumptions; cite source ids and filing pages")

class ProgramValue(BaseModel):
    name: str
    group: str
    stage: str
    probability_of_success: float
    rnpv: float  # USD millions, risk-adjusted
    unrisked_npv: float
    per_share: float

class PipelineValuationResult(BaseModel):
    assumptions: PipelineAssumptions
    programs: List[ProgramValue]
    corporate_value: float
    dcf_value_per_share: float  # sum-of-the-parts rNPV today
    unrisked_value_per_share: float
    pe_eps: float
    pe_target_price: float  # P/E value at the 12-month target date
    target_price: float  # blended, 12-month
    current_price: Optional[float] = None
    upside_pct: Optional[float] = None
    implied_note: str = ""

# ---- Investment thesis ----
class Catalyst(BaseModel):
    timing: str = Field(description="Date or window, e.g. 'Oct 9, 2026' or 'H1 2027'")
    event: str
    expected_impact: str = Field(description="What it means for the stock and our thesis")
    direction: str = Field(description="'up', 'down' or 'two-way'")
    source_ids: List[str] = Field(default_factory=list)

class InvestmentThesis(BaseModel):
    rating: str = Field(description="Buy, Hold or Sell")
    price_target: float
    horizon_months: int
    current_price: Optional[float] = None
    current_price_source: str = Field(description="Source id and date of the share price used, or 'user-provided'")
    summary: str = Field(description="Two to three sentence investment thesis")
    thesis_points: List[str] = Field(description="The key arguments, each citing source ids or filing pages")
    risks_to_thesis: List[str]
    catalysts: List[Catalyst]
    what_would_change_our_view: List[str]

# ---- LangGraph global state ----
class InvestmentAgentState(TypedDict):
    company_name: str
    user_query: str
    ticker: Optional[str]
    plan: Optional[TaskPlan]
    reported_financials: Dict[str, Any]
    research_notes: str
    sources: List[Source]
    retrieved_docs: List[Dict[str, Any]]
    financial_results: List[FinancialMetricResult]
    citations: List[Citation]
    current_price: Optional[float]
    price_source: Optional[str]
    valuation_method: Optional[str]
    valuation: Optional[ValuationResult]
    pipeline_valuation: Optional[PipelineValuationResult]
    thesis: Optional[InvestmentThesis]
    draft_report: str
    faithfulness_score: float
    current_step: str
    error: Optional[str]

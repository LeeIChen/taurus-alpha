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

# ---- LangGraph global state ----
class InvestmentAgentState(TypedDict):
    company_name: str
    user_query: str
    plan: Optional[TaskPlan]
    retrieved_docs: List[Dict[str, Any]]
    financial_results: List[FinancialMetricResult]
    citations: List[Citation]
    draft_report: str
    faithfulness_score: float
    current_step: str
    error: Optional[str]

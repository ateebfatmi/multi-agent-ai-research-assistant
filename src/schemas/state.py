from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel, Field

class SubTask(BaseModel):
    id: str = Field(description="Unique identifier for the subtask, e.g., 'task_1'")
    topic: str = Field(description="Specific research question or angle to explore")
    source_type: str = Field(description="Source type: 'web' or 'arxiv'")

class ResearchPlan(BaseModel):
    tasks: List[SubTask] = Field(description="List of 2 to 4 decomposed research subtasks")

class SourceItem(BaseModel):
    id: str
    title: str
    url: str
    content: str
    source_type: str  # 'web' or 'arxiv'
    relevance_score: Optional[float] = 1.0

class CriticEvaluation(BaseModel):
    is_sufficient: bool = Field(description="Whether the gathered evidence is sufficient, reliable, and consistent to answer the research topic comprehensively.")
    critique: str = Field(description="Detailed feedback explaining gaps, inconsistencies, or unverified claims if is_sufficient is False.")
    refinement_queries: List[str] = Field(default_factory=list, description="Follow-up search queries if more research is required.")

class AgentState(TypedDict):
    query: str
    intent: Optional[str]
    intent_reasoning: Optional[str]
    plan: List[Dict[str, Any]]
    raw_findings: List[Dict[str, Any]]
    verified_findings: List[Dict[str, Any]]
    critic_evaluation: Optional[Dict[str, Any]]
    retry_count: int
    final_report: str
    citations: List[Dict[str, Any]]

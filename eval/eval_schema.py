from pydantic import BaseModel, Field

class FactualityScore(BaseModel):
    groundedness_score: float = Field(description="Score between 0.0 and 1.0 indicating whether claims in the report are supported by evidence")
    relevance_score: float = Field(description="Score between 0.0 and 1.0 indicating topic relevance")
    citation_accuracy_score: float = Field(description="Score between 0.0 and 1.0 for valid link references")
    verdict: str = Field(description="Brief assessment of the report quality and hallucination risk")

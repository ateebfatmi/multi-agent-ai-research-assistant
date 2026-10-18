from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.agents.llm_factory import get_llm
from src.schemas.state import AgentState, CriticEvaluation

CRITIC_PROMPT = """You are the Senior Fact-Checker & Verification Agent in an agentic research team.
Your objective is to evaluate whether the gathered research evidence is sufficient, factually sound, and relevant to answer the user query.

User Query:
{query}

Subtasks Planned:
{plan}

Raw Evidence Gathered:
{findings}

Current Retry Count: {retry_count}

Evaluation Criteria:
1. Are the key aspects of the user query covered by the sources?
2. Are the findings consistent, or do they contain obvious contradictions/low-quality noise?
3. If this is already retry 2 or higher, prioritize marking is_sufficient = True to avoid infinite looping, unless the findings are completely empty.

If evidence is insufficient or misses critical angles, set is_sufficient = False, provide constructive critique, and suggest 1 to 2 targeted refinement_queries for re-search.
Otherwise, set is_sufficient = True.
"""

def critic_node(state: AgentState) -> Dict[str, Any]:
    """Critic Agent: Evaluates factual completeness and decides whether to re-research."""
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(CriticEvaluation)
    
    prompt = ChatPromptTemplate.from_template(CRITIC_PROMPT)
    chain = prompt | structured_llm
    
    retry_count = state.get("retry_count", 0)
    
    # Format evidence for prompt
    findings_str = "\n".join([
        f"- [{item.get('source_type', 'web').upper()}] {item.get('title')}: {item.get('content')[:350]}..."
        for item in state.get("raw_findings", [])
    ])
    
    evaluation: CriticEvaluation = chain.invoke({
        "query": state["query"],
        "plan": str(state.get("plan", [])),
        "findings": findings_str if findings_str else "No evidence found.",
        "retry_count": retry_count
    })
    
    verified = state.get("raw_findings", []) if evaluation.is_sufficient else state.get("verified_findings", [])
    
    return {
        "critic_evaluation": evaluation.model_dump(),
        "verified_findings": verified,
        "retry_count": retry_count + 1
    }

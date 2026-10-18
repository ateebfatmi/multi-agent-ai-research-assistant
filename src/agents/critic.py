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
1. Coverage: Are core technical aspects, mechanisms, and real-world implications adequately backed by the sources?
2. Quality & Factuality: Are the excerpts substantive, authoritative, and consistent, or are they mere surface-level noise or error messages?
3. Termination Guard: If retry_count >= 2, mark is_sufficient = True to prevent infinite loops, unless absolutely zero usable evidence was found.

Decision:
- If evidence has major gaps or misses critical angles, set is_sufficient = False, explain the precise missing gap in 'critique', and formulate 1-2 targeted 'refinement_queries'.
- Otherwise, set is_sufficient = True with constructive synthesis guidance.
"""

def critic_node(state: AgentState) -> Dict[str, Any]:
    """Critic Agent: Evaluates factual completeness, filters valid evidence, and controls re-search loops."""
    llm = get_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(CriticEvaluation)
    
    prompt = ChatPromptTemplate.from_template(CRITIC_PROMPT)
    chain = prompt | structured_llm
    
    retry_count = state.get("retry_count", 0)
    raw_findings = state.get("raw_findings", [])
    
    # Filter out error entries or uninformative blanks
    valid_findings = [
        item for item in raw_findings
        if not item.get("id", "").endswith("_error")
        and not item.get("id", "").endswith("missing_key")
        and len(item.get("content", "").strip()) > 30
    ]
    
    findings_str = "\n".join([
        f"- [{item.get('source_type', 'web').upper()}] {item.get('title')}: {item.get('content')[:350]}..."
        for item in valid_findings
    ])
    
    evaluation: CriticEvaluation = chain.invoke({
        "query": state["query"],
        "plan": str(state.get("plan", [])),
        "findings": findings_str if findings_str else "No valid evidence retrieved.",
        "retry_count": retry_count
    })
    
    # Store verified findings when sufficient, or pass accumulated valid findings
    return {
        "critic_evaluation": evaluation.model_dump(),
        "verified_findings": valid_findings if valid_findings else raw_findings,
        "retry_count": retry_count + 1
    }

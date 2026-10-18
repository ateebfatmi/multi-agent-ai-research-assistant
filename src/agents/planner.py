from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.agents.llm_factory import get_llm
from src.schemas.state import AgentState, ResearchPlan

PLANNER_PROMPT = """You are the Lead Research Orchestrator & Planner in a multi-agent academic research team.
Your goal is to break down the user's research topic into 2 to 4 well-scoped, distinct subtasks.

Rules:
1. Break down the query logically (e.g., background & concepts, current methods/advances, challenges/future outlook).
2. For each subtask, decide if it requires general 'web' search or 'arxiv' academic paper search.
3. Keep topics concise, clear, and targeted.

User Research Topic:
{query}
"""

def planner_node(state: AgentState) -> Dict[str, Any]:
    """Planner Agent: Decomposes open-ended query into subtasks."""
    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(ResearchPlan)
    
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    chain = prompt | structured_llm
    
    plan: ResearchPlan = chain.invoke({"query": state["query"]})
    
    return {
        "plan": [task.model_dump() for task in plan.tasks],
        "retry_count": state.get("retry_count", 0)
    }

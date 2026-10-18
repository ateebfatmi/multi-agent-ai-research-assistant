from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from src.agents.llm_factory import get_llm
from src.schemas.state import AgentState, ResearchPlan

PLANNER_PROMPT = """You are the Senior Research Architect & Lead Planner in an autonomous academic research team.
Your objective is to decompose the user's research topic into 2 to 4 highly focused, non-overlapping investigation tracks.

Strategic Planning Guidelines:
1. Multidisciplinary Breakdown:
   - Track 1 (Foundations & Architecture): Core mathematical/theoretical principles, algorithmic mechanisms.
   - Track 2 (Empirical Benchmarks & SOTA): Leading models, quantitative performance metrics, and comparative evaluation.
   - Track 3 (Open Bottlenecks & Practical Realities): Trade-offs (latency, memory, safety), hardware limitations, and future research vectors.
2. Source Alignment:
   - Assign 'arxiv' for technical/theoretical rigor, novel architectures, formal proofs, and academic papers. Use concise, high-yield academic keywords without boolean punctuation noise.
   - Assign 'web' for recent industry deployments, hardware benchmarks, open-source code releases, and current ecosystem trends.
3. Clarity: Ensure each subtask topic is an optimized, high-yield search query.

Research Topic:
{query}
"""

def planner_node(state: AgentState) -> Dict[str, Any]:
    """Planner Agent: Decomposes topic into strategic academic & web subtasks."""
    llm = get_llm(temperature=0.2)
    structured_llm = llm.with_structured_output(ResearchPlan)
    
    prompt = ChatPromptTemplate.from_template(PLANNER_PROMPT)
    chain = prompt | structured_llm
    
    plan: ResearchPlan = chain.invoke({"query": state["query"]})
    
    return {
        "plan": [task.model_dump() for task in plan.tasks],
        "retry_count": state.get("retry_count", 0)
    }

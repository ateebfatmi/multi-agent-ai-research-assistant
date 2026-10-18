from langgraph.graph import StateGraph, END
from src.schemas.state import AgentState
from src.agents.router import router_node
from src.agents.planner import planner_node
from src.agents.researcher import researcher_node
from src.agents.critic import critic_node
from src.agents.synthesis import synthesis_node

def route_intent(state: AgentState) -> str:
    """Conditional edge after router: conversational / direct QA skips research pipeline."""
    intent = state.get("intent", "research")
    if intent in ("conversational", "direct_qa"):
        return END
    return "planner"

def decide_next_step(state: AgentState) -> str:
    """Conditional edge: check if Critic approved findings or if retry limit reached."""
    critic_eval = state.get("critic_evaluation")
    retry_count = state.get("retry_count", 0)
    
    # If critic says it's sufficient or we reached retry limit (max 2 retries), proceed to synthesis
    if not critic_eval or critic_eval.get("is_sufficient", True) or retry_count >= 2:
        return "synthesis"
    
    # Otherwise, loop back to researcher with refinement queries
    return "researcher"

def create_research_graph():
    """Builds and compiles the multi-agent research LangGraph."""
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("synthesis", synthesis_node)
    
    # Define Graph Edges
    workflow.set_entry_point("router")
    
    workflow.add_conditional_edges(
        "router",
        route_intent,
        {
            "planner": "planner",
            END: END
        }
    )
    
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "critic")
    
    # Conditional Transition from Critic
    workflow.add_conditional_edges(
        "critic",
        decide_next_step,
        {
            "researcher": "researcher",
            "synthesis": "synthesis"
        }
    )
    
    workflow.add_edge("synthesis", END)
    
    return workflow.compile()


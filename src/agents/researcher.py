import concurrent.futures
from typing import Dict, Any, List
from src.schemas.state import AgentState
from src.tools.search_tools import search_tavily, search_arxiv

def fetch_subtask_findings(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Helper worker to fetch research concurrently."""
    topic = task.get("topic", "")
    source_type = task.get("source_type", "web")
    
    if source_type == "arxiv":
        return search_arxiv(topic, max_results=2)
    else:
        return search_tavily(topic, max_results=3)

def researcher_node(state: AgentState) -> Dict[str, Any]:
    """Researcher Agent: Concurrent parallel search across web and arXiv."""
    plan = state.get("plan", [])
    critic_eval = state.get("critic_evaluation")
    
    findings: List[Dict[str, Any]] = list(state.get("raw_findings", []))
    
    # If triggered by critic feedback loop, use targeted refinement queries
    if critic_eval and critic_eval.get("refinement_queries"):
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(search_tavily, q, 2) for q in critic_eval["refinement_queries"]]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    findings.extend(res)
                except Exception as e:
                    print(f"Refinement query error: {e}")
        return {"raw_findings": findings}

    # Standard run: Execute subtasks concurrently using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan) or 1) as executor:
        future_to_task = {executor.submit(fetch_subtask_findings, task): task for task in plan}
        for future in concurrent.futures.as_completed(future_to_task):
            try:
                results = future.result()
                findings.extend(results)
            except Exception as e:
                task = future_to_task[future]
                print(f"Error executing subtask '{task.get('topic')}': {e}")
                
    return {"raw_findings": findings}

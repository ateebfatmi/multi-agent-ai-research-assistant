import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from src.graph import create_research_graph

load_dotenv()

TEST_QUERIES = [
    "What are the trade-offs of Test-Time Compute vs Pre-training scaling in LLMs?",
    "Recent breakthroughs in Multi-Modal Agentic Systems for medical diagnostics",
    "Explain FlashAttention-3 architecture improvements"
]

def run_baseline_single_prompt(query: str, llm):
    """Baseline: Single-call direct LLM answer without search or multi-agent verification."""
    prompt = f"Write an academic research report answering this query: {query}. Include citations where possible."
    res = llm.invoke(prompt)
    if isinstance(res.content, list):
        text_parts = [part.get("text", "") if isinstance(part, dict) else str(part) for part in res.content]
        return "\n".join(text_parts)
    return str(res.content)

def run_benchmark():
    print("=== Multi-Agent Research Assistant vs Single-Prompt Baseline Evaluation ===\n")
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0.2, google_api_key=os.getenv("GEMINI_API_KEY"))
    graph = create_research_graph()
    
    results = []
    for query in TEST_QUERIES:
        print(f"\n[Evaluating Query]: {query}")
        
        # 1. Baseline
        print("-> Running Baseline...")
        baseline_out = run_baseline_single_prompt(query, llm)
        
        # 2. Multi-Agent
        print("-> Running Multi-Agent Graph...")
        init_state = {
            "query": query,
            "plan": [],
            "raw_findings": [],
            "verified_findings": [],
            "critic_evaluation": None,
            "retry_count": 0,
            "final_report": "",
            "citations": []
        }
        agent_res = graph.invoke(init_state)
        
        results.append({
            "query": query,
            "baseline_word_count": len(baseline_out.split()),
            "baseline_has_traceable_urls": "http" in baseline_out,
            "agent_sources_count": len(agent_res.get("citations", [])),
            "agent_retry_count": agent_res.get("retry_count", 0),
            "agent_word_count": len(agent_res.get("final_report", "").split()),
        })
    
    print("\n=== Benchmark Summary ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    run_benchmark()

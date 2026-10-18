# Viva Voce Preparation & Project Defense Guide
**Project:** Multi-Agent Research Assistant (Agentic AI System)  
**Frameworks:** LangGraph, Google Gemini (`gemini-3.8-flash`), Tavily API, arXiv API, Chainlit  

---

## 1. Core Architecture Questions

### Q1: Why use LangGraph instead of traditional LangChain Chains or standard LLM prompts?
**Answer:**
Traditional linear chains (DAGs) cannot easily support cyclical reasoning loops. Research mimics human iterative refinement:
1. When humans find an evidence gap, they don't give up; they re-query.
2. In LangGraph, the workflow is modeled as a cyclic state machine where nodes represent specialized agents and edges represent conditional transitions.
3. The **Critic / Verifier** agent conditionally evaluates findings: if insufficient or contradictory, it returns execution back to the **Researcher** node with targeted refinement queries (`critic -> researcher -> critic -> synthesis`).

### Q2: How is Shared State and Memory managed between the agents?
**Answer:**
We use a centralized TypedDict state schema (`AgentState`) containing:
- `query`: The initial user research question.
- `plan`: The decomposed subtasks from the Planner agent.
- `raw_findings`: Accumulated literature and web evidence tagged with unique source IDs and metadata.
- `critic_evaluation`: Verification verdict, critique text, and targeted follow-up queries.
- `retry_count`: Loop guard to avoid infinite state cycles (capped at 2 loops).
- `final_report` & `citations`: Synthesized output with indexed bibliographic sources.

---

## 2. Agent Roles Breakdown

| Agent | Responsibility | Implementation Node |
| :--- | :--- | :--- |
| **Planner / Orchestrator** | Breaks open-ended questions into 2–4 targeted subtasks, tagging each as `web` or `arxiv`. | `planner_node` |
| **Parallel Researcher** | Dispatches asynchronous threads to fetch from Tavily (web) and arXiv (academic papers) simultaneously. | `researcher_node` |
| **Critic / Verifier** | Inspects raw findings for hallucination, conflicting data, and query completeness. | `critic_node` |
| **Synthesis / Writer** | Merges verified insights into an academic format with inline bracket citations `[1]`, `[2]`. | `synthesis_node` |

---

## 3. Evaluation & Measurable Improvement (Synopsis Objective)

### Q3: How do you prove this multi-agent system is better than a standard LLM?
**Answer:**
We provide an automated evaluation harness in `eval/benchmark.py`:
1. **Traceability:** Standard single-prompt LLMs hallucinate URLs or rely on dated training data. Our system provides direct clickable links to live arXiv pre-prints and verified web sources.
2. **Factual Completeness:** The parallel decomposition ensures multiple independent facets of a topic are addressed rather than a shallow summary.
3. **Hallucination Reduction:** The explicit verification phase between research and synthesis filters out noisy, irrelevant, or ungrounded data.

---

## 4. Key Performance Optimizations
- **Parallel Search Execution:** Subtasks are fetched concurrently using `ThreadPoolExecutor`, reducing total search latency from $N \times T$ to $\max(T)$.
- **PDF Export Engine:** Automatically parses Markdown into an academic-formatted PDF report using `ReportLab` so users can download findings with one click.

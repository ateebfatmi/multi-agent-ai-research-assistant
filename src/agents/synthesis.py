from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from src.agents.llm_factory import get_llm
from src.schemas.state import AgentState

SYNTHESIS_PROMPT = """You are the Principal Scientific Writer & Synthesis Lead in an autonomous research team.
Your objective is to author an exhaustive, mathematically and architecturally rigorous research report answering the user's research topic.
Your synthesis must rely strictly on the verified literature and findings provided below.

User Research Topic:
{query}

Verified Findings & Literature:
{findings}

Mandatory Report Structure:
1. Executive Summary: Core thesis, breakthrough implications, and high-level synthesis of findings.
2. Theoretical Foundations & Algorithmic Mechanics: In-depth exploration of core principles, architectures, and mechanisms.
3. Comparative Analysis & Empirical Benchmarks: Quantitative comparisons, empirical metrics, strengths vs weaknesses of competing approaches.
4. Critical Bottlenecks & Open Questions: Computational overhead, failure modes, safety/alignment concerns, and unresolved theoretical barriers.
5. Strategic Outlook & Future Directions: Roadmap for the next 2-5 years.

Strict Academic Guidelines:
- Write in dense, clear, publication-grade academic prose with clear markdown headings (`##`, `###`), tables or bullet points where appropriate.
- Every major technical or empirical claim MUST be explicitly anchored to its inline source citation: e.g., `[1]`, `[2]`.
- Under NO circumstance hallucinate URLs or citations not present in the verified literature.
- If findings have limitations or gaps, acknowledge them honestly under the critical analysis section.
"""

def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """Synthesis Agent: Synthesizes verified findings into a cited, comprehensive markdown report."""
    llm = get_llm(temperature=0.3)
    
    # Prefer verified_findings, fallback to raw_findings
    raw = state.get("verified_findings") or state.get("raw_findings", [])
    
    # Filter out error records
    findings = [
        item for item in raw
        if not item.get("id", "").endswith("_error")
        and not item.get("id", "").endswith("missing_key")
        and len(item.get("content", "").strip()) > 20
    ]
    
    # Format numbered sources
    sources_text = []
    citations = []
    seen_urls = set()
    
    idx = 1
    for item in findings:
        url = item.get("url", "").strip()
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
            
        title = item.get("title", f"Source {idx}").strip()
        content = item.get("content", "").strip()
        source_type = item.get("source_type", "web")
        
        sources_text.append(f"[{idx}] ({source_type.upper()}) {title}\nURL: {url}\nExcerpt: {content}\n")
        citations.append({"index": idx, "title": title, "url": url or "#", "source_type": source_type})
        idx += 1
        
    prompt = ChatPromptTemplate.from_template(SYNTHESIS_PROMPT)
    chain = prompt | llm
    
    response = chain.invoke({
        "query": state["query"],
        "findings": "\n---\n".join(sources_text) if sources_text else "No external findings available."
    })
    
    if isinstance(response.content, list):
        final_text = "\n".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content])
    else:
        final_text = str(response.content)
    
    return {
        "final_report": final_text,
        "citations": citations
    }

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate
from src.agents.llm_factory import get_llm
from src.schemas.state import AgentState

SYNTHESIS_PROMPT = """You are the Lead Scientific & Technical Writer in a research team.
Your objective is to write a comprehensive, rigorous, well-structured research report answering the user's research query based exclusively on verified findings.

User Query:
{query}

Verified Findings & Literature:
{findings}

Requirements for the Report:
1. Executive Summary: High-level synthesis of key discoveries.
2. Background & Theoretical Foundations: Context and primary concepts.
3. Key Technical Insights & Architecture: In-depth technical breakdown.
4. Open Challenges & Future Directions: Critical analysis of unresolved limitations.
5. References & Source Traceability: List every referenced source with its title and link matching the inline numbers [1], [2], etc.

Style Guidelines:
- Use formal academic markdown format with bold headings and bullet points.
- Anchor all key factual assertions to inline citations like [1], [2].
- DO NOT make up claims that are not supported by the findings.
"""

def synthesis_node(state: AgentState) -> Dict[str, Any]:
    """Synthesis Agent: Synthesizes verified findings into a cited, comprehensive markdown report."""
    llm = get_llm(temperature=0.3)
    
    findings = state.get("raw_findings", [])
    
    # Format numbered sources
    sources_text = []
    citations = []
    for i, item in enumerate(findings):
        idx = i + 1
        title = item.get("title", f"Source {idx}")
        url = item.get("url", "#")
        content = item.get("content", "")
        source_type = item.get("source_type", "web")
        sources_text.append(f"[{idx}] ({source_type.upper()}) {title}\nURL: {url}\nExcerpt: {content}\n")
        citations.append({"index": idx, "title": title, "url": url, "source_type": source_type})
        
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

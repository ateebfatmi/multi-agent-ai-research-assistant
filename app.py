import os
import tempfile
from typing import List
import chainlit as cl
from dotenv import load_dotenv
from src.graph import create_research_graph
from src.utils.exporter import export_report_to_pdf

load_dotenv()

# Ensure Chainlit local storage directory exists to prevent FileNotFoundError on file uploads/exports
os.makedirs(".files", exist_ok=True)

# Precompile the LangGraph workflow
research_graph = create_research_graph()

@cl.set_starters
async def set_starters() -> List[cl.Starter]:
    """Provides modern, clickable research topic cards on the landing screen."""
    return [
        cl.Starter(
            label="⚡ FlashAttention-3 Architecture",
            message="Explain FlashAttention-3 architectural improvements, hardware asynchrony, and FP8 acceleration.",
        ),
        cl.Starter(
            label="🧠 Test-Time Compute Scaling",
            message="What are the trade-offs of Test-Time Compute vs Pre-training scaling in Large Language Models?",
        ),
        cl.Starter(
            label="🧬 CRISPR-Cas9 Prime Editing",
            message="Recent breakthroughs in CRISPR-Cas9 prime editing mechanisms and clinical delivery systems.",
        ),
        cl.Starter(
            label="🤖 Humanoid VLA Policies",
            message="State of the art in end-to-end vision-language-action policies for humanoid robotics.",
        )
    ]

@cl.on_chat_start
async def on_chat_start():
    welcome_banner = (
        "### 🔬 Multi-Agent Autonomous Research Assistant\n"
        "*Minor Project II — Powered by LangGraph, Google Gemini, arXiv & Tavily*\n\n"
        "<div style='display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px;'>"
        "<span style='background: rgba(139, 92, 246, 0.15); color: #8b5cf6; padding: 4px 10px; border-radius: 9999px; font-size: 0.82rem; font-weight: 600;'>🧠 Intent Triage</span>"
        "<span style='background: rgba(6, 182, 212, 0.15); color: #06b6d4; padding: 4px 10px; border-radius: 9999px; font-size: 0.82rem; font-weight: 600;'>🧭 Lead Planner</span>"
        "<span style='background: rgba(16, 185, 129, 0.15); color: #10b981; padding: 4px 10px; border-radius: 9999px; font-size: 0.82rem; font-weight: 600;'>🔎 Parallel Researcher</span>"
        "<span style='background: rgba(245, 158, 11, 0.15); color: #f59e0b; padding: 4px 10px; border-radius: 9999px; font-size: 0.82rem; font-weight: 600;'>⚖️ Verification Critic</span>"
        "<span style='background: rgba(244, 63, 94, 0.15); color: #f43f5e; padding: 4px 10px; border-radius: 9999px; font-size: 0.82rem; font-weight: 600;'>✍️ Synthesis Writer</span>"
        "</div>\n\n"
        "Ask any complex technical question to initiate autonomous multi-source research, or select one of the prompt starters below."
    )
    await cl.Message(content=welcome_banner).send()

@cl.on_message
async def on_message(message: cl.Message):
    user_query = message.content.strip()
    await execute_workflow(user_query)

async def execute_workflow(user_query: str):
    initial_state = {
        "query": user_query,
        "intent": "research",
        "intent_reasoning": "",
        "plan": [],
        "raw_findings": [],
        "verified_findings": [],
        "critic_evaluation": None,
        "retry_count": 0,
        "final_report": "",
        "citations": []
    }
    
    current_state = initial_state
    
    # Real-time multi-agent execution visualizer
    async with cl.Step(name="🚀 Autonomous Multi-Agent System") as root_step:
        for event in research_graph.stream(initial_state):
            for node_name, node_output in event.items():
                current_state.update(node_output)
                
                if node_name == "router":
                    intent = current_state.get("intent", "research")
                    reasoning = current_state.get("intent_reasoning", "")
                    
                    if intent in ("conversational", "direct_qa"):
                        root_step.name = "💬 Conversational Response"
                        async with cl.Step(name="🧠 Intent Triage: Direct Conversation", parent_id=root_step.id) as step:
                            step.output = (
                                f"**Classification:** `{intent.upper()}`\n\n"
                                f"**Analysis:** {reasoning}"
                            )
                    else:
                        async with cl.Step(name="🧠 0. Intent Triage & Scoping Agent", parent_id=root_step.id) as step:
                            scoped = current_state.get("query", user_query)
                            step.output = (
                                f"**Status:** ✅ Confirmed Research Topic\n\n"
                                f"**Analysis:** {reasoning}\n\n"
                                f"**Scoped Problem:** `{scoped}`"
                            )

                elif node_name == "planner":
                    async with cl.Step(name="🧭 1. Orchestrator / Planner Agent", parent_id=root_step.id) as step:
                        tasks = current_state.get("plan", [])
                        plan_md = "\n".join([
                            f"- **`[{t.get('source_type', 'web').upper()}]`** {t.get('topic')}"
                            for t in tasks
                        ])
                        step.output = f"Decomposed inquiry into **{len(tasks)} targeted investigation tracks**:\n\n{plan_md}"
                
                elif node_name == "researcher":
                    async with cl.Step(name="🔎 2. Parallel Researcher Agent", parent_id=root_step.id) as step:
                        findings = current_state.get("raw_findings", [])
                        arxiv_count = sum(1 for f in findings if f.get('source_type') == 'arxiv')
                        web_count = sum(1 for f in findings if f.get('source_type') == 'web')
                        step.output = (
                            f"Concurrently queried academic and live web repositories:\n\n"
                            f"- 📄 **arXiv Pre-prints:** `{arxiv_count}` papers retrieved\n"
                            f"- 🌐 **Tavily Live Web:** `{web_count}` sources retrieved\n"
                            f"- 📊 **Total Evidence Pool:** `{len(findings)}` citations collected"
                        )
                
                elif node_name == "critic":
                    async with cl.Step(name="⚖️ 3. Verification & Critic Agent", parent_id=root_step.id) as step:
                        critique = current_state.get("critic_evaluation", {})
                        passed = critique.get("is_sufficient", True)
                        status_badge = (
                            "✅ **Factual Consistency & Coverage Check Passed**"
                            if passed else
                            "🔄 **Critic Loop Activated: Additional evidence required — Triggering targeted re-search**"
                        )
                        feedback = critique.get('critique', 'Findings verified and consistent across sources.')
                        refinements = critique.get('refinement_queries', [])
                        
                        out_text = f"**Verification Status:**\n{status_badge}\n\n**Critic Analysis:**\n{feedback}"
                        if refinements:
                            out_text += "\n\n**Directed Follow-up Searches:**\n" + "\n".join([f"- `{q}`" for q in refinements])
                        step.output = out_text
                
                elif node_name == "synthesis":
                    async with cl.Step(name="✍️ 4. Synthesis & Technical Writer Agent", parent_id=root_step.id) as step:
                        step.output = "Authored publication-grade academic report with numbered citations and comprehensive bibliography."

    intent = current_state.get("intent", "research")
    final_report = current_state.get("final_report", "Research completed, but response was empty.")
    
    # Casual small talk or direct QA doesn't need bibliography or PDF export
    if intent in ("conversational", "direct_qa"):
        await cl.Message(content=final_report).send()
        return

    # Append formatted traceable bibliography
    citations = current_state.get("citations", [])
    if citations:
        sources_md = "\n\n---\n### 📚 Traceable Sources & References\n"
        for c in citations:
            url = c['url']
            title = c['title']
            stype = c['source_type'].upper()
            sources_md += f"- **[{c['index']}]** [{title}]({url}) `({stype})`\n"
        final_report += sources_md

    # Generate PDF export
    elements = []
    try:
        temp_dir = tempfile.gettempdir()
        safe_name = "".join(x for x in user_query[:30] if x.isalnum() or x in " _-").strip()
        pdf_path = os.path.join(temp_dir, f"Research_Report_{safe_name}.pdf")
        export_report_to_pdf(final_report, pdf_path)
        elements.append(
            cl.File(
                name=f"Research_Report_{safe_name}.pdf",
                path=pdf_path,
                display="inline"
            )
        )
    except Exception as e:
        print(f"Error creating PDF: {e}")

    await cl.Message(content=final_report, elements=elements).send()

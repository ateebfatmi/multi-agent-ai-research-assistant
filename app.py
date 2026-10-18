import os
import tempfile
import chainlit as cl
from dotenv import load_dotenv
from src.graph import create_research_graph
from src.utils.exporter import export_report_to_pdf

load_dotenv()

# Precompile the LangGraph workflow
research_graph = create_research_graph()

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(
        content=(
            "### 🔬 Multi-Agent Research Assistant (Agentic AI System)\n"
            "*Minor Project II — Powered by LangGraph & Google Gemini*\n\n"
            "**Architecture:**\n"
            "- 🧠 **Intent Triage & Router Agent:** Intelligently screens queries, handles conversational banter/chit-chat directly, and scopes academic research tasks.\n"
            "- 🧭 **Planner Agent:** Decomposes complex research topics into targeted sub-investigations.\n"
            "- 🔎 **Researcher Agent:** Concurrently queries **Tavily (Web)** and **arXiv (Academic)**.\n"
            "- ⚖️ **Critic / Verifier:** Validates factual claims, checks source reliability & triggers re-search loops.\n"
            "- ✍️ **Synthesis Agent:** Drafts an academic report with inline traceable citations.\n\n"
            "💡 *Enter any research topic below to begin autonomous research, or ask a question!*"
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    user_query = message.content.strip()
    
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
    
    # Visualizing Multi-Agent Execution Steps in Real-Time
    async with cl.Step(name="🚀 Autonomous Multi-Agent System") as root_step:
        for event in research_graph.stream(initial_state):
            for node_name, node_output in event.items():
                current_state.update(node_output)
                
                if node_name == "router":
                    intent = current_state.get("intent", "research")
                    reasoning = current_state.get("intent_reasoning", "")
                    if intent in ("conversational", "direct_qa"):
                        root_step.name = "💬 Conversational Agent"
                        async with cl.Step(name="🧠 Intent Triage: Direct Conversation", parent_id=root_step.id) as step:
                            step.output = f"**Classification:** `{intent.upper()}`\n**Analysis:** {reasoning}"
                    else:
                        async with cl.Step(name="🧠 0. Intent Triage & Scoping Agent", parent_id=root_step.id) as step:
                            refined = current_state.get("query", user_query)
                            step.output = f"**Status:** Confirmed Academic/Technical Research\n**Analysis:** {reasoning}\n**Scoped Topic:** `{refined}`"

                elif node_name == "planner":
                    async with cl.Step(name="🧭 1. Orchestrator / Planner Agent", parent_id=root_step.id) as step:
                        tasks = current_state.get("plan", [])
                        plan_md = "\n".join([f"- **[{t.get('source_type', 'web').upper()}]** {t.get('topic')}" for t in tasks])
                        step.output = f"Decomposed query into **{len(tasks)} parallel subtasks**:\n\n{plan_md}"
                
                elif node_name == "researcher":
                    async with cl.Step(name="🔎 2. Parallel Researcher Agent", parent_id=root_step.id) as step:
                        findings = current_state.get("raw_findings", [])
                        arxiv_count = sum(1 for f in findings if f.get('source_type') == 'arxiv')
                        web_count = sum(1 for f in findings if f.get('source_type') == 'web')
                        step.output = f"Concurrently fetched **{len(findings)} references**:\n- 📄 arXiv Papers: {arxiv_count}\n- 🌐 Web Sources: {web_count}"
                
                elif node_name == "critic":
                    async with cl.Step(name="⚖️ 3. Verification & Critic Agent", parent_id=root_step.id) as step:
                        critique = current_state.get("critic_evaluation", {})
                        passed = critique.get("is_sufficient", True)
                        status_badge = "✅ **Passed Credibility & Consistency Check**" if passed else "🔄 **Critic Loop: Insufficient evidence — Triggering targeted re-search**"
                        feedback = critique.get('critique', 'Findings verified and consistent.')
                        refinements = critique.get('refinement_queries', [])
                        
                        out_text = f"**Verification Status:** {status_badge}\n\n**Critic Note:** {feedback}"
                        if refinements:
                            out_text += f"\n\n**Targeted Follow-up Queries:**\n" + "\n".join([f"- `{q}`" for q in refinements])
                        step.output = out_text
                
                elif node_name == "synthesis":
                    async with cl.Step(name="✍️ 4. Synthesis & Technical Writer Agent", parent_id=root_step.id) as step:
                        step.output = "Drafted comprehensive report with numbered citations `[1]`, `[2]` and bibliography."

    intent = current_state.get("intent", "research")
    final_report = current_state.get("final_report", "Research completed, but response was empty.")
    
    # If the user engaged in small talk, banter, or a direct question, send clean direct response
    if intent in ("conversational", "direct_qa"):
        await cl.Message(content=final_report).send()
        return

    # Final Report Assembly with Formatted Citations Section for Research Queries
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
        pdf_path = os.path.join(temp_dir, f"Research_Report_{os.getpid()}.pdf")
        export_report_to_pdf(final_report, pdf_path)
        elements.append(
            cl.File(
                name="Research_Report.pdf",
                path=pdf_path,
                display="inline"
            )
        )
    except Exception as e:
        print(f"Error creating PDF: {e}")

    await cl.Message(content=final_report, elements=elements).send()

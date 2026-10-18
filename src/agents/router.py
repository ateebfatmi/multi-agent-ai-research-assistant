from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.agents.llm_factory import get_llm
from src.schemas.state import AgentState

class IntentClassification(BaseModel):
    intent: str = Field(
        description=(
            "Must be one of: 'research', 'conversational', or 'direct_qa'. "
            "- 'research': Deep technical, academic, scientific, factual, or exploratory topics that warrant web search and academic literature review. "
            "- 'conversational': Greetings (hi/hello), chit-chat, jokes, banter, insults, slang (e.g. Hindi/Hinglish slang like 'tu chutiya hai', 'kya haal hai', etc.), questions about the bot itself, or playful remarks. "
            "- 'direct_qa': Very simple standalone factual/math/conversion questions (e.g. 'what is 15 * 8?', 'capital of France') that require a direct 1-sentence answer without launching an academic research pipeline."
        )
    )
    reasoning: str = Field(
        description="Brief explanation of why this intent was identified."
    )
    direct_reply: Optional[str] = Field(
        default="",
        description=(
            "Required if intent is 'conversational' or 'direct_qa'. "
            "Craft an intelligent, witty, charming, and contextually aware response. "
            "If the user is using Hindi/Hinglish banter or slang (even insults like 'tu chutiya hai'), reply with wit, humor, and charm in Hinglish, without taking offense, and playfully nudge them to give a real research topic. "
            "Never generate a scientific report or academic analysis for insults or small talk."
        )
    )
    refined_topic: Optional[str] = Field(
        default="",
        description="If intent is 'research', provide an academically precise, scoped version of the research topic."
    )

ROUTER_PROMPT = """You are the Chief Intelligence & Intent Classifier for an autonomous multi-agent research system.
Your job is to analyze the incoming user message and determine the optimal handling strategy before any heavy research tools are invoked.

System Context:
This system is an Agentic AI Research Assistant designed for deep academic and technical investigations.
However, human users often test the system with:
1. Informal greetings ("hi", "hey", "wassup")
2. Banter, teasing, or colloquial insults in English, Hindi, Hinglish, etc. (e.g., "tu chutiya hai", "you are stupid", "are you dumb?")
3. Questions about the assistant ("who made you?", "what can you do?")
4. Quick arithmetic or trivia ("what is 2+2?")
5. Genuine complex research topics ("recent breakthroughs in LLM test-time compute", "CRISPR-Cas9 therapeutic applications", etc.)

Guidelines:
- If the user uses slang, curses, or playful insults (such as "tu chutiya hai"):
  * Set intent = 'conversational'.
  * DO NOT conduct an academic investigation or write a scientific thesis on the insult! That looks robotic and absurd.
  * Instead, give a witty, charismatic, culturally authentic response (in Hinglish if prompted in Hinglish/Hindi) with a touch of humor and good vibes, gently steering them to give you a serious topic to research.
- If the user sends a greeting or asks what you do:
  * Set intent = 'conversational'. Introduce yourself as an autonomous multi-agent researcher and invite them to explore complex topics.
- If it is a real research topic:
  * Set intent = 'research'. Formulate a refined, academically clear scope.

User Message:
{query}
"""

def router_node(state: AgentState) -> Dict[str, Any]:
    """Router Agent: Assesses user intent, handles chit-chat/banter directly, or routes to research."""
    llm = get_llm(temperature=0.3)
    structured_llm = llm.with_structured_output(IntentClassification)
    
    prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    chain = prompt | structured_llm
    
    result: IntentClassification = chain.invoke({"query": state["query"]})
    
    intent = result.intent.lower().strip()
    if intent not in ("research", "conversational", "direct_qa"):
        intent = "research"
        
    updates: Dict[str, Any] = {
        "intent": intent,
        "intent_reasoning": result.reasoning
    }
    
    if intent in ("conversational", "direct_qa"):
        updates["final_report"] = result.direct_reply or "Hello! I am your AI Research Assistant. Please provide a research topic to explore."
        updates["citations"] = []
    else:
        if result.refined_topic and len(result.refined_topic.strip()) > 3:
            updates["query"] = result.refined_topic.strip()
            
    return updates

import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(model_name: str = None, temperature: float = 0.2):
    """Factory to get the Google Gemini Chat model with automatic backoff retry."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not model_name:
        model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        max_retries=5
    )

import os
from langchain_google_genai import ChatGoogleGenerativeAI

def get_llm(model_name: str = "gemini-3.8-flash", temperature: float = 0.2):
    """Factory to get the Google Gemini Chat model with automatic backoff retry."""
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
        max_retries=6
    )

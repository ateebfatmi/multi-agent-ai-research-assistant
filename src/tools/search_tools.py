import os
import arxiv
from typing import List, Dict, Any
from tavily import TavilyClient

def search_tavily(query: str, max_results: int = 3) -> List[Dict[str, Any]]:
    """Search the web using Tavily Search API."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return [{
            "id": "tavily_missing_key",
            "title": "Tavily API Key Not Configured",
            "url": "https://tavily.com",
            "content": "Please set TAVILY_API_KEY in your .env file to enable live web search.",
            "source_type": "web"
        }]
    
    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results, search_depth="advanced")
        results = []
        for i, item in enumerate(response.get("results", [])):
            results.append({
                "id": f"web_{i+1}",
                "title": item.get("title", "Web Source"),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "source_type": "web"
            })
        return results
    except Exception as e:
        return [{
            "id": "web_error",
            "title": f"Search Error for: {query}",
            "url": "",
            "content": f"Failed to execute Tavily search: {str(e)}",
            "source_type": "web"
        }]

def search_arxiv(query: str, max_results: int = 2) -> List[Dict[str, Any]]:
    """Search academic research papers using arXiv API."""
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        results = []
        for i, paper in enumerate(client.results(search)):
            results.append({
                "id": f"arxiv_{i+1}",
                "title": paper.title,
                "url": paper.entry_id,
                "content": f"Abstract: {paper.summary}\nAuthors: {', '.join([a.name for a in paper.authors])}\nPublished: {paper.published.strftime('%Y-%m-%d')}",
                "source_type": "arxiv"
            })
        return results
    except Exception as e:
        return [{
            "id": "arxiv_error",
            "title": f"arXiv Search Error for: {query}",
            "url": "",
            "content": f"Failed to search arXiv: {str(e)}",
            "source_type": "arxiv"
        }]

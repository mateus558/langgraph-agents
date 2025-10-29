"""WebSearch agent package for Searx-based web search with LLM summarization.

This package provides a class-based agent that:
1. Categorizes queries using heuristics and LLM
2. Searches via SearxNG with category-aware policies
3. Summarizes results with source whitelisting

Example:
    from cognition.websearch import WebSearchAgent, AgentConfig

    config = AgentConfig(
        searx_host="http://localhost:8080",
        k=5,
        model_name="llama3.1",
    )
    agent = WebSearchAgent(config)

    result = agent.invoke({
        "query": "latest Python news",
        "messages": [],
        "categories": None,
        "results": None,
        "summary": None,
    })
"""

from .agent import WebSearchAgent
from .config import SearchAgentConfig, SearchState
from .constants import ALLOWED_CATEGORIES

__all__ = [
    "WebSearchAgent",
    "SearchAgentConfig",
    "SearchState",
    "ALLOWED_CATEGORIES",
]

__version__ = "1.0.0"

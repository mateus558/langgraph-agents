"""Configuration and state definitions for WebSearch agent.

This module defines the TypedDict state and dataclass configuration
used throughout the WebSearch agent.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from typing_extensions import TypedDict

from core.contracts import AgentConfig

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class SearchState(TypedDict):
    """State for the WebSearch agent graph.

    Attributes:
        query: The search query string.
        categories: List of Searx categories determined for the query.
        results: List of search results from Searx.
        summary: Final LLM-generated summary of the search results.
        lang: Detected or configured language code.
        query_en: English-translated query for categorization.
    """
    query: str
    categories: list[str] | None
    results: list[dict[str, object]] | None
    summary: str | None
    lang: str | None
    query_en: str | None


@dataclass
class SearchAgentConfig(AgentConfig):
    """Configuration for WebSearch agent.

    Attributes:
        searx_host: SearxNG instance URL.
        k: Number of final results to return.
        model_name: Ollama model name for LLM operations.
        base_url: Ollama server URL.
        temperature: LLM temperature for response generation.
        num_ctx: Context window size for the LLM.
        max_categories: Maximum number of categories to use per query.
    lang: Language preference for search results (e.g., "en-US", "en-GB").
        safesearch: Safe search level (0=off, 1=moderate, 2=strict).
        timeout_s: Request timeout in seconds.
        retries: Number of retry attempts for failed Searx requests.
        backoff_base: Base multiplier for exponential backoff.
        engines_allow: Per-category engine allowlist (optional).
        engines_block: Per-category engine blocklist (optional).
    """
    searx_host: str = "http://192.168.30.100:8095"
    k: int = 30
    max_categories: int = 4
    lang: str | None = "en-US"
    safesearch: int = 1
    timeout_s: float = 8.0
    retries: int = 2
    backoff_base: float = 0.6
    engines_allow: dict[str, list[str]] | None = None
    engines_block: dict[str, list[str]] | None = None

    def __post_init__(self):
        # Initialize base model config
        super().__post_init__()
        # Allow overriding via environment variables
        self.searx_host = os.getenv("SEARX_HOST", self.searx_host)
        # Optional numeric overrides
        if (val := os.getenv("SEARCH_K")):
            try:
                self.k = int(val)
            except ValueError:
                pass
        if (val := os.getenv("SEARX_TIMEOUT_S")):
            try:
                self.timeout_s = float(val)
            except ValueError:
                pass

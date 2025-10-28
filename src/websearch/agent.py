"""WebSearch agent implementation.

This module defines the WebSearchAgent class that orchestrates the flow
categorize -> search -> summarize using SearxNG and an LLM.

Key improvements vs. previous version:
- Guaranteed model initialization (when absent in config)
- `with_structured_output` created at call time (not frozen in closure)
- Nodes don't mutate input state; return only deltas
- Search errors don't insert HumanMessage; use clean return and logging
- Union of allow/block engines for all categories
- Extra sanitization for removing URLs outside whitelist
- Replaced prints with logging
"""

from __future__ import annotations

import json
import logging
import re
import time
from json import JSONDecodeError
from typing import Any, cast

from langchain.chat_models import init_chat_model
from langchain_community.utilities import SearxSearchWrapper
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy
from requests import RequestException

from core.contracts import AgentProtocol
from websearch.config import SearchAgentConfig, SearchState
from websearch.constants import ALLOWED_CATEGORIES
from websearch.heuristics import CategoryResponse, heuristic_categories
from websearch.utils import (
    canonical_url,
    dedupe_results,
    diversify_topk,
    normalize_urls,
    pick_time_range,
)

# Optional: load .env here if needed. In production, prefer doing it in the app bootstrap.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Simple logging configuration in case the app has not configured it yet
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class WebSearchAgent(AgentProtocol):
    """WebSearch agent class: categorize -> search -> summarize.

    Pipeline:
      1) Categorize: heuristics + LLM (structured output) to choose Searx categories
      2) Search: query SearxNG with parameters per category and recency policy
      3) Summarize: summarize citing only links from whitelist

    Example usage:
        config = SearchAgentConfig(searx_host="http://localhost:8080", k=5)
        agent = WebSearchAgent(config)

        result = agent.invoke({
            "query": "latest Python news",
            "messages": [],
            "categories": None,
            "results": None,
            "summary": None,
        })
        print(result["summary"])
    """

    def __init__(self, config: SearchAgentConfig | None = None):
        """Initialize the agent.

        Args:
            config: Configuration with model, Searx, and search parameters.
        """
        self.config = config or SearchAgentConfig()
        # Ensure a model exists if it is not provided in the configuration
        if getattr(self.config, "model", None) is None:
            self.config.model = cast(BaseChatModel, self._build_model())
        self.graph = self._build_graph()

    def invoke(self, state: SearchState) -> Any:
        """Invoke the graph with the provided state."""
        return self.graph.invoke(state)

    def get_mermaid(self) -> str:
        """Return a Mermaid diagram of the graph."""
        try:
            return str(self.graph.get_graph().draw_mermaid())
        except Exception:
            # Simple fallback to avoid failures if the API changes
            return "graph TD\n  START --> categorize_query --> web_search --> summarize --> END"

    # -----------------------------
    # BUILDERS
    # -----------------------------
    def _build_model(self):
        """Create the chat model via init_chat_model."""
        # Default provider: use ollama if base_url exists, otherwise openai
        provider = getattr(self.config, "model_provider", None) or ("ollama" if self.config.base_url else "openai")
        kwargs_ctx = {}
        if getattr(self.config, "num_ctx", None) is not None:
            kwargs_ctx["num_ctx"] = self.config.num_ctx

        model = init_chat_model(
            model=self.config.model_name,
            model_provider=provider,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            kwargs=kwargs_ctx or None,
        )
        return model

    def _build_categorize_node(self):
        """Node for categorization (heuristics + LLM with structured output)."""

        SYS_PROMPT = SystemMessage(
            content=(
                "You receive a query and must choose the most relevant Searx categories. "
                f"Allowed set: {', '.join(ALLOWED_CATEGORIES)}. "
                "Reply only with JSON exactly compatible with the requested schema."
            )
        )

        FEWSHOT = [
            HumanMessage(content='QUERY: "breaking microsoft acquisition"\nHINTS: ["news","economics"]'),
            AIMessage(content='{"categories":["news","economics"]}'),
            HumanMessage(content='QUERY: "github actions cache permission denied"\nHINTS: ["news"]'),
            AIMessage(content='{"categories":["it"]}'),
            HumanMessage(content='QUERY: "miscellaneous things without context"\nHINTS: []'),
            AIMessage(content='{"categories":["general"]}'),
            HumanMessage(content='QUERY: "dune 2 movie trailer"\nHINTS: ["videos"]'),
            AIMessage(content='{"categories":["videos"]}'),
        ]

        def llm_categories(query: str, hints: list[str], limit: int) -> list[str]:
            """Get categories via LLM (structured output); fallback to 'general' on error."""
            model_instance = getattr(self.config, "model", None)
            if not model_instance:
                return ["general"]
            try:
                model_struct = model_instance.with_structured_output(CategoryResponse)
                user_msg = HumanMessage(content=f'QUERY: "{query}"\nHINTS: {json.dumps(hints, ensure_ascii=False)}')
                resp = model_struct.invoke([SYS_PROMPT, *FEWSHOT, user_msg])
                cats = (getattr(resp, "categories", None) or [])
                # Sanitize to only allowed categories
                cats = [c for c in cats if c in ALLOWED_CATEGORIES]
                cats = (cats or ["general"])[: max(1, limit)]
                return cats
            except Exception as e:
                logger.debug("Structured categorize failed: %r", e)
                return ["general"]

        def _categorize(state: SearchState) -> dict:
            t0 = time.perf_counter()
            query = state["query"]
            hints = heuristic_categories(query)
            cats = llm_categories(query, hints, self.config.max_categories)
            dt = time.perf_counter() - t0
            logger.info("[websearch] node=categorize_query dt=%.3fs cats=%s", dt, cats)
            return {"categories": cats}

        return _categorize

    def _build_web_search_node(self):
        """Node for SearxNG search, with retry and post-processing."""
        search = SearxSearchWrapper(searx_host=self.config.searx_host)

        def searx_call_with_retry(call):
            for attempt in range(self.config.retries + 1):
                try:
                    return call()
                except (RequestException, JSONDecodeError, ValueError) as e:
                    if attempt >= self.config.retries:
                        raise
                    delay = self.config.backoff_base * (2 ** attempt)
                    logger.warning("[websearch] retry=%d delay=%.2fs error=%s", attempt + 1, delay, type(e).__name__)
                    time.sleep(delay)

        def _search(state: SearchState) -> dict:
            t0 = time.perf_counter()
            q = state["query"].strip()
            cats = state.get("categories") or ["general"]
            if "general" not in cats:
                cats = [*cats, "general"]

            kwargs: dict[str, Any] = {
                "categories": cats,
                "num_results": self.config.k * 2,
                "safesearch": self.config.safesearch,
            }
            if self.config.lang:
                kwargs["language"] = self.config.lang
            if (tr := pick_time_range(cats)):
                kwargs["time_range"] = tr

            # Unify allow/block from all categories
            allow_set, block_set = set(), set()
            eng_allow = getattr(self.config, "engines_allow", None) or {}
            eng_block = getattr(self.config, "engines_block", None) or {}
            for c in cats:
                allow_set |= set(eng_allow.get(c, []))
                block_set |= set(eng_block.get(c, []))
            if allow_set:
                kwargs["engines"] = ",".join(sorted(allow_set))
            if block_set:
                # remove any blocked that are already in allow
                kwargs["blocked_engines"] = ",".join(sorted(block_set - allow_set))

            try:
                raw = searx_call_with_retry(lambda: search.results(q, **kwargs)) or []
            except (RequestException, JSONDecodeError, ValueError) as e:
                dt = time.perf_counter() - t0
                logger.error("[websearch] node=web_search error=%s dt=%.3fs", type(e).__name__, dt)
                # Don't pollute user messages; just return clean delta
                return {"results": [], "categories": cats}

            cleaned = diversify_topk(dedupe_results(normalize_urls(raw)), k=self.config.k)
            dt = time.perf_counter() - t0
            logger.info("[websearch] node=web_search dt=%.3fs q=%r n_raw=%d n_clean=%d", dt, q, len(raw), len(cleaned))
            return {"results": cleaned, "categories": cats}

        return _search

    def _build_summarize_node(self):
        """Summarization node with strict URL whitelist."""
        sys = SystemMessage(content="Be factual, concise, and objective. Always respond in the same language as the user's query.")

        def _strip_punct(u: str) -> str:
            # Remove common punctuation from the right, preserving base URL
            return u.rstrip(").,;:!?]")

        def _summarize(state: SearchState) -> dict:
            t0 = time.perf_counter()
            query = state["query"]
            results = state.get("results") or []

            if not results:
                logger.info("[websearch] node=summarize dt=%.3fs (no results)", time.perf_counter() - t0)
                return {"summary": f"No results found for: {query}"}

            urls = [str(r.get("link", "")) for r in results if isinstance(r.get("link"), str)]
            lines = [
                f"{i+1}. {r.get('title', '')} — {r.get('link', '')}\n{r.get('snippet', '')}"
                for i, r in enumerate(results)
            ]

            whitelist_msg = "You can ONLY cite links exactly from the following list:\n" + "\n".join(urls)
            prompt = (
                f"{whitelist_msg}\n\n"
                "Answer the query with 1–3 paragraphs, citing 2–5 links ONLY from this list.\n"
                "Respond in the same language as the query.\n\n"
                f"Query: {query}\n\nResults:\n" + "\n".join(lines)
            )

            model_instance = getattr(self.config, "model", None)
            if not model_instance:
                top = "\n".join(lines[:3])
                logger.info("[websearch] node=summarize dt=%.3fs (fallback)", time.perf_counter() - t0)
                return {"summary": f"(Fallback without LLM)\n{top}"}

            out = model_instance.invoke([sys, HumanMessage(content=prompt)])
            text = (getattr(out, "content", "") or "").strip()

            # Remove any URL that is not in whitelist (with canonicalization)
            safe = {canonical_url(u) for u in urls}
            for token in re.findall(r"https?://\S+", text):
                token_norm = _strip_punct(token)
                if canonical_url(token_norm) not in safe:
                    text = text.replace(token, "")

            dt = time.perf_counter() - t0
            logger.info("[websearch] node=summarize dt=%.3fs", dt)
            return {"summary": text}

        return _summarize

    def _build_graph(self):
        """Assemble and compile the LangGraph: START → categorize → search → summarize → END."""
        g = StateGraph(SearchState)

        g.add_node("categorize_query", self._build_categorize_node())
        g.add_node(
            "web_search",
            self._build_web_search_node(),
            cache_policy=CachePolicy(ttl=120)
        )
        g.add_node(
            "summarize",
            self._build_summarize_node(),
            cache_policy=CachePolicy(ttl=120)
        )

        g.add_edge(START, "categorize_query")
        g.add_edge("categorize_query", "web_search")
        g.add_edge("web_search", "summarize")
        g.add_edge("summarize", END)

        return g.compile(name="WebSearchAgent")


# ============================================================================
# LangGraph Server Exports
# ============================================================================
# These module-level exports are required for LangGraph server deployment.
# The server discovers graphs via langgraph.json configuration.
# Configuration is loaded from environment variables at runtime.
# ============================================================================

def _create_default_agent():
    """Create default web search agent for LangGraph server.

    This function is called at module import time to create the graph
    that the LangGraph server will expose. Configuration is loaded from
    environment variables.

    Returns:
        Compiled LangGraph graph ready for deployment.
    """
    import os

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available

    # Load all configuration from environment
    config = SearchAgentConfig(
        model_name=os.getenv("MODEL_NAME") or os.getenv("WEBSEARCH_MODEL_NAME", "llama3.1"),
        base_url=os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL"),
        searx_host=os.getenv("SEARX_HOST", "http://localhost:8095"),
        k=int(os.getenv("SEARCH_K", "8")),
        max_categories=int(os.getenv("SEARCH_MAX_CATEGORIES", "3")),
        safesearch=int(os.getenv("SEARCH_SAFESEARCH", "1")),
        lang=os.getenv("SEARCH_LANG", "en"),
        retries=int(os.getenv("SEARCH_RETRIES", "2")),
        backoff_base=float(os.getenv("SEARCH_BACKOFF_BASE", "1.0")),
        temperature=float(os.getenv("SEARCH_TEMPERATURE", "0.5")),
        num_ctx=int(os.getenv("SEARCH_NUM_CTX", "8192")),
    )
    agent = WebSearchAgent(config)
    return agent.graph


# Exports for LangGraph server (referenced in langgraph.json)
websearch_agent = _create_default_agent()
websearch = websearch_agent  # Backward compatibility alias

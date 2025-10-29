"""WebSearch agent implementation (async-first)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional, cast
from zoneinfo import ZoneInfo

from langchain_community.utilities import SearxSearchWrapper
from langchain_core.language_models import BaseChatModel
from langgraph.cache.memory import InMemoryCache
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy

from core.contracts import AgentMixin, ModelFactory
from websearch.config import SearchAgentConfig, SearchState
from websearch.language import LangDetector
from websearch.llm import call_llm_safely, translate_for_search
from websearch.nodes import (
    NodeDependencies,
    build_categorize_node,
    build_summarize_node,
    build_web_search_node,
)

try:  # pragma: no cover - defensive import for optional dependency
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - env helper is optional
    pass

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class WebSearchAgent(AgentMixin[SearchState, dict[str, Any]]):
    """LangGraph-based agent orchestrating categorize → search → summarize."""

    config: SearchAgentConfig

    def __init__(self, config: SearchAgentConfig | None = None):
        super().__init__()
        self.config = config or SearchAgentConfig()

        if getattr(self.config, "model", None) is None:
            self.config.model = cast(BaseChatModel, self._build_model())

        self._langdet = LangDetector()
        self._local_tz: Optional[ZoneInfo] = None
        self._tz_lock = asyncio.Lock()
        self._search_wrapper_cls = SearxSearchWrapper

        self.build()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> BaseChatModel:
        model_name = self.config.model_name or ""
        if not model_name:
            raise ValueError("SearchAgentConfig.model_name must be set")

        return ModelFactory.create_chat_model(
            model_name=model_name,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_ctx=self.config.num_ctx,
        )

    def get_model(self) -> BaseChatModel | None:
        return getattr(self.config, "model", None)

    async def _get_local_tz(self) -> ZoneInfo:
        if self._local_tz is not None:
            return self._local_tz
        async with self._tz_lock:
            if self._local_tz is None:
                self._local_tz = await asyncio.to_thread(ZoneInfo, "America/Sao_Paulo")
        return self._local_tz

    async def call_llm_safely(self, model: BaseChatModel, msgs: list[Any], timeout: float = 90.0):
        """Compatibility shim for callers expecting the old instance method."""

        return await call_llm_safely(model, msgs, timeout)

    async def _call_llm(self, msgs: list[Any], timeout: float = 90.0):
        model = self.get_model()
        if model is None:
            raise RuntimeError("LLM model not configured for WebSearchAgent")
        return await call_llm_safely(model, msgs, timeout)

    async def _translate_query(self, text: str, target_lang: str = "en") -> str:
        text = (text or "").strip()
        if not text or target_lang.lower() == "auto":
            return text
        if self.get_model() is None:
            return text
        try:
            return await translate_for_search(text, target_lang, self._call_llm)
        except Exception as exc:  # pragma: no cover - translation is best-effort
            logger.debug("Translation fallback (%s): %s", target_lang, exc)
            return text

    def _search_wrapper_factory(self) -> SearxSearchWrapper:
        return self._search_wrapper_cls(searx_host=self.config.searx_host)

    # ------------------------------------------------------------------
    # Graph assembly
    # ------------------------------------------------------------------

    def _build_graph(self):
        deps = NodeDependencies(
            config=self.config,
            lang_detector=self._langdet,
            translate_query=self._translate_query,
            call_llm=self._call_llm,
            get_model=self.get_model,
            get_local_tz=self._get_local_tz,
            search_wrapper_factory=self._search_wrapper_factory,
        )

        graph: StateGraph[SearchState] = StateGraph(SearchState)
        graph.add_node("categorize_query", build_categorize_node(deps))
        graph.add_node(
            "web_search",
            build_web_search_node(deps),
            cache_policy=CachePolicy(ttl=120),
        )
        graph.add_node(
            "summarize",
            build_summarize_node(deps),
            cache_policy=CachePolicy(ttl=120),
        )

        graph.add_edge(START, "categorize_query")
        graph.add_edge("categorize_query", "web_search")
        graph.add_edge("web_search", "summarize")
        graph.add_edge("summarize", END)

        cache_backend = InMemoryCache()
        return graph.compile(name="WebSearchAgent", cache=cache_backend)


# ============================================================================
# LangGraph Server Exports
# ============================================================================

def _create_default_agent():
    """Create the default agent for LangGraph server (lazy build on first use)."""

    import os

    try:  # pragma: no cover - optional dependency
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover - optional dependency
        pass

    def _to_bool(x: Optional[str], default: bool = True) -> bool:
        if x is None:
            return default
        x = x.strip().lower()
        return x in {"1", "true", "yes", "y", "on"}

    config = SearchAgentConfig(
        model_name=os.getenv("MODEL_NAME") or os.getenv("WEBSEARCH_MODEL_NAME", "llama3.1"),
        base_url=os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL"),
        searx_host=os.getenv("SEARX_HOST", "http://192.168.30.100:8095"),
        k=int(os.getenv("SEARCH_K", "30")),
        max_categories=int(os.getenv("SEARCH_MAX_CATEGORIES", "3")),
        safesearch=int(os.getenv("SEARCH_SAFESEARCH", "1")),
        lang=os.getenv("SEARCH_LANG", "auto"),
        retries=int(os.getenv("SEARCH_RETRIES", "2")),
        backoff_base=float(os.getenv("SEARCH_BACKOFF_BASE", "1.0")),
        temperature=float(os.getenv("SEARCH_TEMPERATURE", "0.5")),
        num_ctx=int(os.getenv("SEARCH_NUM_CTX", "8192")),
    )
    setattr(config, "pivot_to_english", _to_bool(os.getenv("SEARCH_PIVOT_TO_EN", "1"), True))

    agent = WebSearchAgent(config)
    return agent.agent


# Exports for LangGraph server (referenced in langgraph.json)
websearch_agent = _create_default_agent()
websearch = websearch_agent  # backward compatibility alias


__all__ = ["WebSearchAgent", "websearch_agent", "websearch", "SearxSearchWrapper"]

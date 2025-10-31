"""WebSearch agent implementation (async-first)."""

from __future__ import annotations

import asyncio
import logging
from datetime import timezone as dt_timezone, tzinfo
from typing import Any, Optional, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
from config import get_settings

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

        # Initialize embedder if not already set
        if self.config.embedder is None:
            self.config.embedder = self._build_embedder()

        self._langdet = LangDetector()
        self._local_tz: Optional[tzinfo] = None
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

    def _build_embedder(self) -> object | None:
        """Initialize embedder for MMR reranking.

        Tries to create embeddings model with graceful fallback if dependencies missing.
        Priority order:
        1. HuggingFaceEmbeddings (intfloat/e5-small-v2) - local, fast
        2. OpenAIEmbeddings (text-embedding-3-small) - if API key available
        3. None - fallback to domain-based diversification

        Returns:
            Embeddings model instance or None if unavailable.
        """
        import os

        # Try HuggingFace embeddings first (local, no API key needed)
        try:
            from langchain_huggingface import HuggingFaceEmbeddings # type: ignore

            # Auto-detect GPU availability
            try:
                import torch # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info("[websearch] Initializing HuggingFaceEmbeddings for MMR (device: %s)", device)
            except ImportError:
                device = "cpu"
                logger.info("[websearch] Initializing HuggingFaceEmbeddings for MMR (device: cpu, torch not available)")

            # Allow model override via environment variable
            model_name = os.getenv("EMBEDDINGS_MODEL_NAME", "intfloat/e5-base-v2")
            
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True},
            )
        except ImportError:
            logger.debug("[websearch] langchain-huggingface not available")
        except Exception as exc:
            logger.debug("[websearch] HuggingFaceEmbeddings init failed: %s", exc)

        # Try OpenAI embeddings if API key available
        if os.getenv("OPENAI_API_KEY"):
            try:
                from langchain_openai import OpenAIEmbeddings

                logger.info("[websearch] Initializing OpenAIEmbeddings for MMR")
                return OpenAIEmbeddings(model="text-embedding-3-small")
            except ImportError:
                logger.debug("[websearch] langchain-openai not available")
            except Exception as exc:
                logger.debug("[websearch] OpenAIEmbeddings init failed: %s", exc)

        logger.info("[websearch] No embeddings model available, MMR will use fallback diversification")
        return None

    def get_model(self) -> BaseChatModel | None:
        return getattr(self.config, "model", None)

    async def _get_local_tz(self) -> tzinfo:
        if self._local_tz is not None:
            return self._local_tz
        async with self._tz_lock:
            if self._local_tz is None:
                # Prefer explicit config.local_timezone; otherwise fall back to
                # the global settings timezone (IANA name). This makes the
                # agent follow the same configured timezone used elsewhere.
                tz_name = getattr(self.config, "local_timezone", None)
                if not tz_name:
                    tz_name = getattr(get_settings(), "timezone", "America/Sao_Paulo")
                def _load_zone(name: str):
                    try:
                        return ZoneInfo(name)
                    except ZoneInfoNotFoundError:
                        logger.warning(
                            "[websearch] timezone_not_found falling back to UTC",
                            extra={"requested_tz": name},
                        )
                    except Exception as exc:
                        logger.warning(
                            "[websearch] timezone_load_error falling back to UTC: %s",
                            exc,
                            extra={"requested_tz": name},
                        )
                    # Update config to reflect fallback
                    self.config.local_timezone = "UTC"
                    return dt_timezone.utc

                self._local_tz = await asyncio.to_thread(_load_zone, tz_name)
        return self._local_tz


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
            embedder=self.config.embedder,
        )

        graph: StateGraph[SearchState] = StateGraph(SearchState)
        graph.add_node(
            "categorize_query",
            build_categorize_node(deps),
            cache_policy=CachePolicy(ttl=60),
        )
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
        k=int(os.getenv("SEARCH_K", "20")),
        max_categories=int(os.getenv("SEARCH_MAX_CATEGORIES", "3")),
        safesearch=int(os.getenv("SEARCH_SAFESEARCH", "1")),
        lang=os.getenv("SEARCH_LANG", "auto"),
        retries=int(os.getenv("SEARCH_RETRIES", "2")),
        backoff_base=float(os.getenv("SEARCH_BACKOFF_BASE", "1.0")),
        temperature=float(os.getenv("SEARCH_TEMPERATURE", "0.5")),
        pivot_to_english=_to_bool(os.getenv("SEARCH_PIVOT_TO_EN", "1"), True),
        local_timezone=os.getenv("LOCAL_TIMEZONE", "America/Sao_Paulo"),
    )

    agent = WebSearchAgent(config)
    return agent.agent


# Exports for LangGraph server (referenced in langgraph.json)
websearch_agent = _create_default_agent()
websearch = websearch_agent  # backward compatibility alias


__all__ = ["WebSearchAgent", "websearch_agent", "websearch", "SearxSearchWrapper"]

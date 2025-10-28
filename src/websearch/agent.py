# websearch/agent.py

"""WebSearch agent implementation (async-first).

Flow: categorize -> search -> summarize using SearxNG and an LLM.

Highlights:
- Async-first em todos os nós (ainvoke/astream)
- SearxNG síncrono encapsulado com asyncio.to_thread
- Retry/backoff assíncrono (await asyncio.sleep)
- ModelFactory centraliza diferenças OpenAI/Ollama
- Nós retornam apenas deltas (state as source of truth)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from json import JSONDecodeError
from typing import Any, AsyncIterator, cast

from langchain_community.utilities import SearxSearchWrapper
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy
from requests import RequestException

from core.contracts import AgentMixin, ModelFactory
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

# Optional: load .env aqui; em prod prefira fazer no bootstrap da app.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class WebSearchAgent(AgentMixin):
    """Agento de WebSearch: categorize -> search -> summarize (async)."""

    config: SearchAgentConfig

    def __init__(self, config: SearchAgentConfig | None = None):
        super().__init__()
        self.config = config or SearchAgentConfig()

        # Garante modelo caso não venha em config
        if getattr(self.config, "model", None) is None:
            self.config.model = cast(BaseChatModel, self._build_model())

        # Eager build (troque para lazy removendo esta linha se preferir)
        self.build()

    # ---------------------------
    # BUILDERS
    # ---------------------------

    def _build_model(self) -> BaseChatModel:
        """Cria o modelo via ModelFactory."""
        model_name = self.config.model_name or ""
        if not model_name:
            raise ValueError("SearchAgentConfig.model_name must be set")

        return ModelFactory.create_chat_model(
            model_name=model_name,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_ctx=self.config.num_ctx,
        )

    def _build_categorize_node(self):
        """Node de categorização (heurísticas + LLM structured output)."""

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

        async def llm_categories(query: str, hints: list[str], limit: int) -> list[str]:
            model_instance = getattr(self.config, "model", None)
            if not model_instance:
                return ["general"]
            try:
                model_struct = model_instance.with_structured_output(CategoryResponse)
                user_msg = HumanMessage(content=f'QUERY: "{query}"\nHINTS: {json.dumps(hints, ensure_ascii=False)}')
                # prefer async; se o backend não suportar, to_thread resolve
                try:
                    resp = await model_struct.ainvoke([SYS_PROMPT, *FEWSHOT, user_msg])  # type: ignore[attr-defined]
                except AttributeError:
                    resp = await asyncio.to_thread(model_struct.invoke, [SYS_PROMPT, *FEWSHOT, user_msg])
                cats = (getattr(resp, "categories", None) or [])
                cats = [c for c in cats if c in ALLOWED_CATEGORIES]
                cats = (cats or ["general"])[: max(1, limit)]
                return cats
            except Exception as e:
                logger.debug("Structured categorize failed: %r", e)
                return ["general"]

        async def _categorize(state: SearchState) -> dict:
            t0 = time.perf_counter()
            query = state["query"]
            hints = heuristic_categories(query)
            cats = await llm_categories(query, hints, self.config.max_categories)
            dt = time.perf_counter() - t0
            logger.info("[websearch] node=categorize_query dt=%.3fs cats=%s", dt, cats)
            return {"categories": cats}

        return _categorize

    def _build_web_search_node(self):
        """Node de busca SearxNG com retry/backoff assíncrono e pós-processamento."""
        search = SearxSearchWrapper(searx_host=self.config.searx_host)

        async def searx_call_with_retry(call):
            for attempt in range(self.config.retries + 1):
                try:
                    # SearxSearchWrapper é síncrono; roda em thread
                    return await asyncio.to_thread(call)
                except (RequestException, JSONDecodeError, ValueError) as e:
                    if attempt >= self.config.retries:
                        raise
                    delay = self.config.backoff_base * (2 ** attempt)
                    logger.warning(
                        "[websearch] retry=%d delay=%.2fs error=%s", attempt + 1, delay, type(e).__name__
                    )
                    await asyncio.sleep(delay)

        async def _search(state: SearchState) -> dict:
            t0 = time.perf_counter()
            q = state["query"].strip()
            cats = state.get("categories") or ["general"]
            # Garante 'general' e remove duplicatas preservando ordem
            seen = set()
            cats = [c for c in ([*cats, "general"]) if not (c in seen or seen.add(c))]

            kwargs: dict[str, Any] = {
                "categories": cats,
                "num_results": self.config.k * 2,
                "safesearch": self.config.safesearch,
            }
            if self.config.lang:
                kwargs["language"] = self.config.lang
            tr = pick_time_range(cats)
            if tr:
                kwargs["time_range"] = tr

            # União de allow/block de todas as categorias
            allow_set, block_set = set(), set()
            eng_allow = getattr(self.config, "engines_allow", None) or {}
            eng_block = getattr(self.config, "engines_block", None) or {}
            for c in cats:
                allow_set |= set(eng_allow.get(c, []))
                block_set |= set(eng_block.get(c, []))
            if allow_set:
                kwargs["engines"] = ",".join(sorted(allow_set))
            if block_set:
                kwargs["blocked_engines"] = ",".join(sorted(block_set - allow_set))

            try:
                raw = await searx_call_with_retry(lambda: search.results(q, **kwargs)) or []
            except (RequestException, JSONDecodeError, ValueError) as e:
                dt = time.perf_counter() - t0
                logger.error("[websearch] node=web_search error=%s dt=%.3fs", type(e).__name__, dt)
                return {"results": [], "categories": cats}

            cleaned = diversify_topk(dedupe_results(normalize_urls(raw)), k=self.config.k)
            dt = time.perf_counter() - t0
            logger.info(
                "[websearch] node=web_search dt=%.3fs q=%r n_raw=%d n_clean=%d",
                dt, q, len(raw), len(cleaned)
            )
            return {"results": cleaned, "categories": cats}

        return _search

    def _build_summarize_node(self):
        """Node de sumarização com whitelist estrita de URLs."""
        sys = SystemMessage(
            content="Be factual, concise, and objective. Always respond in the same language as the user's query."
        )

        def _strip_punct(u: str) -> str:
            return u.rstrip(").,;:!?]")

        async def _summarize(state: SearchState) -> dict:
            t0 = time.perf_counter()
            query = state["query"]
            results = state.get("results") or []

            if not results:
                dt = time.perf_counter() - t0
                logger.info("[websearch] node=summarize dt=%.3fs (no results)", dt)
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
                dt = time.perf_counter() - t0
                logger.info("[websearch] node=summarize dt=%.3fs (fallback)", dt)
                return {"summary": f"(Fallback without LLM)\n{top}"}

            # Prefer async; se não houver, executa sync no threadpool
            try:
                out = await model_instance.ainvoke([sys, HumanMessage(content=prompt)])  # type: ignore[attr-defined]
            except AttributeError:
                out = await asyncio.to_thread(model_instance.invoke, [sys, HumanMessage(content=prompt)])

            text = (getattr(out, "content", "") or "").strip()

            # Remoção de URLs fora da whitelist (com canonicalização)
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
        """Monta e compila o LangGraph: START → categorize → search → summarize → END."""
        g = StateGraph(SearchState)

        g.add_node("categorize_query", self._build_categorize_node())
        g.add_node(
            "web_search",
            self._build_web_search_node(),
            cache_policy=CachePolicy(ttl=120),
        )
        g.add_node(
            "summarize",
            self._build_summarize_node(),
            cache_policy=CachePolicy(ttl=120),
        )

        g.add_edge(START, "categorize_query")
        g.add_edge("categorize_query", "web_search")
        g.add_edge("web_search", "summarize")
        g.add_edge("summarize", END)

        return g.compile(name="WebSearchAgent")


# ============================================================================
# LangGraph Server Exports
# ============================================================================

def _create_default_agent():
    """Cria o agente default para LangGraph server (lazy build na 1ª chamada)."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

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
    # Retorna o grafo já compilado (AgentMixin.build foi chamado em __init__)
    return agent.agent


# Exports para LangGraph server (referenced in langgraph.json)
websearch_agent = _create_default_agent()
websearch = websearch_agent  # alias retrocompatível

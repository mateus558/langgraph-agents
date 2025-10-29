# websearch/agent.py

"""WebSearch agent implementation (async-first).

Flow: categorize -> search -> summarize using SearxNG and an LLM.

Highlights:
- Async-first nodes (ainvoke/astream)
- Synchronous SearxNG wrapper executed via asyncio.to_thread
- Async retry/backoff (await asyncio.sleep)
- ModelFactory centralizes OpenAI/Ollama differences
- Nodes return deltas only (state as the source of truth)
- RunnableLambda-wrapped nodes with typed state to satisfy LangGraph types
- Auto language detection for Searx (SEARCH_LANG=auto)
- Categorization happens on an **English-translated** query (pivot)
- Optional English pivot for search results (SEARCH_PIVOT_TO_EN=1)
- Async-safe ZoneInfo loading (avoids BlockingError)
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from json import JSONDecodeError
from datetime import datetime, timezone
from typing import Any, Optional, cast
from zoneinfo import ZoneInfo

from langchain_community.utilities import SearxSearchWrapper
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import CachePolicy
from langgraph.cache.memory import InMemoryCache
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# -----------------------------------------------------------------------------
# Lightweight language detection with graceful fallbacks
# -----------------------------------------------------------------------------
class LangDetector:
    """Best-effort language detector with optional deps.

    Priority:
      1) pycld3 (fast, accurate)
      2) langdetect
      3) heuristic fallback (accent/stopword hints)
    Returns a 2-letter ISO code suitable for Searx (e.g., 'en', 'pt', 'es').
    """

    def __init__(self) -> None:
        try:
            import pycld3  # type: ignore
            self._cld3 = pycld3
        except Exception:
            self._cld3 = None
        try:
            from langdetect import detect  # type: ignore
            self._detect = detect
        except Exception:
            self._detect = None

    @staticmethod
    def _norm(code: Optional[str]) -> Optional[str]:
        if not code:
            return None
        code = code.lower().strip().split("-")[0]
        if code in {"pt", "en", "es", "fr", "de", "it"}:
            return code
        if code in {"ptbr", "pt_br"}:
            return "pt"
        return code[:2] if len(code) >= 2 else None

    async def detect2(self, text: str) -> Optional[str]:
        txt = (text or "").strip()
        if not txt:
            return None
        # 1) pycld3
        if self._cld3 is not None:
            def _cld3_detect(t: str) -> Optional[str]:
                if self._cld3 is None:
                    return None
                lang = self._cld3.get_language(t)
                if lang and getattr(lang, "is_reliable", False):
                    return self._norm(getattr(lang, "language", None))
                return self._norm(getattr(lang, "language", None))
            try:
                return await asyncio.to_thread(_cld3_detect, txt)
            except Exception:
                pass
        # 2) langdetect
        if self._detect is not None:
            try:
                code = await asyncio.to_thread(self._detect, txt)
                return self._norm(code)
            except Exception:
                pass
        # 3) heuristic: quick accents/stopwords
        lowered = txt.lower()
        if any(ch in lowered for ch in "ãõáéíóúçâêô") or "quando" in lowered or "próximo" in lowered:
            return "pt"
        if "¿" in lowered or "¡" in lowered or "cuándo" in lowered:
            return "es"
        return None


class WebSearchAgent(AgentMixin):
    """WebSearch agent: categorize -> search -> summarize (async)."""

    config: SearchAgentConfig

    def __init__(self, config: SearchAgentConfig | None = None):
        super().__init__()
        self.config = config or SearchAgentConfig()

        # Ensure a model exists if not provided in config
        if getattr(self.config, "model", None) is None:
            self.config.model = cast(BaseChatModel, self._build_model())

        # Timezone cache (ZoneInfo may touch disk)
        self._local_tz: Optional[ZoneInfo] = None
        self._tz_lock = asyncio.Lock()

        # Language detector (lazy, optional deps)
        self._langdet = LangDetector()

        # Eager build (switch to lazy by removing this line if desired)
        self.build()

    # ---------------------------
    # BUILDERS
    # ---------------------------

    @staticmethod
    def _to_searx_locale(code: Optional[str]) -> Optional[str]:
        """Normalize a language code to a SearxNG-compatible locale.

        Examples:
        - "en" -> "en"
        - "en-us" / "en_US" -> "en-US"
        - "pt-br" / "pt_BR" -> "pt-BR"
        - "auto" / "" / None -> None (let Searx decide)
        """
        if not code:
            return None
        c = code.strip()
        if not c:
            return None
        if c.lower() == "auto":
            return None
        c = c.replace("_", "-")
        parts = c.split("-")
        if len(parts) == 1:
            return parts[0].lower()
        lang = parts[0].lower()
        region = parts[1].upper() if parts[1] else ""
        return f"{lang}-{region}" if region else lang

    def _build_model(self) -> BaseChatModel:
        """Create the model via ModelFactory."""
        model_name = self.config.model_name or ""
        if not model_name:
            raise ValueError("SearchAgentConfig.model_name must be set")

        return ModelFactory.create_chat_model(
            model_name=model_name,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_ctx=self.config.num_ctx,
        )

    async def _get_local_tz(self) -> ZoneInfo:
        """Load and cache ZoneInfo in a worker thread to avoid blocking the event loop."""
        if self._local_tz is not None:
            return self._local_tz
        async with self._tz_lock:
            if self._local_tz is None:
                self._local_tz = await asyncio.to_thread(ZoneInfo, "America/Sao_Paulo")
        return self._local_tz

    # --- Optional translation pivot to English for categorization/search ---
    async def _translate_for_search(self, text: str, target_lang: str = "en") -> str:
        """LLM-based translation for pivot. Plain text out, no quotes."""
        text = (text or "").strip()
        if not text or target_lang.lower() == "auto":
            return text
        model_instance = getattr(self.config, "model", None)
        if not model_instance:
            return text
        sys = SystemMessage(
            content=(
                "Translate the user's query for web search categorization. "
                "Output ONLY the translation as plain text; no quotes, no extra text."
            )
        )
        human = HumanMessage(content=f"Target language: {target_lang}\nText:\n{text}")
        out = await self.call_llm_safely(model_instance, [sys, human], timeout=20.0)
        translated = (getattr(out, "content", "") or "").strip()
        if len(translated) >= 2 and translated[0] in {'"', "'"} and translated[-1] == translated[0]:
            translated = translated[1:-1].strip()
        return translated or text

    def _build_categorize_node(self) -> RunnableLambda[SearchState, dict]:
        """Categorization node (heuristics + LLM structured output) on **English pivot**."""

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

        async def llm_categories(query_en: str, hints: list[str], limit: int) -> list[str]:
            model_instance = getattr(self.config, "model", None)
            if not model_instance:
                return ["general"]
            try:
                model_struct = model_instance.with_structured_output(CategoryResponse)
                user_msg = HumanMessage(content=f'QUERY: "{query_en}"\nHINTS: {json.dumps(hints, ensure_ascii=False)}')
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

        async def _categorize(state: SearchState, config: RunnableConfig | None = None) -> dict:
            t0 = time.perf_counter()
            query = state["query"]

            # Detect language and translate to English for categorization
            lang = await self._langdet.detect2(query) or "en"
            query_en = query if lang == "en" else await self._translate_for_search(query, "en")

            # Combine heuristic hints from original and English text
            hints_orig = heuristic_categories(query)
            hints_en = heuristic_categories(query_en) if query_en != query else []
            hints = list({*hints_orig, *hints_en})  # dedupe

            cats = await llm_categories(query_en, hints, self.config.max_categories)
            dt = time.perf_counter() - t0
            logger.info("[websearch] node=categorize_query dt=%.3fs cats=%s lang=%s", dt, cats, lang)
            return {"categories": cats, "lang": lang, "query_en": query_en}

        return RunnableLambda(_categorize)

    def _build_web_search_node(self) -> RunnableLambda[SearchState, dict]:
        """SearxNG search node with async retry/backoff, auto-lang, and optional EN pivot for recall."""
        search = SearxSearchWrapper(searx_host=self.config.searx_host)

        async def searx_call_with_retry(call):
            for attempt in range(self.config.retries + 1):
                try:
                    return await asyncio.to_thread(call)
                except (RequestException, JSONDecodeError, ValueError) as e:
                    if attempt >= self.config.retries:
                        raise
                    delay = self.config.backoff_base * (2 ** attempt)
                    logger.warning(
                        "[websearch] retry=%d delay=%.2fs error=%s", attempt + 1, delay, type(e).__name__
                    )
                    await asyncio.sleep(delay)

        async def _search(state: SearchState, config: RunnableConfig | None = None) -> dict:
            t0 = time.perf_counter()
            q = state["query"].strip()
            q_en = (state.get("query_en") or q).strip()
            cats = state.get("categories") or ["general"]
            seen: set[str] = set()
            cats = [c for c in ([*cats, "general"]) if not (c in seen or seen.add(c))]

            # --- language (auto or fixed) ---
            lang_cfg = (self.config.lang or "auto").strip().lower()
            lang: Optional[str] = lang_cfg
            if lang_cfg in {"", "auto"}:
                detected = state.get("lang") or await self._langdet.detect2(q)
                if detected:
                    lang = detected
            # Normalize to Searx locale (None means let Searx decide)
            lang_norm = self._to_searx_locale(lang)

            # Decide whether to run an English pivot for search recall
            likely_non_en = bool(re.search(r"[^\x00-\x7F]", q)) or any(
                w in q.lower() for w in ("quando", "próximo", "próxima", "ação", "notícia", "pesquisa")
            )
            pivot_enabled = bool(getattr(self.config, "pivot_to_english", True))
            
            logger.info(
                "[websearch] pivot_check: pivot_enabled=%s lang=%s likely_non_en=%s will_pivot=%s",
                pivot_enabled, lang, likely_non_en, 
                (pivot_enabled and (lang or "auto") != "en" and likely_non_en)
            )

            def build_kwargs(lang_code: Optional[str]) -> dict[str, Any]:
                kw: dict[str, Any] = {
                    "categories": cats,
                    "num_results": self.config.k * 2,
                    "safesearch": self.config.safesearch,
                }
                # Only set language if we have a concrete locale (not "auto")
                if lang_code:
                    kw["language"] = lang_code
                tr = pick_time_range(cats)
                if tr:
                    kw["time_range"] = tr
                allow_set, block_set = set(), set()
                eng_allow = getattr(self.config, "engines_allow", None) or {}
                eng_block = getattr(self.config, "engines_block", None) or {}
                for c in cats:
                    allow_set |= set(eng_allow.get(c, []))
                    block_set |= set(eng_block.get(c, []))
                if allow_set:
                    kw["engines"] = ",".join(sorted(allow_set))
                if block_set:
                    kw["blocked_engines"] = ",".join(sorted(block_set - allow_set))
                return kw

            async def run_query(query: str, lang_code: Optional[str]):
                return await searx_call_with_retry(lambda: search.results(query, **build_kwargs(lang_code))) or []

            results_union: list[dict] = []
            lang_display = lang_norm or "auto"
            pivot_lang: Optional[str] = None
            try:
                if pivot_enabled and (lang or "auto") != "en" and likely_non_en:
                    # Use the English-translated text from categorization stage to avoid double translation
                    translated = q_en if q_en else await self._translate_for_search(q, "en")
                    pivot_lang = "en"
                    raw_orig, raw_en = await asyncio.gather(
                        run_query(q, lang_norm),
                        run_query(translated, "en"),
                    )
                    results_union = raw_orig + raw_en
                else:
                    results_union = await run_query(q, lang_norm)
            except (RequestException, JSONDecodeError, ValueError) as e:
                dt = time.perf_counter() - t0
                logger.error("[websearch] node=web_search error=%s dt=%.3fs", type(e).__name__, dt)
                return {"results": [], "categories": cats, "lang": lang_display}

            cleaned = diversify_topk(dedupe_results(normalize_urls(results_union)), k=self.config.k)

            # Fallback: if no results and "general" was included, retry without it
            if not cleaned and "general" in cats and len(cats) > 1:
                logger.warning(
                    "[websearch] No results with categories %s, retrying without 'general'",
                    cats
                )
                fallback_cats = [c for c in cats if c != "general"]
                if fallback_cats:
                    # Quick retry with working categories only
                    try:
                        if pivot_enabled and (lang or "auto") != "en" and likely_non_en:
                            translated = q_en if q_en else await self._translate_for_search(q, "en")
                            raw_orig, raw_en = await asyncio.gather(
                                run_query(q, lang_norm),
                                run_query(translated, "en"),
                            )
                            results_union = raw_orig + raw_en
                        else:
                            results_union = await run_query(q, lang_norm)
                        cleaned = diversify_topk(dedupe_results(normalize_urls(results_union)), k=self.config.k)
                        cats = fallback_cats  # Update categories in return
                        logger.info("[websearch] Fallback with cats=%s produced %d results", fallback_cats, len(cleaned))
                    except Exception as e:
                        logger.warning("[websearch] Fallback failed: %s", e)

            dt = time.perf_counter() - t0
            logger.info(
                "[websearch] node=web_search dt=%.3fs q=%r n_raw=%d n_clean=%d lang=%s pivot=%s",
                dt, q, len(results_union), len(cleaned), lang_display, (pivot_lang or "none"),
            )
            
            # Debug: log first raw result to check field names
            if results_union and not cleaned:
                logger.warning("[websearch] Raw results present but cleaned is empty. Sample raw result keys: %s",
                              list(results_union[0].keys()) if results_union else "N/A")
            
            return {"results": cleaned, "categories": cats, "lang": lang_display}

        return RunnableLambda(_search)

    # -------------------------------------------------------------------------
    # Async-safe wrapper for model calls (prevents blocking I/O in event loop)
    # -------------------------------------------------------------------------
    async def call_llm_safely(self, model, msgs, timeout: float = 90.0):
        provider = getattr(model, "model_provider", "") or getattr(model, "client", "").__class__.__name__.lower()

        async def _try_async():
            if hasattr(model, "ainvoke") and str(provider).lower() in {"openai", "anthropic", "google"}:
                return await model.ainvoke(msgs)
            return await asyncio.to_thread(model.invoke, msgs)

        return await asyncio.wait_for(_try_async(), timeout=timeout)

    # -------------------------------------------------------------------------
    # Summarization node
    # -------------------------------------------------------------------------
    def _build_summarize_node(self) -> RunnableLambda[SearchState, dict]:
        """Summarization node with strict URL whitelist and current timestamp injection."""

        def _strip_punct(u: str) -> str:
            return u.rstrip(").,;:!?]")

        async def _summarize(state: SearchState, config: RunnableConfig | None = None) -> dict:
            t0 = time.perf_counter()
            query = state["query"]
            results = state.get("results") or []

            if not results:
                dt = time.perf_counter() - t0
                logger.info("[websearch] node=summarize dt=%.3fs (no results)", dt)
                return {"summary": f"No results found for: {query}"}

            # Time anchors using cached async ZoneInfo (avoid BlockingError)
            now_utc = datetime.now(timezone.utc)
            tz = await self._get_local_tz()
            now_local = now_utc.astimezone(tz)
            now_utc_str = now_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
            now_local_str = now_local.strftime("%Y-%m-%d %H:%M:%S %Z")

            sys = SystemMessage(
                content=(
                    "You are a factual summarizer. Be concise, verifiable, objective. "
                    "Always answer in the user's language.\n"
                    f"Current time (UTC): {now_utc_str}\n"
                    f"Current time (America/Sao_Paulo): {now_local_str}\n\n"
                    "TEMPORAL POLICY:\n"
                    "• Interpret 'next', 'upcoming', and schedules strictly relative to the current time above.\n"
                    "• Convert times to America/Sao_Paulo when presenting times; include weekday and explicit date.\n"
                    "• If a source mentions a past date, do NOT call it 'next'; mark it as 'already occurred'.\n"
                    "• Prefer sources with explicit dates/times over vague claims; prefer newest when conflicting.\n"
                    "• If no future date > now is found, say 'No upcoming date found in the results.' Do not guess.\n"
                    "• For time-related queries, the 'Summary' MUST start with an 'As of <UTC>/<local>:' clause.\n"
                )
            )

            urls = [str(r.get("link", "")) for r in results if isinstance(r.get("link"), str)]
            items: list[str] = []
            for i, r in enumerate(results, start=1):
                title = str(r.get("title") or "").strip()
                link = str(r.get("link") or "").strip()
                snip = str(r.get("snippet") or "").strip()
                if link:
                    items.append(f"[{i}] {title}\nURL: {link}\nSnippet: {snip}")

            whitelist_msg = (
                "ALLOWED SOURCES (cite only these, using their numeric index):\n"
                + "\n".join(f"[{i+1}] {u}" for i, u in enumerate(urls))
            )

            prompt = (
                f"{whitelist_msg}\n\n"
                "TASK: Summarize and answer the query using only the listed sources.\n"
                "MANDATORY OUTPUT FORMAT:\n"
                "Summary:\n"
                "- Start with: 'As of <YYYY-MM-DD HH:MM UTC / local>:'\n"
                "- Include the next upcoming or most recent relevant date strictly compared to the current time.\n"
                "Details (optional):\n"
                "- Add factual context such as periodicity, duration, or schedule patterns when mentioned.\n"
                "Sources:\n"
                "- List only the citation indices used, ordered by importance (e.g., [1], [4], [2]).\n\n"
                "CITATION AND CONFLICT RULES:\n"
                "- Every date or quantitative claim must have at least one citation [n].\n"
                "- When multiple sources disagree, identify the conflict and prefer the newest data.\n"
                "- If no upcoming or relevant date is found, explicitly say: 'No future or current date found in the results.'\n\n"
                f"Query: {query}\n\n"
                "RESULTS (to base your answer on):\n" + "\n\n".join(items)
            )

            model_instance = getattr(self.config, "model", None)
            if not model_instance:
                top = "\n".join(items[:3])
                dt = time.perf_counter() - t0
                logger.info("[websearch] node=summarize dt=%.3fs (fallback)", dt)
                return {"summary": f"(Fallback without LLM)\n{top}"}

            out = await self.call_llm_safely(model_instance, [sys, HumanMessage(content=prompt)], timeout=90.0)

            text = (getattr(out, "content", "") or "").strip()

            # Clean URLs not in whitelist
            safe = {canonical_url(u) for u in urls}
            for token in re.findall(r"https?://\S+", text):
                token_norm = _strip_punct(token)
                if canonical_url(token_norm) not in safe:
                    text = text.replace(token, "")

            dt = time.perf_counter() - t0
            logger.info("[websearch] node=summarize dt=%.3fs", dt)
            return {"summary": text}

        return RunnableLambda(_summarize)

    def _build_graph(self):
        """Assemble and compile LangGraph: START → categorize → search → summarize → END."""
        g: StateGraph[SearchState] = StateGraph(SearchState)

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

        cache_backend = InMemoryCache()

        return g.compile(name="WebSearchAgent", cache=cache_backend)


# ============================================================================
# LangGraph Server Exports
# ============================================================================

def _create_default_agent():
    """Create the default agent for LangGraph server (lazy build on first use)."""
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
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
    # Optional flag to enable English pivot in search
    setattr(config, "pivot_to_english", _to_bool(os.getenv("SEARCH_PIVOT_TO_EN", "1"), True))

    agent = WebSearchAgent(config)
    return agent.agent


# Exports for LangGraph server (referenced in langgraph.json)
websearch_agent = _create_default_agent()
websearch = websearch_agent  # backward compatibility alias

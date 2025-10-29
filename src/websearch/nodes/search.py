"""Web search node builder for the websearch agent."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from json import JSONDecodeError
from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig, RunnableLambda
from requests import RequestException

from websearch.config import SearchState
from websearch.utils import (
    dedupe_results,
    diversify_topk,
    normalize_urls,
    pick_time_range,
    to_searx_locale,
)

from .shared import NodeDependencies

logger = logging.getLogger(__name__)


def build_web_search_node(deps: NodeDependencies) -> RunnableLambda:
    """Return the web search node runnable."""

    search_client = deps.search_wrapper_factory()

    async def searx_call_with_retry(call: Callable[[], list[dict[str, Any]]]):
        for attempt in range(deps.config.retries + 1):
            try:
                return await asyncio.to_thread(call)
            except (RequestException, JSONDecodeError, ValueError) as exc:
                if attempt >= deps.config.retries:
                    raise
                delay = deps.config.backoff_base * (2 ** attempt)
                logger.warning(
                    "[websearch] retry=%d delay=%.2fs error=%s", attempt + 1, delay, type(exc).__name__
                )
                await asyncio.sleep(delay)
        return []

    def build_kwargs(lang_code: Optional[str], categories: list[str]) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "categories": categories,
            "num_results": deps.config.k * 2,
            "safesearch": deps.config.safesearch,
        }
        if lang_code:
            kw["language"] = lang_code
        tr = pick_time_range(categories)
        if tr:
            kw["time_range"] = tr
        allow_set, block_set = set(), set()
        eng_allow = getattr(deps.config, "engines_allow", None) or {}
        eng_block = getattr(deps.config, "engines_block", None) or {}
        for cat in categories:
            allow_set |= set(eng_allow.get(cat, []))
            block_set |= set(eng_block.get(cat, []))
        if allow_set:
            kw["engines"] = ",".join(sorted(allow_set))
        if block_set:
            kw["blocked_engines"] = ",".join(sorted(block_set - allow_set))
        return kw

    async def run_query(query: str, lang_code: Optional[str], categories: list[str]) -> list[dict[str, Any]]:
        return await searx_call_with_retry(
            lambda: search_client.results(query, **build_kwargs(lang_code, categories))
        )

    async def _search(state: SearchState, config: RunnableConfig | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        q = state["query"].strip()
        q_en = (state.get("query_en") or q).strip()
        cats = state.get("categories") or ["general"]
        seen: set[str] = set()
        cats = [c for c in ([*cats, "general"]) if not (c in seen or seen.add(c))]

        lang_cfg = (deps.config.lang or "auto").strip().lower()
        lang: Optional[str] = lang_cfg
        if lang_cfg in {"", "auto"}:
            detected = state.get("lang") or await deps.lang_detector.detect(q)
            if detected:
                lang = detected
        lang_norm = to_searx_locale(lang)

        likely_non_en = bool(re.search(r"[^\x00-\x7F]", q)) or any(
            w in q.lower() for w in ("quando", "próximo", "próxima", "ação", "notícia", "pesquisa")
        )
        pivot_enabled = bool(getattr(deps.config, "pivot_to_english", True))

        logger.info(
            "[websearch] pivot_check: pivot_enabled=%s lang=%s likely_non_en=%s will_pivot=%s",
            pivot_enabled,
            lang,
            likely_non_en,
            (pivot_enabled and (lang or "auto") != "en" and likely_non_en),
        )

        results_union: list[dict[str, Any]] = []
        lang_display = lang_norm or "auto"
        pivot_lang: Optional[str] = None

        async def pivot_queries(active_categories: list[str]) -> list[dict[str, Any]]:
            translated = q_en if q_en else await deps.translate_query(q, "en")
            raw_orig, raw_en = await asyncio.gather(
                run_query(q, lang_norm, active_categories),
                run_query(translated, "en", active_categories),
            )
            return raw_orig + raw_en

        async def plain_query(active_categories: list[str]) -> list[dict[str, Any]]:
            return await run_query(q, lang_norm, active_categories)

        try:
            if pivot_enabled and (lang or "auto") != "en" and likely_non_en:
                pivot_lang = "en"
                results_union = await pivot_queries(cats)
            else:
                results_union = await plain_query(cats)
        except (RequestException, JSONDecodeError, ValueError) as exc:
            dt = time.perf_counter() - t0
            logger.error("[websearch] node=web_search error=%s dt=%.3fs", type(exc).__name__, dt)
            return {"results": [], "categories": cats, "lang": lang_display}

        cleaned = diversify_topk(dedupe_results(normalize_urls(results_union)), k=deps.config.k)

        if not cleaned and "general" in cats and len(cats) > 1:
            logger.warning("[websearch] No results with categories %s, retrying without 'general'", cats)
            fallback_cats = [c for c in cats if c != "general"]
            if fallback_cats:
                try:
                    if pivot_enabled and (lang or "auto") != "en" and likely_non_en:
                        results_union = await pivot_queries(fallback_cats)
                    else:
                        results_union = await plain_query(fallback_cats)
                    cleaned = diversify_topk(dedupe_results(normalize_urls(results_union)), k=deps.config.k)
                    cats = fallback_cats
                    logger.info(
                        "[websearch] Fallback with cats=%s produced %d results", fallback_cats, len(cleaned)
                    )
                except Exception as exc:  # pragma: no cover - fallback best-effort
                    logger.warning("[websearch] Fallback failed: %s", exc)

        dt = time.perf_counter() - t0
        logger.info(
            "[websearch] node=web_search dt=%.3fs q=%r n_raw=%d n_clean=%d lang=%s pivot=%s",
            dt,
            q,
            len(results_union),
            len(cleaned),
            lang_display,
            (pivot_lang or "none"),
        )

        if results_union and not cleaned:
            logger.warning(
                "[websearch] Raw results present but cleaned is empty. Sample raw result keys: %s",
                list(results_union[0].keys()) if results_union else "N/A",
            )

        return {"results": cleaned, "categories": cats, "lang": lang_display}

    return RunnableLambda(_search)


__all__ = ["build_web_search_node"]

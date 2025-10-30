"""Web search node builder for the websearch agent."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from json import JSONDecodeError
import random
from typing import Any, Callable, Optional

from langchain_core.runnables import RunnableConfig, RunnableLambda
from requests import RequestException

from websearch.config import SearchState
from websearch.utils import (
    dedupe_results,
    diversify_topk,
    diversify_topk_mmr,
    normalize_urls,
    pick_time_range,
    to_searx_locale,
)

from .shared import NodeDependencies

logger = logging.getLogger(__name__)


def build_web_search_node(deps: NodeDependencies) -> RunnableLambda:
    """Return the web search node runnable."""

    search_client = deps.search_wrapper_factory()

    # Cap concurrent Searx calls per agent to protect the backend
    _sema = asyncio.Semaphore(max(1, deps.config.searx_max_concurrency))

    async def searx_call_with_retry(call: Callable[[], list[dict[str, Any]]]):
        for attempt in range(deps.config.retries + 1):
            try:
                async with _sema:
                    return await asyncio.to_thread(call)
            except (RequestException, JSONDecodeError, ValueError) as exc:
                if attempt >= deps.config.retries:
                    raise
                # Exponential backoff with jitter (0.8x–1.2x)
                delay = deps.config.backoff_base * (2 ** attempt)
                delay *= (0.8 + random.random() * 0.4)
                logger.warning(
                    "[websearch] retry=%d delay=%.2fs error=%s", attempt + 1, delay, type(exc).__name__
                )
                await asyncio.sleep(delay)
        return []

    def build_kwargs(lang_code: Optional[str], categories: list[str]) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "num_results": deps.config.k * 2,
            "safesearch": deps.config.safesearch,
        }
        # Only include categories if non-empty; empty means let SearxNG choose defaults
        if categories:
            kw["categories"] = categories
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
        kw = build_kwargs(lang_code, categories)
        logger.debug(
            "[websearch] Searx call params: language=%s categories=%s time_range=%s engines=%s blocked=%s",
            kw.get("language"),
            kw.get("categories"),
            kw.get("time_range"),
            kw.get("engines"),
            kw.get("blocked_engines"),
        )
        return await searx_call_with_retry(lambda: search_client.results(query, **kw))

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
            # If no results came back, try once without specifying a language filter
            if not results_union:
                try:
                    logger.info("[websearch] No results from initial query, retrying without language filter")
                    results_union = await run_query(q, None, cats)
                    logger.info(
                        "[websearch] No-language retry produced %d results", len(results_union)
                    )
                except Exception as exc:  # pragma: no cover - best-effort retry
                    logger.debug("[websearch] No-language retry failed: %s", exc)
            # If still no results, try without categories (let Searx default)
            if not results_union:
                try:
                    logger.info("[websearch] Still no results, retrying with no categories (defaults)")
                    results_union = await run_query(q, lang_norm, [])
                    logger.info(
                        "[websearch] No-categories retry produced %d results", len(results_union)
                    )
                except Exception as exc:  # pragma: no cover - best-effort retry
                    logger.debug("[websearch] No-categories retry failed: %s", exc)
            # And as a final early step, try both no language and no categories together
            if not results_union:
                try:
                    logger.info("[websearch] Final early retry: no language and no categories")
                    results_union = await run_query(q, None, [])
                    logger.info(
                        "[websearch] No-lang+no-cats retry produced %d results", len(results_union)
                    )
                except Exception as exc:  # pragma: no cover - best-effort retry
                    logger.debug("[websearch] No-lang+no-cats retry failed: %s", exc)
        except (RequestException, JSONDecodeError, ValueError) as exc:
            dt = time.perf_counter() - t0
            logger.error("[websearch] node=web_search error=%s dt=%.3fs", type(exc).__name__, dt)
            return {"results": [], "categories": cats, "lang": lang_display}

        # Apply MMR reranking if embedder is available, otherwise use domain diversification
        try:
            cleaned = await diversify_topk_mmr(
                dedupe_results(normalize_urls(results_union)),
                k=deps.config.k,
                query=q,
                embedder=deps.embedder,
                lambda_mult=deps.config.mmr_lambda,
                fetch_k=deps.config.mmr_fetch_k,
                use_vectorstore_mmr=deps.config.use_vectorstore_mmr,
            )
        except Exception as exc:
            logger.warning("[websearch] MMR reranking failed: %s, using fallback", exc)
            cleaned = diversify_topk(dedupe_results(normalize_urls(results_union)), k=deps.config.k)

        # If we still have no cleaned results but received raw results, try a last-ditch retry
        # without specifying a language filter. Some engines behave better without 'language'.
        if results_union and not cleaned:
            try:
                logger.info("[websearch] Retry without language filter as cleaned results are empty")
                results_union = await run_query(q, None, cats)
                cleaned = diversify_topk(dedupe_results(normalize_urls(results_union)), k=deps.config.k)
                logger.info("[websearch] Retry without language filter produced %d results", len(cleaned))
            except Exception as exc:  # pragma: no cover - best-effort retry
                logger.debug("[websearch] Retry without language filter failed: %s", exc)

        # If still empty, attempt alternative language fallbacks (en/pt) best-effort.
        # This helps when detection misfires (e.g., Portuguese culinary queries without diacritics).
        if not cleaned:
            alt_langs: list[str] = []
            # Prefer trying English and Portuguese explicitly
            for code in ("en", "pt"):
                if lang_norm != code:
                    alt_langs.append(code)

            for alt in alt_langs:
                try:
                    alt_results = await run_query(q, alt, cats)
                    results_union += alt_results
                except Exception:  # pragma: no cover - best-effort
                    pass

            if results_union and not cleaned:
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
            # Provide more context: how many raw results actually have a usable URL after normalization
            with_link = sum(1 for r in results_union if r.get("link"))
            total = len(results_union)
            msg = (
                "[websearch] Raw results present but cleaned is empty. "
                f"with_link={with_link}/{total} sample_keys={list(results_union[0].keys()) if results_union else 'N/A'}"
            )
            if with_link == 0:
                # Likely instant answers/special results without URLs from some engines
                logger.info(msg)
            else:
                logger.warning(msg)

        return {"results": cleaned, "categories": cats, "lang": lang_display}

    return RunnableLambda(_search)


__all__ = ["build_web_search_node"]

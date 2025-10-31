"""Web search node builder for the websearch agent."""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass
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


@dataclass(slots=True)
class _LanguagePlan:
    lang: Optional[str]
    lang_norm: Optional[str]
    likely_non_en: bool
    pivot_enabled: bool

    @property
    def should_pivot(self) -> bool:
        lang = self.lang or "auto"
        return self.pivot_enabled and lang != "en" and self.likely_non_en

    @property
    def display(self) -> str:
        return self.lang_norm or "auto"


def _likely_non_english(text: str) -> bool:
    lowered = text.lower()
    if re.search(r"[^\x00-\x7F]", text):
        return True
    keywords = (
        "quando",
        "próximo",
        "próxima",
        "ação",
        "notícia",
        "pesquisa",
    )
    return any(word in lowered for word in keywords)


def build_web_search_node(deps: NodeDependencies) -> RunnableLambda:
    """Return the web search node runnable."""

    search_client = deps.search_wrapper_factory()
    _sema = asyncio.Semaphore(max(1, deps.config.searx_max_concurrency))

    async def searx_call_with_retry(call: Callable[[], list[dict[str, Any]]]) -> list[dict[str, Any]]:
        """Call Searx with retries and jittered backoff."""

        for attempt in range(deps.config.retries + 1):
            try:
                async with _sema:
                    return await asyncio.to_thread(call)
            except (RequestException, JSONDecodeError, ValueError) as exc:
                if attempt >= deps.config.retries:
                    raise
                delay = deps.config.backoff_base * (2 ** attempt)
                delay *= 0.8 + random.random() * 0.4
                logger.warning(
                    "[websearch] retry=%d delay=%.2fs error=%s",
                    attempt + 1,
                    delay,
                    type(exc).__name__,
                )
                await asyncio.sleep(delay)
        return []

    def build_kwargs(lang_code: Optional[str], categories: list[str]) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "num_results": deps.config.k * 2,
            "safesearch": deps.config.safesearch,
        }
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

    def prepare_categories(raw: Optional[list[str]]) -> list[str]:
        base = raw or ["general"]
        seen: set[str] = set()
        ordered: list[str] = []
        for cat in [*base, "general"]:
            if cat and cat not in seen:
                ordered.append(cat)
                seen.add(cat)
        return ordered

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

    async def resolve_language(query: str, state_lang: Optional[str]) -> _LanguagePlan:
        lang_cfg = (deps.config.lang or "auto").strip().lower()
        lang: Optional[str] = lang_cfg
        if lang_cfg in {"", "auto"}:
            detected = state_lang or await deps.lang_detector.detect(query)
            if detected:
                lang = detected
        lang_norm = to_searx_locale(lang)
        pivot_enabled = bool(getattr(deps.config, "pivot_to_english", True))
        return _LanguagePlan(
            lang=lang,
            lang_norm=lang_norm,
            likely_non_en=_likely_non_english(query),
            pivot_enabled=pivot_enabled,
        )

    async def _search(state: SearchState, config: RunnableConfig | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        q = state["query"].strip()
        q_en = (state.get("query_en") or q).strip()
        cats = prepare_categories(state.get("categories"))

        plan = await resolve_language(q, state.get("lang"))
        pivot_label = "en" if plan.should_pivot else "none"
        logger.info(
            "[websearch] pivot_check: pivot_enabled=%s lang=%s likely_non_en=%s will_pivot=%s",
            plan.pivot_enabled,
            plan.lang,
            plan.likely_non_en,
            plan.should_pivot,
        )

        pivot_translation: Optional[str] = q_en if q_en else None

        async def primary_query(active_categories: list[str]) -> list[dict[str, Any]]:
            nonlocal pivot_translation
            if plan.should_pivot:
                if not pivot_translation:
                    pivot_translation = await deps.translate_query(q, "en")
                raw_orig, raw_en = await asyncio.gather(
                    run_query(q, plan.lang_norm, active_categories),
                    run_query(pivot_translation, "en", active_categories),
                )
                return raw_orig + raw_en
            return await run_query(q, plan.lang_norm, active_categories)

        async def gather_results(active_categories: list[str]) -> list[dict[str, Any]]:
            results = await primary_query(active_categories)
            if not results:
                try:
                    logger.info("[websearch] No results from initial query, retrying without language filter")
                    results = await run_query(q, None, active_categories)
                    logger.info("[websearch] No-language retry produced %d results", len(results))
                except Exception as exc:  # pragma: no cover - best-effort retry
                    logger.debug("[websearch] No-language retry failed: %s", exc)
            if not results:
                try:
                    logger.info("[websearch] Still no results, retrying with no categories (defaults)")
                    results = await run_query(q, plan.lang_norm, [])
                    logger.info("[websearch] No-categories retry produced %d results", len(results))
                except Exception as exc:  # pragma: no cover - best-effort retry
                    logger.debug("[websearch] No-categories retry failed: %s", exc)
            if not results:
                try:
                    logger.info("[websearch] Final early retry: no language and no categories")
                    results = await run_query(q, None, [])
                    logger.info("[websearch] No-lang+no-cats retry produced %d results", len(results))
                except Exception as exc:  # pragma: no cover - best-effort retry
                    logger.debug("[websearch] No-lang+no-cats retry failed: %s", exc)
            return results

        def normalize_results(res: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return dedupe_results(normalize_urls(res))

        async def rerank_results(
            results_union: list[dict[str, Any]],
            categories: list[str],
        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
            try:
                cleaned = await diversify_topk_mmr(
                    normalize_results(results_union),
                    k=deps.config.k,
                    query=q,
                    embedder=deps.embedder,
                    lambda_mult=deps.config.mmr_lambda,
                    fetch_k=deps.config.mmr_fetch_k,
                    use_vectorstore_mmr=deps.config.use_vectorstore_mmr,
                )
            except Exception as exc:
                logger.warning("[websearch] MMR reranking failed: %s, using fallback", exc)
                cleaned = diversify_topk(normalize_results(results_union), k=deps.config.k)

            if results_union and not cleaned:
                try:
                    logger.info("[websearch] Retry without language filter as cleaned results are empty")
                    results_union = await run_query(q, None, categories)
                    cleaned = diversify_topk(normalize_results(results_union), k=deps.config.k)
                    logger.info("[websearch] Retry without language filter produced %d results", len(cleaned))
                except Exception as exc:  # pragma: no cover - best-effort retry
                    logger.debug("[websearch] Retry without language filter failed: %s", exc)

            if not cleaned:
                alt_langs = [code for code in ("en", "pt") if plan.lang_norm != code]
                for alt in alt_langs:
                    try:
                        alt_results = await run_query(q, alt, categories)
                        results_union += alt_results
                    except Exception:  # pragma: no cover - best-effort
                        pass
                if results_union and not cleaned:
                    cleaned = diversify_topk(normalize_results(results_union), k=deps.config.k)

            if not cleaned and "general" in categories and len(categories) > 1:
                logger.warning("[websearch] No results with categories %s, retrying without 'general'", categories)
                fallback_cats = [c for c in categories if c != "general"]
                if fallback_cats:
                    try:
                        results_union = await primary_query(fallback_cats)
                        cleaned = diversify_topk(normalize_results(results_union), k=deps.config.k)
                        categories = fallback_cats
                        logger.info(
                            "[websearch] Fallback with cats=%s produced %d results",
                            fallback_cats,
                            len(cleaned),
                        )
                    except Exception as exc:  # pragma: no cover - fallback best-effort
                        logger.warning("[websearch] Fallback failed: %s", exc)

            return cleaned, results_union, categories

        try:
            results_union = await gather_results(cats)
        except (RequestException, JSONDecodeError, ValueError) as exc:
            dt = time.perf_counter() - t0
            logger.error("[websearch] node=web_search error=%s dt=%.3fs", type(exc).__name__, dt)
            return {"results": [], "categories": cats, "lang": plan.display}

        cleaned, results_union, cats = await rerank_results(results_union, cats)

        dt = time.perf_counter() - t0
        logger.info(
            "[websearch] node=web_search dt=%.3fs q=%r n_raw=%d n_clean=%d lang=%s pivot=%s",
            dt,
            q,
            len(results_union),
            len(cleaned),
            plan.display,
            pivot_label,
        )

        if results_union and not cleaned:
            with_link = sum(1 for r in results_union if r.get("link"))
            total = len(results_union)
            msg = (
                "[websearch] Raw results present but cleaned is empty. "
                f"with_link={with_link}/{total} sample_keys={list(results_union[0].keys()) if results_union else 'N/A'}"
            )
            if with_link == 0:
                logger.info(msg)
            else:
                logger.warning(msg)

        return {"results": cleaned, "categories": cats, "lang": plan.display}

    return RunnableLambda(_search)


__all__ = ["build_web_search_node"]

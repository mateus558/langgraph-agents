"""Categorization node builder for the websearch agent."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from websearch.constants import ALLOWED_CATEGORIES
from websearch.heuristics import CategoryResponse, heuristic_categories
from websearch.config import SearchState

from .shared import NodeDependencies

logger = logging.getLogger(__name__)


_SYS_PROMPT = SystemMessage(
    content=(
        "You receive a query and must choose the most relevant Searx categories. "
        f"Allowed set: {', '.join(ALLOWED_CATEGORIES)}. "
        "Reply only with JSON exactly compatible with the requested schema."
    )
)

_FEWSHOT = [
    HumanMessage(content='QUERY: "breaking microsoft acquisition"\nHINTS: ["news","economics"]'),
    AIMessage(content='{"categories":["news","economics"]}'),
    HumanMessage(content='QUERY: "github actions cache permission denied"\nHINTS: ["news"]'),
    AIMessage(content='{"categories":["it"]}'),
    HumanMessage(content='QUERY: "miscellaneous things without context"\nHINTS: []'),
    AIMessage(content='{"categories":["general"]}'),
    HumanMessage(content='QUERY: "dune 2 movie trailer"\nHINTS: ["videos"]'),
    AIMessage(content='{"categories":["videos"]}'),
]


async def _structured_categories(model, query_en: str, hints: list[str], limit: int) -> list[str]:
    model_struct = model.with_structured_output(CategoryResponse)
    user_msg = HumanMessage(content=f'QUERY: "{query_en}"\nHINTS: {json.dumps(hints, ensure_ascii=False)}')
    try:
        resp = await model_struct.ainvoke([_SYS_PROMPT, *_FEWSHOT, user_msg])  # type: ignore[attr-defined]
    except AttributeError:
        resp = await asyncio.to_thread(model_struct.invoke, [_SYS_PROMPT, *_FEWSHOT, user_msg])
    cats = (getattr(resp, "categories", None) or [])
    cats = [c for c in cats if c in ALLOWED_CATEGORIES]
    cats = (cats or ["general"])[: max(1, limit)]
    return cats


def build_categorize_node(deps: NodeDependencies) -> RunnableLambda:
    """Return the categorize node runnable with provided dependencies."""

    async def _categorize(state: SearchState, config: RunnableConfig | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]

        lang = await deps.lang_detector.detect(query) or "en"
        query_en = query if lang == "en" else await deps.translate_query(query, "en")

        hints_orig = heuristic_categories(query)
        hints_en = heuristic_categories(query_en) if query_en != query else []
        hints = list({*hints_orig, *hints_en})

        cats: list[str]
        model = deps.get_model()
        if model is None:
            cats = ["general"]
        else:
            try:
                cats = await _structured_categories(model, query_en, hints, deps.config.max_categories)
            except Exception as exc:  # pragma: no cover - structured output optional
                logger.debug("Structured categorize failed: %r", exc)
                cats = ["general"]

        dt = time.perf_counter() - t0
        logger.info("[websearch] node=categorize_query dt=%.3fs cats=%s lang=%s", dt, cats, lang)
        return {"categories": cats, "lang": lang, "query_en": query_en}

    return RunnableLambda(_categorize)


__all__ = ["build_categorize_node"]

"""Summarization node builder for the websearch agent."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableLambda

from websearch.config import SearchState
from websearch.utils import canonical_url

from websearch.prompts import build_websearch_summary_messages
from core.time import build_web_time_vars

from .shared import NodeDependencies

logger = logging.getLogger(__name__)


def _strip_punct(url: str) -> str:
    return url.rstrip(").,;:!?]")


def build_summarize_node(deps: NodeDependencies) -> RunnableLambda:
    """Return the summarize node runnable."""

    async def _summarize(state: SearchState, config: RunnableConfig | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        query = state["query"]
        results = state.get("results") or []

        if not results:
            dt = time.perf_counter() - t0
            logger.info("[websearch] node=summarize dt=%.3fs (no results)", dt)
            return {"summary": f"No results found for: {query}"}

        tz = await deps.get_local_tz()
        tv = build_web_time_vars(tz)

        urls = [str(r.get("link", "")) for r in results if isinstance(r.get("link"), str)]
        items: list[str] = []
        for idx, result in enumerate(results, start=1):
            title = str(result.get("title") or "").strip()
            link = str(result.get("link") or "").strip()
            snippet = str(result.get("snippet") or "").strip()
            if link:
                items.append(f"[{idx}] {title}\nURL: {link}\nSnippet: {snippet}")

        whitelist_msg = (
            "ALLOWED SOURCES (cite only these, using their numeric index):\n"
            + "\n".join(f"[{i+1}] {u}" for i, u in enumerate(urls))
        )

        messages_payload = build_websearch_summary_messages(
            query=query,
            whitelist=whitelist_msg,
            results="\n\n".join(items),
            utc_time=tv["utc_time"],
            local_time=tv["local_time"],
            local_label=tv["local_label"],
        )

        model = deps.get_model()
        if not model:
            top = "\n".join(items[:3])
            dt = time.perf_counter() - t0
            logger.info("[websearch] node=summarize dt=%.3fs (fallback)", dt)
            return {"summary": f"(Fallback without LLM)\n{top}"}

        out = await deps.call_llm(list(messages_payload), 90.0)
        text = (getattr(out, "content", "") or "").strip()

        safe = {canonical_url(u) for u in urls}
        for token in re.findall(r"https?://\S+", text):
            token_norm = _strip_punct(token)
            if canonical_url(token_norm) not in safe:
                text = text.replace(token, "")

        dt = time.perf_counter() - t0
        logger.info("[websearch] node=summarize dt=%.3fs", dt)
        return {"summary": text}

    return RunnableLambda(_summarize)


__all__ = ["build_summarize_node"]

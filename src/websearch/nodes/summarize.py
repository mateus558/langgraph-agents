"""Summarization node builder for the websearch agent."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda

from websearch.config import SearchState
from websearch.utils import canonical_url

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

        now_utc = datetime.now(timezone.utc)
        tz = await deps.get_local_tz()
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

        prompt = (
            f"{whitelist_msg}\n\n"
            "TASK: Summarize and answer the query using only the listed sources.\n"
            "MANDATORY OUTPUT FORMAT:\n"
            "Summary:\n"
            "- Start with: 'As of <YYYY-MM-DD HH:MM UTC / local>:'\n"
            "- If the user query is time related, include the next upcoming or most recent relevant date strictly compared to the current time.\n"
            "- Only add information that is relevant to the user query.\n"
            "Details (optional):\n"
            "- Add factual context such as periodicity, duration, or schedule patterns when the user query suggests so.\n"
            "Sources:\n"
            "- List only the citation indices used, ordered by importance (e.g., [1], [4], [2]).\n\n"
            "CITATION AND CONFLICT RULES:\n"
            "- Every date or quantitative claim must have at least one citation [n].\n"
            "- When multiple sources disagree, identify the conflict and prefer the newest data.\n"
            "- If the user query has time relationships: If no upcoming or relevant date is found, explicitly say: 'No future or current date found in the results.'\n\n"
            f"Query: {query}\n\n"
            "RESULTS (to base your answer on):\n" + "\n\n".join(items)
        )

        model = deps.get_model()
        if not model:
            top = "\n".join(items[:3])
            dt = time.perf_counter() - t0
            logger.info("[websearch] node=summarize dt=%.3fs (fallback)", dt)
            return {"summary": f"(Fallback without LLM)\n{top}"}

        out = await deps.call_llm([sys, HumanMessage(content=prompt)], 90.0)
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

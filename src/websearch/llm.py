"""LLM helper utilities for the websearch agent."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


async def call_llm_safely(model: BaseChatModel, msgs: list[Any], timeout: float = 90.0):
    """Invoke the underlying chat model without blocking the event loop."""

    provider = getattr(model, "model_provider", "") or getattr(getattr(model, "client", None), "__class__", type("", (), {}))
    provider_name = str(getattr(provider, "__name__", provider)).lower()

    async def _try_async():
        if hasattr(model, "ainvoke") and provider_name in {"openai", "anthropic", "google"}:
            return await model.ainvoke(msgs)
        return await asyncio.to_thread(model.invoke, msgs)

    return await asyncio.wait_for(_try_async(), timeout=timeout)


async def translate_for_search(
    text: str,
    target_lang: str,
    call_llm: Callable[[list[Any], float], Awaitable[Any]],
) -> str:
    """Translate text to the given language using the provided LLM callable."""

    text = (text or "").strip()
    if not text or target_lang.lower() == "auto":
        return text

    sys = SystemMessage(
        content=(
            "Translate the user's query for web search categorization. "
            "Output ONLY the translation as plain text; no quotes, no extra text."
        )
    )
    human = HumanMessage(content=f"Target language: {target_lang}\nText:\n{text}")

    try:
        out = await call_llm([sys, human], 20.0)
    except Exception as exc:  # pragma: no cover - translation is best-effort
        logger.debug("Translation failed: %s", exc)
        return text

    translated = (getattr(out, "content", "") or "").strip()
    if len(translated) >= 2 and translated[0] in {'"', "'"} and translated[-1] == translated[0]:
        translated = translated[1:-1].strip()

    return translated or text


__all__ = ["call_llm_safely", "translate_for_search"]

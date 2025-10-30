"""Shared types for websearch node builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol
from zoneinfo import ZoneInfo

from langchain_core.language_models import BaseChatModel

from websearch.config import SearchAgentConfig
from websearch.language import LangDetector


class SupportsSearch(Protocol):
    def results(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:  # pragma: no cover - interface contract
        ...


@dataclass(slots=True)
class NodeDependencies:
    """Container for cross-cutting dependencies used by the graph nodes."""

    config: SearchAgentConfig
    lang_detector: LangDetector
    translate_query: Callable[[str, str], Awaitable[str]]
    call_llm: Callable[[list[Any], float], Awaitable[Any]]
    get_model: Callable[[], BaseChatModel | None]
    get_local_tz: Callable[[], Awaitable[ZoneInfo]]
    search_wrapper_factory: Callable[[], SupportsSearch]
    embedder: object | None = None


__all__ = ["NodeDependencies", "SupportsSearch"]

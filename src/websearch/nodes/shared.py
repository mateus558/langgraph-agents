"""Shared types for websearch node builders."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import tzinfo
from typing import Any, Awaitable, Callable, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from websearch.ranking import Reranker, SupportsEmbedder

from websearch.config import SearchAgentConfig
from websearch.language import LangDetector


class SupportsSearch(Protocol):
    def results(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:  # pragma: no cover - interface contract
        ...


class SupportsEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...  # pragma: no cover - protocol
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...  # pragma: no cover - protocol


@dataclass(slots=True)
class NodeDependencies:
    """Container for cross-cutting dependencies used by the graph nodes."""

    config: SearchAgentConfig
    lang_detector: LangDetector
    translate_query: Callable[[str, str], Awaitable[str]]
    call_llm: Callable[[list[BaseMessage], float], Awaitable[BaseMessage]]
    get_model: Callable[[], BaseChatModel | None]
    local_tz: tzinfo
    search_wrapper_factory: Callable[[], SupportsSearch]
    embedder: SupportsEmbedder | None = None
    reranker: Reranker | None = None

__all__ = ["NodeDependencies", "SupportsSearch", "SupportsEmbedder"]

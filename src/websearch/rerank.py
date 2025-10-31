"""Reusable reranking helpers for web search results."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from websearch.utils import (
    dedupe_results,
    diversify_topk,
    diversify_topk_mmr,
    normalize_urls,
)

logger = logging.getLogger(__name__)


class SupportsEmbedder(Protocol):
    """Protocol for embedder objects used by MMR reranking."""

    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    """Minimal interface for result rerankers."""

    async def rerank(self, *, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]: ...


def _normalize_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply URL normalization and deduplication before reranking."""

    return dedupe_results(normalize_urls(results))


@dataclass(slots=True)
class MMRReranker:
    """MMR-based reranker with graceful fallback to domain diversification."""

    embedder: SupportsEmbedder | None
    use_vectorstore: bool
    lambda_mult: float
    fetch_k: int

    async def rerank(self, *, query: str, results: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:  # type: ignore[override]
        if not results:
            return []

        normalized = _normalize_results(results)
        try:
            cleaned = await diversify_topk_mmr(
                normalized,
                k=min(k, len(normalized)),
                query=query,
                embedder=self.embedder,
                lambda_mult=self.lambda_mult,
                fetch_k=max(1, min(self.fetch_k, len(normalized))),
                use_vectorstore_mmr=self.use_vectorstore,
            )
        except Exception as exc:
            logger.warning("[rerank] MMR failed (%s), using domain diversification", exc)
            cleaned = diversify_topk(normalized, k=min(k, len(normalized)))

        if not cleaned:
            cleaned = diversify_topk(normalized, k=min(k, len(normalized)))
        return cleaned


__all__ = ["Reranker", "MMRReranker", "SupportsEmbedder"]


"""Reusable reranking helpers for web search results."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, cast

import asyncio
from urllib.parse import urlparse

from websearch.utils import dedupe_results, normalize_urls

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


__all__ = ["Reranker", "MMRReranker", "SupportsEmbedder", "diversify_topk", "diversify_topk_mmr"]


def diversify_topk(results: list[dict], k: int) -> list[dict]:
    """Diversify results across domains using round-robin selection."""

    by_domain: dict[str, list[dict]] = {}
    for result in results:
        domain = urlparse(result.get("link", "")).netloc
        by_domain.setdefault(domain, []).append(result)

    picked: list[dict] = []
    domains = list(by_domain.items())
    index = 0

    while len(picked) < k and any(bucket for _, bucket in domains):
        domain, bucket = domains[index % len(domains)]
        if bucket:
            picked.append(bucket.pop(0))
        index += 1
        domains = [(d, b) for d, b in domains if b]
        if not domains:
            break
    return picked


async def diversify_topk_mmr(
    results: list[dict],
    k: int,
    query: str,
    embedder: SupportsEmbedder | None = None,
    lambda_mult: float = 0.55,
    fetch_k: int = 50,
    use_vectorstore_mmr: bool = True,
) -> list[dict]:
    """Diversify results using Maximal Marginal Relevance with fallbacks."""

    if not results:
        return []

    fetch_k = min(fetch_k, len(results))
    k = min(k, len(results))

    if embedder is None:
        logger.debug("[diversify_topk_mmr] No embedder available, using domain-based fallback")
        return diversify_topk(results, k)

    try:
        if use_vectorstore_mmr:
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_core.documents import Document

                def _faiss_mmr() -> list[dict]:
                    docs = []
                    for r in results[:fetch_k]:
                        title = r.get("title", "")
                        snippet = r.get("snippet", "")
                        link = r.get("link", "")
                        content = f"{title}\n{snippet}".strip()
                        if content:
                            docs.append(Document(page_content=content, metadata={"link": link, "result": r}))

                    if not docs:
                        return []

                    vectorstore = FAISS.from_documents(docs, embedder)  # type: ignore[arg-type]
                    mmr_docs = vectorstore.max_marginal_relevance_search(
                        query, k=k, fetch_k=len(docs), lambda_mult=lambda_mult
                    )
                    return [doc.metadata["result"] for doc in mmr_docs]

                reranked = await asyncio.to_thread(_faiss_mmr)
                logger.debug(
                    "[diversify_topk_mmr] FAISS MMR completed: %d results from %d candidates",
                    len(reranked),
                    fetch_k,
                )
                return reranked

            except ImportError:
                logger.debug("[diversify_topk_mmr] FAISS not available, trying standalone MMR")
            except Exception as exc:
                logger.warning("[diversify_topk_mmr] FAISS MMR failed: %s, trying standalone MMR", exc)

        from langchain_core.vectorstores.utils import maximal_marginal_relevance

        def _standalone_mmr() -> list[dict]:
            texts = []
            for r in results[:fetch_k]:
                title = r.get("title", "")
                snippet = r.get("snippet", "")
                content = f"{title}\n{snippet}".strip()
                if content:
                    texts.append(content)

            if not texts:
                return []

            query_embedding = embedder.embed_query(query)  # type: ignore[attr-defined]
            doc_embeddings = embedder.embed_documents(texts)  # type: ignore[attr-defined]

            try:
                import numpy as np

                query_embedding_np = np.array(query_embedding)
                doc_embeddings_np = np.array(doc_embeddings)
            except ImportError:
                query_embedding_np = query_embedding
                doc_embeddings_np = doc_embeddings

            def _ensure_list(value: Any) -> Any:
                return value.tolist() if hasattr(value, "tolist") else value

            query_for_mmr = _ensure_list(query_embedding_np)
            docs_for_mmr = _ensure_list(doc_embeddings_np)

            if not isinstance(query_for_mmr, list):
                query_for_mmr = list(query_for_mmr)
            if not isinstance(docs_for_mmr, list):
                docs_for_mmr = list(docs_for_mmr)

            indices = maximal_marginal_relevance(
                cast(Any, query_for_mmr),
                cast(Any, docs_for_mmr),
                k=k,
                lambda_mult=lambda_mult,
            )

            return [results[i] for i in indices if i < len(results)]

        reranked = await asyncio.to_thread(_standalone_mmr)
        logger.debug(
            "[diversify_topk_mmr] Standalone MMR completed: %d results from %d candidates",
            len(reranked),
            fetch_k,
        )
        return reranked

    except Exception as exc:
        logger.warning("[diversify_topk_mmr] Unexpected error: %s, using fallback", exc)

    logger.debug("[diversify_topk_mmr] Using domain-based diversification fallback")
    return diversify_topk(results, k)

from __future__ import annotations

import pytest

from websearch.ranking import MMRReranker


class _MockEmbedder:
    def embed_query(self, text: str) -> list[float]:  # pragma: no cover - simple mock
        return [float(len(text))]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover - simple mock
        return [[float(len(t))] for t in texts]


@pytest.mark.asyncio
async def test_mmrreranker_uses_mmr_when_available(monkeypatch):
    outputs: list[list[dict]] = []

    async def fake_diversify_mmr(results, **kwargs):  # type: ignore[override]
        outputs.append(results)
        return list(reversed(results))

    monkeypatch.setattr("websearch.ranking.reranker.diversify_topk_mmr", fake_diversify_mmr)

    reranker = MMRReranker(embedder=_MockEmbedder(), use_vectorstore=True, lambda_mult=0.5, fetch_k=10)
    items = [{"link": "https://a"}, {"link": "https://b"}]

    cleaned = await reranker.rerank(query="hello", results=items, k=2)

    assert cleaned == list(reversed(items))
    assert outputs and outputs[0][0]["link"] == "https://a"


@pytest.mark.asyncio
async def test_mmrreranker_fallback_when_mmr_fails(monkeypatch):
    async def boom(*_args, **_kwargs):  # type: ignore[override]
        raise RuntimeError("fail")

    monkeypatch.setattr("websearch.ranking.reranker.diversify_topk_mmr", boom)

    def fake_diversify(results, k):  # type: ignore[override]
        return results[:k]

    monkeypatch.setattr("websearch.ranking.reranker.diversify_topk", fake_diversify)

    reranker = MMRReranker(embedder=None, use_vectorstore=False, lambda_mult=0.5, fetch_k=5)
    items = [{"link": "https://a"}, {"link": "https://a"}, {"link": "https://b"}]

    cleaned = await reranker.rerank(query="q", results=items, k=2)

    # Deduplication should drop duplicate URLs and fallback should select first two unique
    assert len(cleaned) == 2
    assert cleaned[0]["link"] == "https://a"
    assert cleaned[1]["link"] == "https://b"

"""Unit tests for websearch utility and ranking helpers."""

from __future__ import annotations

import pytest

from src.websearch.utils import canonical_url, normalize_urls, dedupe_results, to_searx_locale
from src.websearch.ranking import diversify_topk, diversify_topk_mmr


def test_canonical_url_basic():
    u = "http://www.example.com/path?utm_source=foo&fbclid=bar&id=1#section"
    out = canonical_url(u)
    assert out == "https://example.com/path?id=1"


def test_normalize_urls_standard_and_result_dict():
    results = [
        {"url": "http://www.example.com/a?utm_campaign=x"},
        {"Result": {"url": "https://site.com/b", "title": "B", "snippet": "snip"}},
    ]
    normalize_urls(results)
    assert results[0]["link"] == "https://example.com/a"
    assert results[1]["link"] == "https://site.com/b"
    assert results[1]["title"] == "B"
    assert results[1]["snippet"] == "snip"


def test_normalize_urls_result_string():
    results = [{"Result": "https://foo.bar/c?gclid=123"}]
    normalize_urls(results)
    assert results[0]["link"] == "https://foo.bar/c"


def test_dedupe_results():
    rs = [
        {"link": "https://a.com/x"},
        {"link": "https://a.com/x"},
        {"link": "https://b.com/y"},
        {},  # missing link
    ]
    out = dedupe_results(rs)
    assert [r.get("link") for r in out] == ["https://a.com/x", "https://b.com/y"]


def test_diversify_topk_domain_rotation():
    rs = [
        {"link": "https://a.com/1"},
        {"link": "https://a.com/2"},
        {"link": "https://b.com/3"},
        {"link": "https://b.com/4"},
        {"link": "https://c.com/5"},
    ]
    out = diversify_topk(rs, k=4)
    # Round-robin should pick a,b,c, then back to a or b depending on list order
    assert len(out) == 4
    assert out[0]["link"].startswith("https://a.com/")
    assert out[1]["link"].startswith("https://b.com/")
    assert out[2]["link"].startswith("https://c.com/")


def test_to_searx_locale():
    assert to_searx_locale(None) is None
    assert to_searx_locale("") is None
    assert to_searx_locale("auto") is None
    assert to_searx_locale("en") == "en"
    assert to_searx_locale("en-US") == "en-US"
    assert to_searx_locale("pt_br") == "pt-BR"


# ============================================================================
# MMR Reranking Tests
# ============================================================================


@pytest.mark.asyncio
async def test_diversify_topk_mmr_no_embedder():
    """Test MMR fallback to domain diversification when no embedder available."""
    results = [
        {"link": "https://a.com/1", "title": "A1", "snippet": "First result from A"},
        {"link": "https://a.com/2", "title": "A2", "snippet": "Second result from A"},
        {"link": "https://b.com/3", "title": "B3", "snippet": "Result from B"},
    ]
    
    out = await diversify_topk_mmr(
        results=results,
        k=3,
        query="test query",
        embedder=None,
        lambda_mult=0.5,
        fetch_k=10,
        use_vectorstore_mmr=True,
    )
    
    # Should fallback to domain diversification
    assert len(out) == 3
    # First three should be one from each domain (round-robin)
    assert out[0]["link"].startswith("https://a.com/")
    assert out[1]["link"].startswith("https://b.com/")


@pytest.mark.asyncio
async def test_diversify_topk_mmr_empty_results():
    """Test MMR with empty input."""
    out = await diversify_topk_mmr(
        results=[],
        k=5,
        query="test",
        embedder=None,
        lambda_mult=0.5,
        fetch_k=10,
    )
    assert out == []


@pytest.mark.asyncio
async def test_diversify_topk_mmr_k_larger_than_results():
    """Test MMR when k is larger than available results."""
    results = [
        {"link": "https://a.com/1", "title": "A", "snippet": "Content A"},
        {"link": "https://b.com/2", "title": "B", "snippet": "Content B"},
    ]
    
    out = await diversify_topk_mmr(
        results=results,
        k=10,
        query="test",
        embedder=None,
        lambda_mult=0.5,
        fetch_k=20,
    )
    
    # Should return all available results
    assert len(out) == 2


class MockEmbedder:
    """Mock embedder for testing standalone MMR path."""
    
    def embed_query(self, text: str) -> list[float]:
        """Return deterministic embedding based on text content."""
        # Simple hash-based embedding
        return [float(hash(text) % 100) / 100.0 for _ in range(8)]
    
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings for documents."""
        return [self.embed_query(t) for t in texts]


@pytest.mark.asyncio
async def test_diversify_topk_mmr_with_mock_embedder():
    """Test MMR with mock embedder (standalone path, no FAISS)."""
    results = [
        {"link": "https://a.com/1", "title": "Python Tutorial", "snippet": "Learn Python programming"},
        {"link": "https://b.com/2", "title": "Python Guide", "snippet": "Guide to Python language"},
        {"link": "https://c.com/3", "title": "Java Tutorial", "snippet": "Learn Java programming"},
        {"link": "https://d.com/4", "title": "JavaScript Guide", "snippet": "Web development with JS"},
    ]
    
    embedder = MockEmbedder()
    
    out = await diversify_topk_mmr(
        results=results,
        k=3,
        query="Python programming",
        embedder=embedder,
        lambda_mult=0.6,  # Favor relevance
        fetch_k=4,
        use_vectorstore_mmr=False,  # Force standalone MMR path
    )
    
    # Should return 3 results
    assert len(out) == 3
    # Results should be reordered (order may vary based on MMR)
    assert all(r in results for r in out)


@pytest.mark.asyncio
async def test_diversify_topk_mmr_lambda_extremes():
    """Test MMR with extreme lambda values."""
    results = [
        {"link": "https://a.com/1", "title": "Python", "snippet": "Python content"},
        {"link": "https://b.com/2", "title": "Python", "snippet": "More Python"},
        {"link": "https://c.com/3", "title": "Java", "snippet": "Java content"},
    ]
    
    embedder = MockEmbedder()
    
    # Max diversity (lambda=0)
    out_diverse = await diversify_topk_mmr(
        results=results,
        k=2,
        query="Python",
        embedder=embedder,
        lambda_mult=0.0,
        fetch_k=3,
        use_vectorstore_mmr=False,
    )
    assert len(out_diverse) == 2
    
    # Max relevance (lambda=1)
    out_relevant = await diversify_topk_mmr(
        results=results,
        k=2,
        query="Python",
        embedder=embedder,
        lambda_mult=1.0,
        fetch_k=3,
        use_vectorstore_mmr=False,
    )
    assert len(out_relevant) == 2


@pytest.mark.asyncio
async def test_diversify_topk_mmr_missing_fields():
    """Test MMR gracefully handles results with missing title/snippet."""
    results = [
        {"link": "https://a.com/1"},  # No title/snippet
        {"link": "https://b.com/2", "title": "B"},  # No snippet
        {"link": "https://c.com/3", "snippet": "C snippet"},  # No title
        {"link": "https://d.com/4", "title": "D", "snippet": "D content"},
    ]
    
    embedder = MockEmbedder()
    
    out = await diversify_topk_mmr(
        results=results,
        k=3,
        query="test",
        embedder=embedder,
        lambda_mult=0.5,
        fetch_k=4,
        use_vectorstore_mmr=False,
    )
    
    # Should handle gracefully and return results
    assert len(out) <= 3
    assert all("link" in r for r in out)

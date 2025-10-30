"""Utility functions for URL handling and result processing.

This module provides utilities for:
- URL normalization and canonicalization
- Search result deduplication
- Result diversification across domains
- MMR-based reranking for relevance and diversity
- Time range policy selection
"""

import asyncio
import logging
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)


def canonical_url(url: str) -> str:
    """Convert URL to canonical form.

    Canonicalization includes:
    - Force HTTPS for HTTP(S) URLs
    - Remove www. prefix from domain
    - Remove tracking parameters (utm_*, fbclid, gclid)
    - Remove URL fragments

    Args:
        url: URL string to canonicalize.

    Returns:
        Canonicalized URL string. Returns original URL if parsing fails.
    """
    try:
        p = urlparse(url)
        scheme = "https" if p.scheme in {"http", "https"} else p.scheme
        netloc = p.netloc.replace("www.", "")
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
             if not k.lower().startswith(("utm_", "fbclid", "gclid"))]
        return urlunparse((scheme, netloc, p.path, "", urlencode(q), ""))
    except Exception:
        return url


def normalize_urls(results: list[dict]) -> list[dict]:
    """Normalize URLs in search results.

    Applies canonical_url() to a URL field and ensures it is in 'link'.
    Many SearxNG engines return different field names for URLs:
    - Standard: 'link', 'url', 'href'
    - Instant answers/special results: 'Result' (often contains a nested dict)
    
    We coerce whichever is present into 'link' for downstream processing.
    If 'Result' is found, we try to extract a URL from it.

    Modifies results in place.

    Args:
        results: List of result dictionaries possibly containing a URL field.

    Returns:
        The same list with normalized URLs in 'link' (modified in place).
    """
    import logging
    logger = logging.getLogger(__name__)
    
    import re
    url_re = re.compile(r"https?://[^\s)\]}>'\"]+", re.IGNORECASE)

    for r in results:
        # Standard fields
        link = r.get("link") or r.get("url") or r.get("href")
        
        # SearxNG instant answer/special result format
        if not link and "Result" in r:
            result_val = r.get("Result")
            # Result might be a dict with URL inside, or a string URL
            if isinstance(result_val, dict):
                logger.debug("normalize_urls: Result is dict with keys: %s", list(result_val.keys()))
                # Common direct URL fields
                link = result_val.get("url") or result_val.get("link") or result_val.get("href")
                # Other occasionally used fields
                if not link:
                    link = result_val.get("source") or result_val.get("permalink") or result_val.get("open_url")
                # Nested structures (e.g., { Result: { data: { url: ... }}})
                if not link and isinstance(result_val.get("data"), dict):
                    data = result_val["data"]
                    link = data.get("url") or data.get("link") or data.get("href")
                # Lists of urls
                if not link and isinstance(result_val.get("urls"), list) and result_val["urls"]:
                    cand = result_val["urls"][0]
                    if isinstance(cand, str) and cand.lower().startswith("http"):
                        link = cand
                # As a last resort, try to extract the first URL from text fields
                if not link:
                    for k in ("content", "snippet", "text", "answer", "body", "description"):
                        val = result_val.get(k)
                        if isinstance(val, str):
                            m = url_re.search(val)
                            if m:
                                link = m.group(0)
                                break
            elif isinstance(result_val, str):
                logger.debug("normalize_urls: Result is string: %s", result_val[:100])
                if result_val.startswith("http"):
                    link = result_val
                else:
                    m = url_re.search(result_val)
                    if m:
                        link = m.group(0)
        
        if link:
            r["link"] = canonical_url(str(link))
            # Also extract title and snippet from Result if present
            if "Result" in r and isinstance(r["Result"], dict):
                if "title" not in r or not r["title"]:
                    r["title"] = r["Result"].get("title", "")
                if "snippet" not in r or not r["snippet"]:
                    r["snippet"] = r["Result"].get("snippet") or r["Result"].get("content", "")
        else:
            # Debug: log results without recognizable URL fields (noise-prone from some Searx engines)
            logger.debug(
                "normalize_urls: result has no extractable URL. Keys: %s, Result type: %s, Result value: %s",
                list(r.keys()), 
                type(r.get("Result")).__name__ if "Result" in r else "N/A",
                str(r.get("Result"))[:200] if "Result" in r else "N/A"
            )
    
    return results


def dedupe_results(results: list[dict]) -> list[dict]:
    """Remove duplicate results by URL.

    Keeps the first occurrence of each unique URL.

    Args:
        results: List of result dictionaries with 'link' field.

    Returns:
        New list with duplicates removed.
    """
    seen = set()
    out = []
    for r in results:
        u = r.get("link", "") or ""
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(r)
    return out


def diversify_topk(results: list[dict], k: int) -> list[dict]:
    """Diversify results across domains.

    Selects top-k results while ensuring diversity by rotating through
    domains. This prevents a single domain from dominating the results.

    Algorithm:
    - Group results by domain
    - Round-robin select one result per domain
    - Continue until k results selected

    Args:
        results: List of result dictionaries with 'link' field.
        k: Number of results to select.

    Returns:
        List of up to k diversified results.
    """
    by_domain: dict[str, list[dict]] = {}
    for r in results:
        dom = urlparse(r.get("link", "")).netloc
        by_domain.setdefault(dom, []).append(r)

    picked: list[dict] = []
    domains = list(by_domain.items())
    i = 0

    while len(picked) < k and any(v for _, v in domains):
        dom, items = domains[i % len(domains)]
        if items:
            picked.append(items.pop(0))
        i += 1
        domains = [(d, v) for d, v in domains if v]
        if not domains:
            break

    return picked


async def diversify_topk_mmr(
    results: list[dict],
    k: int,
    query: str,
    embedder: object | None = None,
    lambda_mult: float = 0.55,
    fetch_k: int = 50,
    use_vectorstore_mmr: bool = True,
) -> list[dict]:
    """Diversify results using MMR (Maximal Marginal Relevance) reranking.

    Three-tier strategy:
    1. If FAISS and embedder available and use_vectorstore_mmr=True: use FAISS MMR
    2. If only embedder available: use standalone MMR utility
    3. Otherwise: fallback to domain-based diversification

    All blocking operations are offloaded to threads to avoid blocking the event loop.

    Args:
        results: List of result dictionaries with 'title', 'snippet', 'link' fields.
        k: Number of final results to return.
        query: The search query for relevance calculation.
        embedder: Optional embeddings model (e.g., HuggingFaceEmbeddings, OpenAIEmbeddings).
        lambda_mult: Balance between relevance (1.0) and diversity (0.0). Default 0.55.
        fetch_k: Number of candidates to consider before MMR filtering. Default 50.
        use_vectorstore_mmr: If True and FAISS available, use FAISS MMR. Default True.

    Returns:
        List of up to k reranked/diversified results.
    """
    if not results:
        return []

    # Limit fetch_k to available results
    fetch_k = min(fetch_k, len(results))
    k = min(k, len(results))

    # If no embedder, fallback to domain diversification
    if embedder is None:
        logger.debug("[diversify_topk_mmr] No embedder available, using domain-based fallback")
        return diversify_topk(results, k)

    try:
        # Try FAISS-based MMR if enabled
        if use_vectorstore_mmr:
            try:
                from langchain_community.vectorstores import FAISS
                from langchain_core.documents import Document

                def _faiss_mmr():
                    # Build documents from results
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

                    # Create FAISS index
                    vectorstore = FAISS.from_documents(docs, embedder)  # type: ignore[arg-type]

                    # Perform MMR search
                    mmr_docs = vectorstore.max_marginal_relevance_search(
                        query, k=k, fetch_k=len(docs), lambda_mult=lambda_mult
                    )

                    # Extract original results
                    return [doc.metadata["result"] for doc in mmr_docs]

                # Run in thread to avoid blocking
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

        # Try standalone MMR utility
        try:
            from langchain_core.vectorstores.utils import maximal_marginal_relevance

            def _standalone_mmr():
                # Prepare texts
                texts = []
                for r in results[:fetch_k]:
                    title = r.get("title", "")
                    snippet = r.get("snippet", "")
                    content = f"{title}\n{snippet}".strip()
                    if content:
                        texts.append(content)

                if not texts:
                    return []

                # Embed query and documents
                query_embedding = embedder.embed_query(query)  # type: ignore[attr-defined]
                doc_embeddings = embedder.embed_documents(texts)  # type: ignore[attr-defined]

                # Convert to numpy arrays if needed (LangChain's MMR expects numpy arrays)
                try:
                    import numpy as np
                    query_embedding_np = np.array(query_embedding)
                    doc_embeddings_np = np.array(doc_embeddings)
                except ImportError:
                    # If numpy not available, try with lists (may fail)
                    query_embedding_np = query_embedding
                    doc_embeddings_np = doc_embeddings

                # Run MMR
                indices = maximal_marginal_relevance(
                    query_embedding_np, doc_embeddings_np, k=k, lambda_mult=lambda_mult  # type: ignore[arg-type]
                )

                # Return reranked results
                return [results[i] for i in indices if i < len(results)]

            reranked = await asyncio.to_thread(_standalone_mmr)
            logger.debug(
                "[diversify_topk_mmr] Standalone MMR completed: %d results from %d candidates",
                len(reranked),
                fetch_k,
            )
            return reranked

        except ImportError:
            logger.debug("[diversify_topk_mmr] Standalone MMR not available, using domain-based fallback")
        except Exception as exc:
            logger.warning("[diversify_topk_mmr] Standalone MMR failed: %s, using fallback", exc)

    except Exception as exc:
        logger.warning("[diversify_topk_mmr] Unexpected error: %s, using fallback", exc)

    # Final fallback: domain-based diversification
    logger.debug("[diversify_topk_mmr] Using domain-based diversification fallback")
    return diversify_topk(results, k)


def pick_time_range(cats: list[str]) -> str | None:
    """Determine time range policy based on categories.

    Different categories benefit from recency filtering:
    - news: recent results preferred (week)
    - it: somewhat recent (month)
    - others: no preference (None)

    Args:
        cats: List of category names.

    Returns:
        Time range string ("day", "week", "month", "year") or None.
    """
    if "news" in cats:
        return "week"
    if "it" in cats:
        return "month"
    return None


def to_searx_locale(code: str | None) -> str | None:
    """Normalize a language code to SearxNG locale format."""

    if not code:
        return None
    c = code.strip()
    if not c:
        return None
    if c.lower() == "auto":
        return None
    c = c.replace("_", "-")
    parts = c.split("-")
    if len(parts) == 1:
        return parts[0].lower()
    lang = parts[0].lower()
    region = parts[1].upper() if parts[1] else ""
    return f"{lang}-{region}" if region else lang

"""Utility functions for URL handling and result processing.

This module provides utilities for:
- URL normalization and canonicalization
- Search result deduplication
- Result diversification across domains
- Time range policy selection
"""

from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


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
    
    for r in results:
        # Standard fields
        link = r.get("link") or r.get("url") or r.get("href")
        
        # SearxNG instant answer/special result format
        if not link and "Result" in r:
            result_val = r.get("Result")
            # Result might be a dict with URL inside, or a string URL
            if isinstance(result_val, dict):
                logger.debug("normalize_urls: Result is dict with keys: %s", list(result_val.keys()))
                link = result_val.get("url") or result_val.get("link") or result_val.get("href")
            elif isinstance(result_val, str):
                logger.debug("normalize_urls: Result is string: %s", result_val[:100])
                if result_val.startswith("http"):
                    link = result_val
        
        if link:
            r["link"] = canonical_url(str(link))
            # Also extract title and snippet from Result if present
            if "Result" in r and isinstance(r["Result"], dict):
                if "title" not in r or not r["title"]:
                    r["title"] = r["Result"].get("title", "")
                if "snippet" not in r or not r["snippet"]:
                    r["snippet"] = r["Result"].get("snippet") or r["Result"].get("content", "")
        else:
            # Debug: log results without recognizable URL fields
            logger.warning(
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

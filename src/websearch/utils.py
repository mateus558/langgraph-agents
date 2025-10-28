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

    Applies canonical_url() to the 'link' field of each result.
    Modifies results in place.

    Args:
        results: List of result dictionaries with 'link' field.

    Returns:
        The same list with normalized URLs (modified in place).
    """
    for r in results:
        if "link" in r and r["link"]:
            r["link"] = canonical_url(r["link"])
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

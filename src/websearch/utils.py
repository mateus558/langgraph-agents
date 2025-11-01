"""Utility functions for URL handling and light post-processing."""

from __future__ import annotations

import logging
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

logger = logging.getLogger(__name__)


def canonical_url(url: str) -> str:
    """Convert URL to canonical form (https, no www, strip tracking params)."""

    try:
        parsed = urlparse(url)
        scheme = "https" if parsed.scheme in {"http", "https"} else parsed.scheme
        netloc = parsed.netloc.replace("www.", "")
        query = [
            (k, v)
            for k, v in parse_qsl(parsed.query, keep_blank_values=True)
            if not k.lower().startswith(("utm_", "fbclid", "gclid"))
        ]
        return urlunparse((scheme, netloc, parsed.path, "", urlencode(query), ""))
    except Exception:
        return url


def normalize_urls(results: list[dict]) -> list[dict]:
    """Normalize URL fields across search results (mutates in place)."""

    import re

    url_re = re.compile(r"https?://[^\s)\]}>'\"]+", re.IGNORECASE)

    for result in results:
        link = result.get("link") or result.get("url") or result.get("href")

        if not link and "Result" in result:
            value = result.get("Result")
            if isinstance(value, dict):
                logger.debug("normalize_urls: Result dict keys: %s", list(value.keys()))
                link = value.get("url") or value.get("link") or value.get("href")
                if not link:
                    link = value.get("source") or value.get("permalink") or value.get("open_url")
                data = value.get("data")
                if not link and isinstance(data, dict):
                    link = data.get("url") or data.get("link") or data.get("href")
                urls = value.get("urls")
                if not link and isinstance(urls, list) and urls:
                    candidate = urls[0]
                    if isinstance(candidate, str) and candidate.lower().startswith("http"):
                        link = candidate
                if not link:
                    for key in ("content", "snippet", "text", "answer", "body", "description"):
                        text = value.get(key)
                        if isinstance(text, str):
                            match = url_re.search(text)
                            if match:
                                link = match.group(0)
                                break
            elif isinstance(value, str):
                logger.debug("normalize_urls: Result string: %s", value[:100])
                if value.startswith("http"):
                    link = value
                else:
                    match = url_re.search(value)
                    if match:
                        link = match.group(0)

        if link:
            result["link"] = canonical_url(str(link))
            value = result.get("Result")
            if isinstance(value, dict):
                if not result.get("title"):
                    result["title"] = value.get("title", "")
                if not result.get("snippet"):
                    result["snippet"] = value.get("snippet") or value.get("content", "")
        else:
            logger.debug(
                "normalize_urls: missing URL. keys=%s result_type=%s",
                list(result.keys()),
                type(result.get("Result")).__name__ if "Result" in result else "N/A",
            )

    return results


def dedupe_results(results: list[dict]) -> list[dict]:
    """Remove duplicate URLs while preserving order."""

    seen: set[str] = set()
    deduped: list[dict] = []
    for result in results:
        url = result.get("link") or ""
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(result)
    return deduped


def pick_time_range(categories: list[str]) -> str | None:
    """Return SearxNG time_range parameter based on categories."""

    if "news" in categories:
        return "week"
    if "it" in categories:
        return "month"
    return None


def to_searx_locale(code: str | None) -> str | None:
    """Normalize language code to Searx locale format."""

    if not code:
        return None
    normalized = code.strip()
    if not normalized:
        return None
    if normalized.lower() == "auto":
        return None
    normalized = normalized.replace("_", "-")
    parts = normalized.split("-")
    lang = parts[0].lower()
    if len(parts) == 1:
        return lang
    region = parts[1].upper() if parts[1] else ""
    return f"{lang}-{region}" if region else lang


__all__ = [
    "canonical_url",
    "normalize_urls",
    "dedupe_results",
    "pick_time_range",
    "to_searx_locale",
]


"""Unit tests for websearch.utils module.

Covers canonical_url, normalize_urls, dedupe_results, diversify_topk, to_searx_locale.
"""

from __future__ import annotations

import pytest

from src.websearch.utils import (
    canonical_url,
    normalize_urls,
    dedupe_results,
    diversify_topk,
    to_searx_locale,
)


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

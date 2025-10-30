#!/usr/bin/env python3
"""Inspect raw SearxNG results and show why links may be missing.

Usage examples:
  uv run python scripts/inspect_searx_raw.py -q "lagarto ensopado" --lang pt
  uv run python scripts/inspect_searx_raw.py -q "news about AI" --categories news --k 30

This script does NOT require the full agent; it calls Searx directly and
prints key distributions, link presence, and a few samples. Use it to
understand when engines return "Result"-only entries without URLs.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

from langchain_community.utilities import SearxSearchWrapper

from websearch.utils import normalize_urls


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--query", required=True)
    # Default None: do not send language unless explicitly provided
    p.add_argument("--lang", default=None)
    p.add_argument(
        "--categories",
        default=os.getenv("SEARCH_CATEGORIES", "general"),
        help="Comma-separated categories (e.g., general,news,it)",
    )
    p.add_argument("--k", type=int, default=int(os.getenv("SEARCH_K", "30")))
    p.add_argument("--safesearch", type=int, default=int(os.getenv("SEARCH_SAFESEARCH", "1")))
    p.add_argument("--searx-host", default=os.getenv("SEARX_HOST", "http://192.168.30.100:8095"))
    p.add_argument("--time-range", default=None, help="Optional time range: day, week, month, year")
    p.add_argument("--engines", default=os.getenv("SEARCH_ENGINES", None))
    p.add_argument("--blocked-engines", default=os.getenv("SEARCH_BLOCKED_ENGINES", None))
    p.add_argument("--dump", default=None, help="Optional path to dump raw JSON for later replay")
    return p.parse_args()


def build_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "categories": [c.strip() for c in args.categories.split(",") if c.strip()],
        "num_results": args.k,
        "safesearch": args.safesearch,
    }
    if args.lang and args.lang.lower() != "auto":
        kw["language"] = args.lang
    if args.time_range:
        kw["time_range"] = args.time_range
    if args.engines:
        kw["engines"] = args.engines
    if args.blocked_engines:
        kw["blocked_engines"] = args.blocked_engines
    return kw


def summarize(results: list[dict[str, Any]]):
    print(f"n_raw={len(results)}")
    if not results:
        return

    # Key frequency
    key_counter = Counter()
    for r in results:
        key_counter.update(r.keys())

    print("Top keys:")
    for k, v in key_counter.most_common(15):
        print(f"  {k}: {v}")

    has_link = sum(1 for r in results if any(r.get(f) for f in ("link", "url", "href")))
    has_result = sum(1 for r in results if "Result" in r)
    print(f"has_std_link={has_link} has_Result={has_result}")

    # Normalize and recount
    normalized = normalize_urls([dict(r) for r in results])
    with_link = sum(1 for r in normalized if r.get("link"))
    print(f"with_link_after_normalize={with_link}")

    # Show a couple of samples
    print("\nSample entries (up to 3):")
    for r in normalized[:3]:
        keys = list(r.keys())
        print("- keys:", keys)
        print("  link:", r.get("link"))
        print("  title:", (r.get("title") or "")[:200])
        print("  snippet:", (r.get("snippet") or "")[:200])


def main():
    args = parse_args()
    client = SearxSearchWrapper(searx_host=args.searx_host)
    results = client.results(args.query, **build_kwargs(args))

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Dumped raw results to: {args.dump}")

    summarize(results)


if __name__ == "__main__":
    main()

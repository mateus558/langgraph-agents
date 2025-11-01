#!/usr/bin/env python3
"""Run the websearch pipeline stages with detailed stats.

This script fetches from SearxNG, then runs:
- normalize_urls
- dedupe_results
- diversify_topk
- diversify_topk_mmr (optional, needs embeddings)

Usage:
  uv run python scripts/debug_websearch_results.py -q "lagarto ensopado" --lang pt --mmr auto

MMR modes:
  --mmr off   : skip MMR, only domain diversify
  --mmr auto  : try to build embedder like the agent (GPU-aware HF -> OpenAI)
  --mmr on    : force attempt (same as auto), error if embedder fails
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any, Optional

from langchain_community.utilities import SearxSearchWrapper

from websearch.config import SearchAgentConfig
from websearch.utils import dedupe_results, normalize_urls
from websearch.ranking import diversify_topk, diversify_topk_mmr


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("-q", "--query", required=True)
    # Default None: avoid enforcing a language unless explicitly requested
    p.add_argument("--lang", default=None)
    p.add_argument(
        "--categories",
        default=os.getenv("SEARCH_CATEGORIES", "general"),
        help="Comma-separated categories (e.g., general,news,it)",
    )
    p.add_argument("--k", type=int, default=int(os.getenv("SEARCH_K", "30")))
    p.add_argument("--safesearch", type=int, default=int(os.getenv("SEARCH_SAFESEARCH", "1")))
    p.add_argument("--searx-host", default=os.getenv("SEARX_HOST", "http://192.168.30.100:8095"))
    p.add_argument("--time-range", default=None)
    p.add_argument("--mmr", choices=["off", "auto", "on"], default="off")
    p.add_argument("--dump-raw", default=None, help="Save raw results JSON to this path")
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
    return kw


def maybe_build_embedder(mode: str) -> Optional[object]:
    if mode == "off":
        return None
    # Reuse agent's embedder initialization logic for consistency
    try:
        from websearch.agent import WebSearchAgent
        cfg = SearchAgentConfig()
        agent = WebSearchAgent(cfg)
        return cfg.embedder
    except Exception as exc:
        if mode == "on":
            raise
        print(f"[info] embedder not available: {exc}")
        return None


def head(items, n=5):
    return items[:n]


def summarize_stage(name: str, results: list[dict[str, Any]]):
    total = len(results)
    with_link = sum(1 for r in results if r.get("link"))
    keys = Counter()
    for r in results:
        keys.update(r.keys())
    print(f"[{name}] total={total} with_link={with_link}")
    if total:
        print(f"[{name}] top_keys:", head(keys.most_common(10)))
        print(f"[{name}] sample_link:", results[0].get("link"))


def main():
    args = parse_args()

    client = SearxSearchWrapper(searx_host=args.searx_host)
    raw = client.results(args.query, **build_kwargs(args))

    if args.dump_raw:
        with open(args.dump_raw, "w", encoding="utf-8") as f:
            json.dump(raw, f, ensure_ascii=False, indent=2)
        print(f"Dumped raw results to: {args.dump_raw}")

    summarize_stage("raw", raw)

    normalized = normalize_urls([dict(r) for r in raw])
    summarize_stage("normalized", normalized)

    deduped = dedupe_results(normalized)
    summarize_stage("deduped", deduped)

    diversified = diversify_topk(deduped, k=args.k)
    summarize_stage("diversified", diversified)

    embedder = maybe_build_embedder(args.mmr)
    if embedder is not None:
        import asyncio

        async def run_mmr():
            return await diversify_topk_mmr(deduped, k=args.k, query=args.query, embedder=embedder)

        reranked = asyncio.run(run_mmr())
        summarize_stage("mmr", reranked)
    else:
        print("[info] MMR skipped (no embedder)")

    if normalized and not diversified:
        print(
            f"[note] cleaned empty after diversification. with_link/total="
            f"{sum(1 for r in normalized if r.get('link'))}/{len(normalized)}"
        )


if __name__ == "__main__":
    main()

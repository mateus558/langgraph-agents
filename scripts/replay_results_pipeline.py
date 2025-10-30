#!/usr/bin/env python3
"""Replay the search results pipeline from a saved JSON file.

Given a file with an array of Searx-like result dicts, this will run:
- normalize_urls
- dedupe_results
- diversify_topk
- diversify_topk_mmr (optional, needs embeddings)

Usage:
  uv run python scripts/replay_results_pipeline.py --file raw.json --k 30 --mmr off
  uv run python scripts/replay_results_pipeline.py --file raw.json --k 30 --mmr auto --query "lagarto ensopado"

Note: --query is only needed if you want MMR (it uses query for relevance).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any, Optional

from websearch.config import SearchAgentConfig
from websearch.utils import (
    dedupe_results,
    diversify_topk,
    diversify_topk_mmr,
    normalize_urls,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="Path to JSON with raw results array")
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--mmr", choices=["off", "auto", "on"], default="off")
    p.add_argument("--query", default="", help="Query text (required for MMR)")
    return p.parse_args()


def maybe_build_embedder(mode: str) -> Optional[object]:
    if mode == "off":
        return None
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

    with open(args.file, "r", encoding="utf-8") as f:
        raw = json.load(f)
        if not isinstance(raw, list):
            raise ValueError("JSON must be an array of result objects")

    summarize_stage("raw", raw)

    normalized = normalize_urls([dict(r) for r in raw])
    summarize_stage("normalized", normalized)

    deduped = dedupe_results(normalized)
    summarize_stage("deduped", deduped)

    diversified = diversify_topk(deduped, k=args.k)
    summarize_stage("diversified", diversified)

    embedder = maybe_build_embedder(args.mmr)
    if embedder is not None and args.query:
        import asyncio

        async def run_mmr():
            return await diversify_topk_mmr(deduped, k=args.k, query=args.query, embedder=embedder)

        reranked = asyncio.run(run_mmr())
        summarize_stage("mmr", reranked)
    elif embedder is not None and not args.query:
        print("[warn] MMR requested but --query is empty; skipping MMR stage")
    else:
        print("[info] MMR skipped (no embedder)")

    if normalized and not diversified:
        print(
            f"[note] cleaned empty after diversification. with_link/total="
            f"{sum(1 for r in normalized if r.get('link'))}/{len(normalized)}"
        )


if __name__ == "__main__":
    main()

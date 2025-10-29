"""LangChain tool wrapper around the WebSearch agent."""

from __future__ import annotations

import os
import random
from threading import RLock
from typing import Any

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# Import from the new websearch package
from websearch.config import SearchAgentConfig

from . import WebSearchAgent

# --------------------------------------------------------------------
# Singleton thread-safe cache per k
# --------------------------------------------------------------------
_AGENTS: dict[int, WebSearchAgent] = {}
_LOCK = RLock()

# Default values - can be overridden by environment variables
DEFAULT_SEARX_HOST = os.getenv("SEARX_HOST", "http://localhost:8095")
DEFAULT_BASE_URL = os.getenv("LLM_BASE_URL") or os.getenv("BASE_URL")
DEFAULT_MODEL_NAME = os.getenv("WEBSEARCH_MODEL_NAME", "llama3.1")
DEFAULT_TEMPERATURE = float(os.getenv("SEARCH_TEMPERATURE", "0.5"))
DEFAULT_NUM_CTX = int(os.getenv("SEARCH_NUM_CTX", "8192"))
DEFAULT_K = int(os.getenv("SEARCH_K", "8"))


def _build_agent_for_k(
    *,
    searx_host: str = DEFAULT_SEARX_HOST,
    base_url: str | None = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    k: int = DEFAULT_K,
):
    cfg = SearchAgentConfig(
        searx_host=searx_host,
        k=k,
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
        num_ctx=num_ctx,
    )
    return WebSearchAgent(cfg)


def _get_agent(
    *,
    searx_host: str = DEFAULT_SEARX_HOST,
    base_url: str | None = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    k: int = DEFAULT_K,
):
    """Get (or create) an agent with the requested 'k', thread-safe."""
    with _LOCK:
        agent = _AGENTS.get(k)
        if agent is None:
            agent = _build_agent_for_k(
                searx_host=searx_host,
                base_url=base_url,
                model_name=model_name,
                temperature=temperature,
                num_ctx=num_ctx,
                k=k,
            )
            _AGENTS[k] = agent
        return agent


# --------------------------------------------------------------------
# Tool
# --------------------------------------------------------------------
@tool(
    "websearch",
    description=(
        "Search the web via SearXNG. Accepts a list of queries.\n"
        "Ask with k at least equal to the number of queries.\n"
    "Usage examples:\n"
    "- websearch(queries=[\"latest tech news\"], top_k=5)\n"
    "- websearch(queries=[\"us economy outlook\", \"inflation forecast 2025\"], top_k=10)"
    ),
    response_format="content_and_artifact",
)  # use return_direct=True if you want to end the conversation
def websearch(
    queries: list[str] | None = None,
    top_k: int | None = None,
) -> tuple[str, dict]:
    """Search the web and return a concise, source-backed summary.

    Features:
    - Accepts a list of `queries` to search in batch.
    - Runs categorize -> searx -> summarize per query under the hood.

    Returns:
        content: Short summary paragraph(s) referencing a few sources.
        artifact: Dictionary with metadata (categories per query, merged results, sampled results, etc.).

    Args:
        queries: Multiple queries to run in parallel (batch).
        top_k: Final number of results to summarize. Defaults to 8.
    """
    try:
        k = int(top_k) if top_k is not None else DEFAULT_K
        # Build the batch of queries
        qlist: list[str] = []
        if queries and isinstance(queries, list):
            qlist = [q for q in (q.strip() for q in queries) if q]

        if not qlist:
            return (
                "No query provided.",
                {"error": "missing_query", "queries": queries, "k": k},
            )
        agent = _get_agent(k=k)

        # Prepare input states for batch invocation
        inputs = [
            {
                "query": q,
                "categories": None,
                "results": None,
                "summary": None,
            }
            for q in qlist
        ]

        # Run in batch if supported, else sequential
        graph: Any = agent.agent
        if hasattr(graph, "batch") and callable(graph.batch):
            from typing import cast as _cast
            states = _cast(list[Any], graph.batch(inputs))
        else:
            states = [graph.invoke(inp) for inp in inputs]

        # Collect and deduplicate results
        merged_results: list[dict[str, Any]] = []
        seen: set[str] = set()
        categories_per_query: dict[str, Any] = {}

        def _result_key(item: dict[str, Any]) -> str:
            url = (item.get("link") or item.get("url") or "").strip()
            if url:
                return url
            # fallback: title+snippet signature
            title = (item.get("title") or "").strip()
            snip = (item.get("snippet") or "").strip()
            return f"{title}|{snip}"

        for q, st in zip(qlist, states, strict=False):
            categories_per_query[q] = st.get("categories")
            res = st.get("results") or []
            if isinstance(res, list):
                for it in res:
                    if not isinstance(it, dict):
                        continue
                    key = _result_key(it)
                    if key and key not in seen:
                        seen.add(key)
                        merged_results.append(it)

        if not merged_results:
            return (
                "No results found.",
                {
                    "queries": qlist,
                    "categories": categories_per_query,
                    "results": [],
                    "sampled": [],
                    "k": k,
                },
            )

        # Randomly sample final K results for summarization
        sample_size = min(k, len(merged_results))
        sampled_results = random.sample(merged_results, sample_size)

        # Summarize with the same model used by the agent when available
        model = None
        try:
            model = _get_agent(k=k).get_model()
        except Exception:
            model = None
        if model is None:
            model = init_chat_model(
                DEFAULT_MODEL_NAME,
                model_provider="ollama",
                base_url=DEFAULT_BASE_URL,
                temperature=DEFAULT_TEMPERATURE,
                kwargs={"num_ctx": DEFAULT_NUM_CTX},
            )

        sys = SystemMessage(
            content=(
                "Summarize search results in a useful and objective way, responding to the original query. "
                "Mention relevant sources (URLs). Respond in the same language as the query."
            )
        )

        # Build the summary prompt using the sampled results
        if len(qlist) == 1:
            qlabel = f"Query: {qlist[0]}\n\n"
        else:
            qlabel = "Queries:\n- " + "\n- ".join(qlist) + "\n\n"

        def _line_for(idx: int, result: dict[str, Any]) -> str:
            url = (result.get("link") or result.get("url") or "").strip()
            return f"{idx}. {result.get('title','')} — {url}\n{result.get('snippet','')}"

        lines = [_line_for(i + 1, r) for i, r in enumerate(sampled_results)]
        prompt = qlabel + "Results (sampled):\n" + "\n".join(lines)

        out = model.invoke([sys, HumanMessage(content=prompt)])
        summary = (getattr(out, "content", "") or "").strip()

        if not summary:
            summary = "No results or summary produced."

        artifact = {
            "queries": qlist,
            "categories": categories_per_query,
            "results": merged_results,
            "sampled": sampled_results,
            "k": k,
        }

        return summary, artifact

    except Exception as e:
        # Short message in content; error details in artifact
        return f"Error: {e}", {"error": str(e), "query": "", "k": top_k or DEFAULT_K}

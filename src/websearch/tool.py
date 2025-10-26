# websearch_tool.py
from __future__ import annotations

from typing import Optional, Tuple, Any, Dict, List
import random
from threading import RLock

from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

# Import from the new websearch package
from . import WebSearchAgent, SearchAgentConfig

# --------------------------------------------------------------------
# Singleton thread-safe do grafo + cache por (k)
# --------------------------------------------------------------------
_AGENTS: Dict[int, WebSearchAgent] = {}
_LOCK = RLock()

DEFAULT_SEARX_HOST = "http://192.168.30.100:8095"
DEFAULT_BASE_URL = "http://192.168.0.5:11434"
DEFAULT_MODEL_NAME = "llama3.1"
DEFAULT_TEMPERATURE = 0.5
DEFAULT_NUM_CTX = 8192
DEFAULT_K = 8


def _build_agent_for_k(
    *,
    searx_host: str = DEFAULT_SEARX_HOST,
    base_url: str = DEFAULT_BASE_URL,
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
    base_url: str = DEFAULT_BASE_URL,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    k: int = DEFAULT_K,
):
    """Obtém (ou cria) um agente com o 'k' solicitado, de forma thread-safe."""
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
        "- websearch(queries=[\"notícias brasileiras últimas\"], top_k=5)\n"
        "- websearch(queries=[\"economia brasil hoje\", \"inflação brasil 2025\"], top_k=10)"
    ),
    response_format="content_and_artifact",
)  # use return_direct=True se quiser encerrar a conversa
def websearch(
    queries: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> Tuple[str, dict]:
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
        qlist: List[str] = []
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
        if hasattr(agent.graph, "batch") and callable(getattr(agent.graph, "batch")):
            states = agent.graph.batch(inputs)  # type: ignore[attr-defined, call-arg]
        else:
            states = [agent.graph.invoke(inp) for inp in inputs]  # type: ignore[arg-type]

        # Collect and deduplicate results
        merged_results: List[Dict[str, Any]] = []
        seen: set[str] = set()
        categories_per_query: Dict[str, Any] = {}

        def _result_key(item: Dict[str, Any]) -> str:
            url = (item.get("url") or "").strip()
            if url:
                return url
            # fallback: title+snippet signature
            title = (item.get("title") or "").strip()
            snip = (item.get("snippet") or "").strip()
            return f"{title}|{snip}"

        for q, st in zip(qlist, states):
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

        # Summarize with the same model family used by the graph
        model = init_chat_model(
            DEFAULT_MODEL_NAME,
            model_provider="ollama",
            base_url=DEFAULT_BASE_URL,
            temperature=DEFAULT_TEMPERATURE,
            kwargs={"num_ctx": DEFAULT_NUM_CTX},
        )

        sys = SystemMessage(
            content=(
                "Resuma os resultados de busca de forma útil e objetiva, respondendo à consulta original. "
                "Mencione fontes relevantes (URLs)."
            )
        )

        # Build the summary prompt using the sampled results
        if len(qlist) == 1:
            qlabel = f"Consulta: {qlist[0]}\n\n"
        else:
            qlabel = "Consultas:\n- " + "\n- ".join(qlist) + "\n\n"

        lines = [
            f"{i+1}. {r.get('title','')} — {r.get('url','')}\n{r.get('snippet','')}"
            for i, r in enumerate(sampled_results)
        ]
        prompt = qlabel + "Resultados (amostrados):\n" + "\n".join(lines)

        out = model.invoke([sys, HumanMessage(content=prompt)])  # type: ignore
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
        # Em content, mensagem curta; em artifact, detalhes do erro
        return f"Error: {e}", {"error": str(e), "query": "", "k": top_k or DEFAULT_K}

"""WebSearch agent implementation.

This module provides the main WebSearchAgent class that orchestrates
the three-step search process: categorize -> search -> summarize.
"""

from __future__ import annotations
from typing import Any, Dict, List
from json import JSONDecodeError
import json
import re
import time

from requests import RequestException
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.utilities import SearxSearchWrapper

from core.contracts import AgentProtocol

from websearch.config import SearchAgentConfig, SearchState
from websearch.constants import ALLOWED_CATEGORIES
from websearch.heuristics import heuristic_categories, CategoryResponse
from websearch.utils import (
    dedupe_results,
    diversify_topk,
    normalize_urls,
    canonical_url,
    pick_time_range,
)
from langgraph.types import CachePolicy
from dotenv import load_dotenv
load_dotenv()  # sem argumentos: carrega arquivo `.env` do diretório atual ou acima


class WebSearchAgent(AgentProtocol):
    """Class-based WebSearch agent with categorize -> search -> summarize pipeline.
    
    This agent performs web searches via SearxNG with intelligent category selection,
    result deduplication/diversification, and LLM-powered summarization.
    
    Pipeline:
        1. Categorize: Determine search categories using heuristics + LLM
        2. Search: Query SearxNG with category-aware settings
        3. Summarize: Generate a concise summary with whitelisted source links
    
    Example:
        config = AgentConfig(searx_host="http://localhost:8080", k=5)
        agent = WebSearchAgent(config)
        
        result = agent.invoke({
            "query": "latest Python news",
            "messages": [],
            "categories": None,
            "results": None,
            "summary": None,
        })
        
        print(result["summary"])
    """

    def __init__(self, config: SearchAgentConfig | None = None):
        """Initialize the WebSearch agent.
        
        Args:
            config: Agent configuration with model, Searx, and search settings.
        """
        self.config = config or SearchAgentConfig()
        self.graph = self._build_graph()

    def invoke(self, state: SearchState) -> Any:
        """Invoke the agent graph with the given state.
        
        Args:
            state: SearchState with at minimum a 'query' field.
            
        Returns:
            Updated SearchState with categories, results, and summary populated.
        """
        return self.graph.invoke(state)

    def get_mermaid(self) -> str:
        """Get Mermaid diagram of the agent's graph.
        
        Returns:
            Mermaid diagram as a string.
        """
        return self.graph.get_graph().draw_mermaid()

    def _build_model(self):
        """Build the chat model using init_chat_model.
        
        Returns:
            Initialized LangChain chat model.
        """
        return init_chat_model(
            self.config.model_name,
            model_provider="ollama",
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            kwargs={"num_ctx": self.config.num_ctx},
        )

    def _build_categorize_node(self):
        """Build the categorize query node.
        
        This node:
        1. Gets heuristic category hints from regex patterns
        2. Uses LLM with structured output to select final categories
        3. Falls back to heuristics if LLM fails
        
        Returns:
            Node function that processes SearchState and returns category update.
        """
        SYS_PROMPT = SystemMessage(
            content=(
                "Você recebe uma consulta e deve escolher as categorias do Searx mais relevantes. "
                f"Conjunto permitido: {', '.join(ALLOWED_CATEGORIES)}. "
                "Responda apenas com JSON compatível com o schema solicitado."
            )
        )

        FEWSHOT = [
            HumanMessage(content='QUERY: "breaking microsoft acquisition"\nHINTS: ["news","economics"]'),
            SystemMessage(content='{"categories":["news","economics"]}'),
            HumanMessage(content='QUERY: "github actions cache permission denied"\nHINTS: ["news"]'),
            SystemMessage(content='{"categories":["it"]}'),
            HumanMessage(content='QUERY: "coisas variadas sem contexto"\nHINTS: []'),
            SystemMessage(content='{"categories":["general"]}'),
            HumanMessage(content='QUERY: "trailer do filme dune 2"\nHINTS: ["videos"]'),
            SystemMessage(content='{"categories":["videos"]}'),
        ]

        # Use structured output
        structured_model = self.config.model.with_structured_output(CategoryResponse) if self.config.model else None

        def llm_categories(query: str, hints: List[str], limit: int) -> List[str]:
            """Get categories from LLM with structured output."""
            if not structured_model:
                return ["general"]
            user_msg = HumanMessage(content=f'QUERY: "{query}"\nHINTS: {json.dumps(hints, ensure_ascii=False)}')
            try:
                resp: CategoryResponse = structured_model.invoke([SYS_PROMPT, *FEWSHOT, user_msg])  # type: ignore
                return resp.categories[:max(1, limit)]
            except Exception:
                return ["general"]

        def _categorize(state: SearchState) -> dict:
            """Categorize node implementation."""
            t0 = time.perf_counter()
            query = state["query"]
            hints = heuristic_categories(query)
            cats = llm_categories(query, hints, self.config.max_categories)
            dt = time.perf_counter() - t0
            print(f"[websearch_agent] node=categorize_query took {dt:.3f}s")
            return {
                "categories": cats,
            }

        return _categorize

    def _build_web_search_node(self):
        """Build the web search node.
        
        This node:
        1. Configures Searx parameters based on categories and config
        2. Applies time range policy for recency-sensitive categories
        3. Calls Searx with retry logic
        4. Normalizes, deduplicates, and diversifies results
        
        Returns:
            Node function that performs web search and returns results.
        """
        search = SearxSearchWrapper(searx_host=self.config.searx_host)

        def searx_call_with_retry(call):
            """Execute Searx call with exponential backoff retry."""
            for attempt in range(self.config.retries + 1):
                try:
                    return call()
                except (RequestException, JSONDecodeError, ValueError):
                    if attempt >= self.config.retries:
                        raise
                    time.sleep(self.config.backoff_base * (2 ** attempt))

        def _search(state: SearchState) -> dict:
            """Web search node implementation."""
            t0 = time.perf_counter()
            q = state["query"].strip()
            # Always ensure 'general' category is present for the search.
            cats = state.get("categories") or []
            if not cats:
                cats = ["general"]
            elif "general" not in cats:
                # Append 'general' to broaden the search and ensure fallback
                cats = [*cats, "general"]

            state["categories"] = cats  # Update state with ensured categories  

            kwargs: Dict[str, Any] = {
                "categories": cats,
                "num_results": self.config.k * 2,
                "safesearch": self.config.safesearch,
            }
            if self.config.lang:
                kwargs["language"] = self.config.lang
            if (tr := pick_time_range(cats)):
                kwargs["time_range"] = tr

            allow = (self.config.engines_allow or {}).get(cats[0], None)
            block = (self.config.engines_block or {}).get(cats[0], None)
            if allow:
                kwargs["engines"] = ",".join(allow)
            if block:
                kwargs["blocked_engines"] = ",".join(block)

            try:
                raw = searx_call_with_retry(lambda: search.results(q, **kwargs)) or []
            except (RequestException, JSONDecodeError, ValueError) as e:
                # Handle non-JSON or HTML responses, timeouts, or other recoverable errors
                dt = time.perf_counter() - t0
                print(f"[websearch_agent] node=web_search error: {type(e).__name__}: {e} (took {dt:.3f}s)")
                return {
                    "results": [],
                    "messages": [HumanMessage(content=f"[SEARCH] erro ao consultar Searx: {type(e).__name__}")],
                }
            cleaned = diversify_topk(dedupe_results(normalize_urls(raw)), k=self.config.k)

            dt = time.perf_counter() - t0
            print(f"[websearch_agent] node=web_search took {dt:.3f}s")
            return {
                "results": cleaned,
            }

        return _search

    def _build_summarize_node(self):
        """Build the summarize results node.
        
        This node:
        1. Takes search results and creates a whitelist of allowed URLs
        2. Prompts LLM to summarize while only citing whitelisted links
        3. Post-processes output to remove any non-whitelisted URLs
        4. Falls back to raw results if LLM unavailable
        
        Returns:
            Node function that generates summary from search results.
        """
        sys = SystemMessage(content="Seja factual, conciso e objetivo.")

        def _summarize(state: SearchState) -> dict:
            """Summarize node implementation."""
            t0 = time.perf_counter()
            query = state["query"]
            results = state.get("results") or []
            
            if not results:
                dt = time.perf_counter() - t0
                print(f"[websearch_agent] node=summarize took {dt:.3f}s")
                return {
                    "summary": f"Não encontrei resultados para: {query}",
                }

            urls = [r.get("link", "") for r in results if r.get("link")]
            lines = [
                f"{i+1}. {r.get('title', '')} — {r.get('link', '')}\n{r.get('snippet', '')}"
                for i, r in enumerate(results)
            ]

            whitelist_msg = "Você **só pode** citar links exatamente da lista a seguir:\n" + "\n".join(urls)
            prompt = (
                f"{whitelist_msg}\n\n"
                "Responda à consulta com 1–3 parágrafos, citando 2–5 links APENAS dessa lista.\n\n"
                f"Consulta: {query}\n\nResultados:\n" + "\n".join(lines)
            )

            if self.config.model is None:
                top = "\n".join(lines[:3])
                dt = time.perf_counter() - t0
                print(f"[websearch_agent] node=summarize took {dt:.3f}s (fallback)")
                return {
                    "summary": f"(Fallback sem LLM)\n{top}",
                }

            out = self.config.model.invoke([sys, HumanMessage(content=prompt)])  # type: ignore
            text = (out.content or "").strip()  # type: ignore

            # Remove any non-whitelisted URLs from the summary
            safe = {canonical_url(u) for u in urls}
            for token in re.findall(r"https?://\S+", text):
                if canonical_url(token) not in safe:
                    text = text.replace(token, "")

            dt = time.perf_counter() - t0
            print(f"[websearch_agent] node=summarize took {dt:.3f}s")
            return {"summary": text}

        return _summarize

    def _build_graph(self):
        """Build and compile the LangGraph.
        
        Creates a sequential graph: START -> categorize -> search -> summarize -> END
        
        Returns:
            Compiled LangGraph ready for invocation.
        """
        g = StateGraph(SearchState)

        g.add_node("categorize_query", self._build_categorize_node())
        g.add_node("web_search", 
                   self._build_web_search_node(),
                   cache_policy=CachePolicy(ttl=120))
        g.add_node("summarize", 
                   self._build_summarize_node(),
                   cache_policy=CachePolicy(ttl=120))

        g.add_edge(START, "categorize_query")
        g.add_edge("categorize_query", "web_search")
        g.add_edge("web_search", "summarize")
        g.add_edge("summarize", END)

        return g.compile(name="WebSearchAgent")

config = SearchAgentConfig()
websearch = WebSearchAgent(config).graph

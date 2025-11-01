# WebSearch Agent

A LangGraph-powered agent that performs category-aware web search via SearxNG and produces a concise, source-backed summary with an LLM. It uses fast heuristics plus optional LLM assistance to classify queries, performs resilient searches (with retries, language pivoting, and engine filters), and reranks results using MMR for relevance and diversity.

## Highlights

- Categorize → Search → Summarize pipeline built with LangGraph
- SearxNG backend with retries, backoff, and category/engine controls
- Language-aware: detection, optional pivot to English, and translation for categorization
- MMR reranking with FAISS/HuggingFace/OpenAI embeddings when available; graceful fallbacks
- Summary constrained to cite only returned sources and follow a strict time policy
- Usable as a Python API, a LangChain Tool, or via the LangGraph server export

---

## Architecture

The agent compiles a graph with three nodes:

1) Categorize (nodes/categorize.py)
   - Detects language (LangDetector) and translates query to English for categorization, if needed.
   - Applies regex heuristics to propose category hints.
   - Optionally calls the LLM with structured output (CategoryResponse) to select final categories constrained to `ALLOWED_CATEGORIES`.
   - Output: `categories`, `lang`, `query_en`.

2) Web Search (nodes/search.py)
   - Calls SearxNG via `langchain_community.utilities.SearxSearchWrapper` with:
     - category-aware policies, time range selection, language filtering, and engine allow/block lists.
     - retries with exponential backoff + jitter and a concurrency semaphore.
     - optional pivot: for likely non‑English input, also search an English translation and merge.
   - Normalizes and deduplicates results (`normalize_urls`, `dedupe_results`).
   - Reranks/diversifies top‑k with `websearch.ranking.diversify_topk_mmr`:
     - Tiered strategy: FAISS+embedder → standalone MMR → domain rotation fallback.
   - Multiple fallbacks if no results: retry without language, without categories, or try common alt languages (en/pt).
   - Output: `results` (list of dicts with `title`, `link`, `snippet`), final `categories`, `lang`.

3) Summarize (nodes/summarize.py)
   - Prompts the LLM with a whitelist of returned URLs and a strict temporal policy.
   - Requires citations by index; removes any URL not in the whitelist from the output.
   - Injects current UTC and local time (configurable timezone) and expects time-sensitive summaries.
   - If no model is configured, returns a simple fallback made from the top items.
   - Output: `summary` (string).

Cross-cutting dependencies are wired via `NodeDependencies` (nodes/shared.py).

---

## Public API

- Class: `websearch.agent.WebSearchAgent`
  - Compiles the graph on init; access via `agent.agent` (LangGraph compiled graph).
  - Input/Output state type: `websearch.config.SearchState` (TypedDict):
    - `query: str`
    - `categories: list[str] | None`
    - `results: list[dict] | None`
    - `summary: str | None`
    - `lang: str | None`
    - `query_en: str | None`

- Tool: `websearch.tool.websearch`
  - LangChain Tool that accepts `queries: list[str]` and `top_k: int`.
  - Returns `(content: str, artifact: dict)` where content is a short summary and artifact contains merged and sampled results.

- LangGraph server export: `websearch.agent.websearch_agent`
  - Lazy-initialized agent configured from environment variables; referenced by `langgraph.json`.

---

## Usage

### Python API

```python
from websearch.agent import WebSearchAgent
from websearch.config import SearchAgentConfig, SearchState

config = SearchAgentConfig(
    model_name="llama3.1",           # via Ollama or your LLM provider
    base_url="http://localhost:11434",  # optional
    searx_host="http://localhost:8095",
    k=8,
)

agent = WebSearchAgent(config)

state: SearchState = {
    "query": "latest Python news",
    "categories": None,
    "results": None,
    "summary": None,
    "lang": None,
    "query_en": None,
}

# Sync
out = agent.agent.invoke(state)
print(out["summary"])          # concise, source-backed summary
print(len(out["results"]))     # up to k diversified results

# Async
# out = await agent.agent.ainvoke(state)
```

### LangChain Tool

```python
from websearch.tool import websearch

content, artifact = websearch.invoke({
    "queries": ["us economy outlook", "inflation forecast 2025"],
    "top_k": 10,
})
print(content)
print(artifact["sampled"])  # sampled results included in the summary
```

### LangGraph Server

The module exposes `websearch_agent`, which `langgraph.json` can reference directly. Configure via environment variables (see below).

---

## Configuration

`SearchAgentConfig` extends a base `AgentConfig` (see `core.contracts.AgentConfig`) and includes:

- Searx/Querying
  - `searx_host: str` (default `http://192.168.30.100:8095`)
  - `k: int` final diversified results (default 30)
  - `max_categories: int` category cap for the query (default 4)
  - `lang: str | None` language code for Searx (e.g., `en-US`, `pt-BR`, or `auto`)
  - `safesearch: int` 0=off, 1=moderate, 2=strict (default 1)
  - `timeout_s: float` per-request timeout to Searx (default 8.0)
  - `retries: int` retry attempts (default 2)
  - `backoff_base: float` base for exponential backoff (default 0.6)
  - `engines_allow: dict[str, list[str]] | None` per-category allowlist
  - `engines_block: dict[str, list[str]] | None` per-category blocklist
  - `searx_max_concurrency: int` per-agent semaphore for Searx calls (default 8)
  - `pivot_to_english: bool` also query English and merge for likely non‑English inputs (default True)
  - `local_timezone: str` IANA timezone for summaries (default `America/Sao_Paulo`)

- LLM
  - `model_name: str` provider/model id (e.g., `llama3.1`)
  - `base_url: str | None` provider base URL
  - `temperature: float` (default 0.5)
  - `num_ctx: int` context window hint

- MMR Reranking
  - `use_vectorstore_mmr: bool` enable FAISS-based MMR when available (default True)
  - `mmr_lambda: float` balance relevance vs diversity in [0,1] (default 0.55)
  - `mmr_fetch_k: int` candidates to consider before MMR (default 50)
  - `embedder: object | None` embeddings backend; if None, the agent attempts to initialize one

### Environment variables

The config reads several env vars on init:

- Core
  - `SEARX_HOST` — SearxNG URL
  - `SEARCH_K` — final diversified results
  - `SEARX_TIMEOUT_S` — request timeout
  - `SEARCH_PIVOT_TO_EN` — enable/disable pivot to English (`1/true/on` or `0/false/off`)
  - `LOCAL_TIMEZONE` — IANA timezone (e.g., `America/Sao_Paulo`)
  - `SEARCH_MAX_CONCURRENCY` — per-agent concurrency cap

- LLM
  - `WEBSEARCH_MODEL_NAME`, `MODEL_NAME`, `LLM_BASE_URL`, `BASE_URL`, `SEARCH_TEMPERATURE`, `SEARCH_NUM_CTX`

- MMR
  - `USE_VECTORSTORE_MMR` — enable FAISS-path MMR (`true/false`)
  - `MMR_LAMBDA` — 0.0–1.0
  - `MMR_FETCH_K` — positive integer
  - `EMBEDDINGS_MODEL_NAME` — HF model id for local embeddings (default `intfloat/e5-base-v2`)
  - `OPENAI_API_KEY` — if set, the agent can use OpenAI embeddings as a fallback

---

## Categories and heuristics

- Allowed categories: see `websearch.constants.ALLOWED_CATEGORIES`.
- Heuristic patterns (regex) guide the LLM and provide deterministic hints.
- Sanitization maps synonyms/aliases to canonical names and filters to allowed set.

---

## Result shape

`results` items follow the SearxNG convention and are normalized to ensure a `link` URL is present. Typical keys:

- `title: str`
- `link: str` (canonicalized)
- `snippet: str`

Utilities in `websearch.utils` also accept Searx "instant answer" shapes (e.g., a `Result` object or string with embedded URL).

---

## Extending

- Add/adjust categories in `constants.py` and heuristics in `heuristics.py`.
- Customize summarization policy or output format in `nodes/summarize.py`.
- Tweak retry, backoff, and pivot behavior in `nodes/search.py`.
- Swap language detection backends by extending `language.py`.
- Provide your own embeddings model via `SearchAgentConfig.embedder`.

---

## Testing

Tests cover utilities, MMR, and the agent wiring with mocks. From the repo root:

```bash
pytest -k websearch -q
```

---

## Troubleshooting

- No results: the agent retries with looser constraints; verify your SearxNG instance and engines configuration.
- MMR disabled: ensure optional dependencies are installed (FAISS, `langchain-huggingface`, `torch` for GPU), or set `USE_VECTORSTORE_MMR=false` to use fallback diversification.
- Unexpected language behavior: set `SEARCH_PIVOT_TO_EN=0` to disable pivoting, or set `SEARCH_LANG` in `SearchAgentConfig` to force a language.
- URLs missing in summaries: expected—summarizer whitelists only returned links and strips any others.
- OpenAI embeddings used instead of HuggingFace: confirm `langchain-huggingface` and `pydantic>=2` are installed. On startup the agent logs `[websearch] langchain-huggingface import failed: ...` when the local embedder cannot initialize and it falls back to OpenAI (if `OPENAI_API_KEY` is set).
- Standalone MMR requires NumPy: if NumPy is missing the agent falls back to domain diversification; install `numpy` to enable non‑FAISS MMR.

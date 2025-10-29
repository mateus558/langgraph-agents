# AI Server – Full Project Review

Review date: 2025-10-28  
Repository: `ai-server` (branch: master)  
Language/Stack: Python 3.11, LangChain/LangGraph, LangGraph CLI, SearxNG

---

## Executive summary

This repository provides two LangGraph-powered agents exposed via the LangGraph server:
- ChatAgent: conversational agent with sliding-window context and summarization
- WebSearchAgent: SearxNG-backed search with heuristic+LLM categorization and source-backed summarization

The project is thoughtfully structured, has a clear deployment story, and includes a small but effective test suite for core utilities and configuration. The WebSearch agent is particularly well organized, with good logging and a disciplined graph/Node design. The Chat agent is cleaner than typical samples, uses an explicit token estimation utility, and integrates summarization in a controlled way.

Key gaps mainly fall under lint/type hygiene, a few defaults that point to local IPs, and some minor consistency issues between documentation, validation scripts, and actual code layout. With focused cleanup (ruff/mypy), a pass at defaults and environment variables, plus a couple of doc fixes, this codebase looks production-ready for a LangGraph server deployment.

Overall rating: 8.2/10

---

## Quality gates

- Build: PASS (Python project; no build errors encountered)
- Lint/Style: FAIL (ruff reports ~115 issues, mostly modernization (typing.List→list), import ordering, whitespace)
- Type check: FAIL (mypy found 18 issues; several unused type: ignore comments, one missing types-requests, a couple of functions returning Any)
- Tests: PASS (11 passed)

Evidence
- Tests: 11 passed in 0.20s (pytest)
- Lint: ruff check src → 115 issues (mostly mechanical, fixable with --fix)
- Type-check: mypy src → 18 errors (see Recommendations)

---

## Architecture and design

- Project layout is clean: `src/` contains core contracts, agents, and utilities; `tests/` contains unit tests.
- Agents are built as LangGraph StateGraphs and exported at module level for server discovery (required by LangGraph CLI). Both modules provide a factory that reads from environment and exports a compiled graph (`chatagent`, `websearch`).
- Configuration is centralized in `src/config.py` and extended in `websearch/config.py`; environment variables override sensible defaults. `get_settings()` implements a cached singleton pattern with `override_settings()` to tweak values at runtime.
- Token accounting and message coercion are abstracted in `utils/messages.py`.
- WebSearch uses a strong pipeline: categorize (heuristics + structured LLM) → fetch via SearxNG → summarize with strict URL whitelist. Nodes return deltas (immutable philosophy), have caching, and use structured logging.
- ChatAgent follows a generate → should_summarize? → summarize pattern with token window management.

What stands out positively
- Clear separation of graph nodes with pure functions returning state deltas
- Structured logging in websearch; timer metrics per node
- Clean env-variable based configuration for LangGraph server exports via `langgraph.json` and Dockerfile `LANGSERVE_GRAPHS`
- Batch support for the websearch tool, careful result merging and dedupe

---

## Code review highlights (by area)

### 1) Core configuration (`src/config.py`)
- Settings dataclass with slots and a safe `from_env()` that normalizes none/null/empty → None for base URL
- Minimal, dependency-free module (dotenv loaded elsewhere)
- `override_settings()` is handy for tests

Suggested tweaks
- Add a `reset_settings_for_tests()` helper (wrapper for setting `_GLOBAL_SETTINGS=None`) to avoid importing internals in tests
- Optional: validate URL formats and model names up-front (pydantic or simple regex)

### 2) Agent contracts (`src/core/contracts.py`)
- `AgentConfig` loads defaults from `get_settings()` and initializes the chat model via `init_chat_model`
- Provider selection is inferred from `base_url` (ollama if provided, else openai). This is pragmatic, documented in code

Suggested tweaks
- Consider explicit `model_provider` in `AgentConfig` (while keeping the current fallback), which simplifies reasoning and type checking

### 3) Utilities (`src/utils/messages.py`)
- TokenEstimator: tries tiktoken, falls back to a 4-chars-per-token heuristic; handles empty inputs and list content
- `coerce_message_content`: robustly converts LC messages to readable strings

Suggested tweaks
- Remove unused import: `from langchain.messages import AnyMessage` (not used in annotations)
- Extract heuristic constant 4 → `CHARS_PER_TOKEN = 4` for clarity

### 4) Chat agent (`src/chatagent/agent.py`)
- Good: loads model config from env or provided config; uses TokenEstimator to track the window; clean, professional system prompt
- Summarizer is injected via the constructor and configured from env
- Module-level export is explicitly justified (server discovery) and built through a factory that loads `.env` if available

Suggested tweaks
- Consider adding structured logging to match websearch style (currently uses logging, which is good)
- Add minimal docstrings for the private node builders for easier navigation

### 5) Chat summarizer (`src/chatagent/summarizer.py`)
- BaseSummarizer centralizes provider-agnostic behavior; Ollama and OpenAI subclasses clarify intent
- Logging of prompt and timing is present and appropriate; no hard-coded IPs
- Filtering of `RemoveMessage` and `__remove_all__` tokens is correct

Suggested tweaks
- File header says `# summarizers.py` but filename is `summarizer.py` (minor nit)
- Consider parameterizing the prompt rules via constructor if future customization is expected

### 6) WebSearch agent (`src/websearch/agent.py`)
- Strong module: disciplined node construction, clean deltas, caching via `CachePolicy(ttl=120)`, structured logging, robust error handling and backoff
- Categorization uses few-shot + structured output via Pydantic model with a heuristics fallback
- Summarization enforces a whitelist of URLs (canonicalized) and strips unapproved links
- Module-level export uses env and provides both `websearch_agent` and alias `websearch`

Suggested tweaks
- Replace legacy `typing.List/Dict/Tuple` with built-in generics; sort imports (ruff will auto-fix)

### 7) WebSearch config/constants/heuristics/utils
- `SearchAgentConfig` extends `AgentConfig`; supports env overrides and sensible defaults
- One default worth changing: `searx_host` currently defaults to `http://192.168.30.100:8095`; prefer `http://localhost:8095` in code (docs already recommend localhost)
- Constants and heuristics are clean and well documented
- URL utilities are practical and concise; minor import/whitespace tidy-ups needed per ruff

### 8) WebSearch LangChain tool (`src/websearch/tool.py`)
- Provides `@tool` for batched websearch with summary, separate from the graph; caches WebSearchAgent per k
- Loads defaults from environment

Suggested tweaks
- Modernize annotations and sort imports via ruff --fix
- Consider explicit `strict=` in zip and avoid `getattr(..., "batch")` pattern (ruff B905/B009)

---

## Tests

- Suite: 11 tests targeting configuration and message utilities
- Results: PASS (11/11)
- CI: not included, but Makefile targets exist; straightforward to add GH Actions/CI to run lint, type, tests

Recommended additions
- Add tests for ChatAgent sliding window and summarization decisions (unit-level; mock LLM)
- Add tests for WebSearch categorize/search/summarize nodes (mock SearxNG + model)
- Add tests for URL whitelist enforcement and canonicalization

---

## Tooling, packaging, and deployment

- pyproject.toml: Well configured with black, ruff, mypy, pytest, coverage; dev extras are defined
- Makefile: sensible tasks for format/lint/type/test/coverage
- .pre-commit-config.yaml: hooks for black, ruff, mypy, plus basic hygiene
- Dockerfile: Based on langchain/langgraph-api:3.11, installs this repo in editable mode, exports graphs via LANGSERVE_GRAPHS; cleans pip tooling for smaller image
- docker-compose.yml: Provides a Postgres service and runs the ai-server; note: references REDIS_URI but no redis service defined
- langgraph.json: Points to `./src/chatagent/agent.py:chatagent` and `./src/websearch/agent.py:websearch`, uses `.env`
- verify_installation.py: Helpful, but checks for `src/core/exceptions.py` and `src/core/validation.py` which are not present—this will report missing files

---

## Security and configuration

What’s good
- Environment-based configuration everywhere that matters; `.env.example` is comprehensive
- No secrets committed; sensitive keys read from env in compose and server runtime

Improvements
- Replace any remaining local-IP defaults with `localhost` (e.g., `SearchAgentConfig.searx_host`) to avoid accidental leakage of personal network details
- docker-compose references `REDIS_URI` pointing to `redis://192.168.30.100:6379/0` but no redis service is defined; either add a redis service or default this to localhost
- Prefer HTTPS for external endpoints in production; document that SearxNG should be behind TLS in real deployments

---

## Observability and reliability

- Logging: WebSearch has excellent structured logging and latency metrics per node; ChatAgent uses logging as well
- Retries/backoff: present for SearxNG calls; good defaults
- Caching: Node-level caching with TTL improves performance for categorize and summarize

Suggestions
- Add request IDs/trace IDs to logs (via context vars or explicit state)
- Consider exposing simple health endpoints or a readiness check if wrapping LangGraph server behind a proxy

---

## Documentation

- README is detailed and matches current deployment pattern (LangGraph server and direct usage)
- QUICK_REFERENCE.md clarifies breaking changes and how to consume the module-level exports vs direct instantiation
- CONTRIBUTING.md is solid

Fixes
- Verify all referenced files in docs exist (e.g., remove references to `core/exceptions.py` and `core/validation.py` unless they’re reintroduced)
- Consider adding a brief ARCHITECTURE.md with diagrams for both agents’ graphs

---

## Dependencies

Runtime: `python-dotenv`, `langchain`, `langchain-community`, `langchain-ollama`, `langchain-openai`, `langgraph`, `langgraph-cli[inmem]`, `pydantic`, `requests`, `tiktoken`  
Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `black`, `ruff`, `mypy`, `pre-commit`

Notes
- All present and reasonable. For mypy, install `types-requests` to silence stub complaints (already configured in pre-commit; also add to dev deps if desired)

---

## Prioritized recommendations

Critical
1) Lint/type hygiene: run `ruff --fix` and address mypy errors
   - Remove unused type: ignore comments
   - Add `types-requests` or `# type: ignore[import-untyped]` where justified
   - Fix functions returning Any (add proper return type or explicit cast)
2) Defaults: change `SearchAgentConfig.searx_host` default to `http://localhost:8095`
3) docker-compose: either add a Redis service or point `REDIS_URI` to `redis://localhost:6379/0`
4) verify_installation.py: update required files list to reflect current repo (remove `core/exceptions.py` and `core/validation.py` unless re-added)

High
5) Expand tests to cover agent nodes (mock LLM + SearxNG); keep unit tests fast
6) Normalize typing across websearch modules (built-in generics, import ordering) and remove trailing whitespace
7) Add CI workflow to run: ruff, mypy, pytest (matrix on 3.11/3.12 if desired)

Medium
8) Consider explicit `model_provider` in `AgentConfig`; keep current inference as default
9) Add request/trace IDs to logs; consider JSON logging depending on deployment
10) Provide a minimal health/readiness check doc when deploying via LangGraph server in containerized environments

Low
11) Minor nits: fix filename header in `summarizer.py`, remove unused imports, extract `CHARS_PER_TOKEN`
12) Add an ARCHITECTURE.md with mermaid diagrams from `get_mermaid()` to aid onboarding

---

## Quick win checklist

- [ ] Run ruff auto-fix and organize imports
- [ ] Fix mypy errors (remove unused ignores; add `types-requests`)
- [ ] Switch `searx_host` default to localhost
- [ ] Update docker-compose `REDIS_URI` or add a redis service
- [ ] Align `verify_installation.py` with actual structure
- [ ] Add CI (ruff + mypy + pytest)

---

## Appendix: commands used (local)

- Tests: `uv run pytest -q` → 11 passed in ~0.20s
- Lint: `uv run ruff check src` → 115 issues (most auto-fixable)
- Type-check: `uv run mypy src` → 18 errors (unused ignores, missing stubs, Any returns)

# LangGraph Agents

Self-hosted LangGraph server exposing production-ready AI agents for conversational chat and intelligent web search.

- Timezone-aware prompts and summaries (configurable via TIMEZONE/TZ_NAME)
- Web search powered by SearxNG with optional MMR reranking
- Clean LangGraph graphs with caching and graceful fallbacks

## Features

- **LangGraph Server Deployment**: Ready-to-deploy server with API endpoints
- **Chat Agent**: Conversational AI with automatic summarization
  - Sliding window token management
  - Context-aware responses
  - Multi-language support
  
- **Web Search Agent**: Intelligent web search using SearxNG
  - LLM-powered query categorization
  - Multi-category search with result deduplication
  - **MMR reranking** for relevance and diversity (new!)
  - Source-backed summarization with URL whitelist

## Deployment Options

### Option 1: LangGraph Server (Recommended for Production)

Start the server and access agents via REST API:

```bash
# Start the server
langgraph dev

# Access chat agent
curl -X POST http://localhost:8123/chatagent/invoke \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

# Access web search agent
curl -X POST http://localhost:8123/websearch/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "latest Python news"}'
```

See `langgraph.json` for configured graphs and `ARCHITECTURE.md` for diagrams.

### Option 2: Direct Python Usage (Development/Testing)

Use agents directly in your Python code:

```python
from src.chatagent.agent import ChatAgent, ChatAgentConfig
from langchain_core.messages import HumanMessage

# Initialize chat agent
agent = ChatAgent(ChatAgentConfig())

# Send a message
result = agent.invoke({
    "messages": [HumanMessage(content="Hello!")],
    "history": [],
    "summary": None,
})

print(result["messages"][-1].content)
```

### Web Search

```python
from src.websearch.agent import WebSearchAgent, SearchAgentConfig

# Initialize search agent
config = SearchAgentConfig(searx_host="http://localhost:8095", k=5)
agent = WebSearchAgent(config)

# Perform search
result = agent.invoke({
    "query": "latest Python news",
    "categories": None,
    "results": None,
    "summary": None
})

print(result["summary"])
```

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) (for local models) or OpenAI API key
- [SearxNG](https://github.com/searxng/searxng) instance (for web search)
- [LangGraph CLI](https://langchain-ai.github.io/langgraph/cloud/reference/cli/) (for server deployment)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd langgraph-agents
```

2. Install dependencies:
```bash
make install-dev
```

3. Copy and configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Set up pre-commit hooks (for development):
```bash
make pre-commit
```

### Running the Server

**Start LangGraph server:**
```bash
langgraph dev
```

The server will be available at `http://localhost:8123` with these endpoints:
- `POST /chatagent/invoke` - Chat agent
- `POST /websearch/invoke` - Web search agent

### Configuration

Edit `.env` file with your settings:

```bash
# Model Configuration
MODEL_NAME=llama3.1
LLM_BASE_URL=http://localhost:11434

# Global Timezone (IANA name)
# Used by ChatAgent and as fallback for WebSearchAgent when LOCAL_TIMEZONE is not set
TIMEZONE=America/Sao_Paulo
# Alternatively, you can use TZ_NAME
# TZ_NAME=America/Sao_Paulo

# Chat Agent
CHAT_MESSAGES_TO_KEEP=5
CHAT_MAX_TOKENS_BEFORE_SUMMARY=4000

# Web Search
SEARX_HOST=http://localhost:8095
SEARCH_K=8
# Optional overrides
# SEARCH_MAX_CATEGORIES=4
# SEARCH_SAFESEARCH=1
# SEARCH_LANG=en-US
# SEARX_TIMEOUT_S=8.0
# SEARCH_RETRIES=2
# SEARCH_BACKOFF_BASE=0.6
# SEARCH_TEMPERATURE=0.5
# SEARCH_PIVOT_TO_EN=1
# SEARCH_MAX_CONCURRENCY=8

# Timezone used specifically by WebSearchAgent summaries (overrides TIMEZONE if set)
# LOCAL_TIMEZONE=America/Sao_Paulo

# Embeddings / MMR (WebSearch)
# EMBEDDINGS_MODEL_NAME=intfloat/e5-base-v2
# OPENAI_API_KEY= # if you want to use OpenAI embeddings as a fallback
# USE_VECTORSTORE_MMR=true
# MMR_LAMBDA=0.55
# MMR_FETCH_K=50
```

See `.env.example` for all available options.

## Documentation

- `src/chatagent/README.md` - ChatAgent details (API, config, streaming)
- `src/websearch/README.md` - WebSearch agent (architecture, MMR, usage)
- `src/symindex/README.md` - Semantic indexer overview
- `ARCHITECTURE.md` - System graphs and how to regenerate
- `QUICK_REFERENCE.md` - Quick start and migration notes
- `CONTRIBUTING.md` - Development guidelines

### References

- LangGraph Local Server (OSS): https://docs.langchain.com/oss/python/langgraph/local-server
- LangGraph Deployments (LangSmith/Cloud): https://docs.langchain.com/langsmith/deployments

### Code Quality

Format code:
```bash
make format
```

Lint code:
```bash
make lint
```

Type check:
```bash
make type-check
```

### Testing

Run tests:
```bash
make test
```

Run tests with coverage:
```bash
make test-cov
```

### Project Structure

```
langgraph-agents/
├── src/
│   ├── config.py              # Global configuration (models, base_url, timezone)
│   ├── chatagent/             # Chat agent implementation
│   │   ├── agent.py           # Main agent logic + server export
│   │   ├── config.py          # State / protocols
│   │   ├── prompts.py         # System and summarizer prompts
│   │   └── summarizer.py      # Summarization logic
│   ├── websearch/             # Web search agent
│   │   ├── agent.py           # Search agent logic + server export
│   │   ├── config.py          # Search configuration
│   │   ├── constants.py       # Category patterns
│   │   ├── heuristics.py      # Query categorization
│   │   ├── language.py        # Language detection
│   │   ├── nodes/             # Node builders (categorize/search/summarize)
│   │   ├── prompts.py         # Summarization prompts
│   │   ├── ranking/           # MMR reranking helpers
│   │   ├── tool.py            # LangChain tool wrapper
│   │   └── utils.py           # Utility functions
│   ├── core/                  # Core components
│   │   ├── contracts.py       # Protocols, model factory, agent mixin
│   │   ├── prompts.py         # Prompt abstraction
│   │   └── time.py            # Time helpers (clock/zone)
│   ├── symindex/              # Semantic indexer
│   │   ├── parser.py          # Python AST → semantic units
│   │   └── README.md          # Package docs
│   └── utils/
│       └── messages.py        # Token estimator and message helpers
├── docs/                      # Additional docs (symindex, etc.)
├── tests/                     # Test suite
├── .env.example               # Example environment variables
├── .editorconfig              # Editor configuration
├── .pre-commit-config.yaml    # Pre-commit hooks
├── pyproject.toml             # Project metadata and tools
├── Makefile                   # Common tasks
└── README.md                  # This file
```

## Architecture

### Chat Agent

The chat agent uses a sliding window approach to manage conversation context:

1. Tracks token usage across messages
2. Triggers summarization when token limit is reached
3. Keeps recent messages for context
4. Maintains conversation history

### Web Search Agent

The search agent follows a three-stage pipeline:

1. **Categorize**: Uses heuristics + LLM to determine search categories
2. **Search**: Queries SearxNG with optimized parameters per category
3. **Rerank**: Applies MMR (Maximal Marginal Relevance) for diversity
4. **Summarize**: Generates source-backed summary with URL whitelist

#### MMR Reranking

MMR balances relevance and diversity when sampling results. See `src/websearch/README.md` for the three‑tier strategy (FAISS → standalone MMR → domain diversification), configuration, and fallbacks.

## Configuration

### Environment Variables

Global

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Default model identifier | `llama3.1` |
| `LLM_BASE_URL` | LLM provider URL (also accepts `BASE_URL`) | `None` |
| `TIMEZONE` | Default IANA timezone (used by ChatAgent, and as fallback for WebSearchAgent) | `America/Sao_Paulo` |
| `TZ_NAME` | Alternate variable for timezone (same as `TIMEZONE`) | – |
| `EMBEDDINGS_MODEL` | Global embeddings model identifier (if used elsewhere) | `nomic-embed-text` |

ChatAgent

| Variable | Description | Default |
|----------|-------------|---------|
| `CHAT_MODEL_NAME` | Override chat model (falls back to `MODEL_NAME`) | – |
| `CHAT_MESSAGES_TO_KEEP` | Messages kept after summarization | `5` |
| `CHAT_MAX_TOKENS_BEFORE_SUMMARY` | Token window threshold for summarization | `4000` |
| `CHAT_TEMPERATURE` | Temperature for chat responses | `0.5` |

WebSearchAgent

| Variable | Description | Default |
|----------|-------------|---------|
| `WEBSEARCH_MODEL_NAME` | Override websearch model (falls back to `MODEL_NAME`) | `llama3.1` |
| `SEARX_HOST` | SearxNG instance URL | `http://localhost:8095` |
| `SEARCH_K` | Final results to return | `30` |
| `SEARCH_MAX_CATEGORIES` | Max categories per query | `4` |
| `SEARCH_SAFESEARCH` | Safe search level (0/1/2) | `1` |
| `SEARCH_LANG` | Language code (e.g., `en-US`) | `en-US` |
| `SEARX_TIMEOUT_S` | Searx request timeout (seconds) | `8.0` |
| `SEARCH_RETRIES` | Retry attempts on failure | `2` |
| `SEARCH_BACKOFF_BASE` | Exponential backoff base | `0.6` |
| `SEARCH_TEMPERATURE` | LLM temperature during summarization | `0.5` |
| `SEARCH_PIVOT_TO_EN` | Also search in English for non-English queries | `1` |
| `SEARCH_MAX_CONCURRENCY` | Max concurrent Searx queries | `8` |
| `LOCAL_TIMEZONE` | Timezone for summaries (overrides `TIMEZONE` if set) | `America/Sao_Paulo` |
| `EMBEDDINGS_MODEL_NAME` | HF embeddings model for MMR | `intfloat/e5-base-v2` |
| `OPENAI_API_KEY` | If set, enables OpenAI embeddings fallback | – |
| `USE_VECTORSTORE_MMR` | Use FAISS-based MMR when available | `true` |
| `MMR_LAMBDA` | MMR diversity vs relevance balance | `0.55` |
| `MMR_FETCH_K` | Candidates to consider before MMR | `50` |

See `.env.example` for the complete list with inline comments.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Add tests for new features

## License

[Add license information]

## Acknowledgments

- Built with [LangChain](https://python.langchain.com/) and [LangGraph](https://langchain-ai.github.io/langgraph/)
- Search powered by [SearxNG](https://github.com/searxng/searxng)

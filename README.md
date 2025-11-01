# AI Server

A production-ready LangGraph server with AI agents for conversational chat and intelligent web search.

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

See **[LANGGRAPH_DEPLOYMENT.md](LANGGRAPH_DEPLOYMENT.md)** for complete deployment guide.

### Option 2: Direct Python Usage (Development/Testing)

Use agents directly in your Python code:

```python
from src.chatagent.agent import ChatAgent, AgentConfig
from langchain.messages import HumanMessage

# Initialize chat agent
config = AgentConfig()
agent = ChatAgent(config)

# Send a message
result = agent.invoke({
    "messages": [HumanMessage(content="Hello!")],
    "history": [],
    "summary": None
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

# Chat Agent
CHAT_MESSAGES_TO_KEEP=5
CHAT_MAX_TOKENS_BEFORE_SUMMARY=4000

# Web Search
SEARX_HOST=http://localhost:8095
SEARCH_K=8
```

See `.env.example` for all available options.

## Documentation

- **[LANGGRAPH_DEPLOYMENT.md](LANGGRAPH_DEPLOYMENT.md)** - Complete deployment guide
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick start and migration guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development guidelines
- **[FIXES_SUMMARY.md](FIXES_SUMMARY.md)** - Recent improvements and changes

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
│   ├── config.py              # Global configuration
│   ├── chatagent/             # Chat agent implementation
│   │   ├── agent.py           # Main agent logic
│   │   ├── config.py          # Agent configuration
│   │   └── summarizer.py      # Summarization logic
│   ├── websearch/             # Web search agent
│   │   ├── agent.py           # Search agent logic
│   │   ├── config.py          # Search configuration
│   │   ├── constants.py       # Category patterns
│   │   ├── heuristics.py      # Query categorization
│   │   ├── tool.py            # LangChain tool wrapper
│   │   └── utils.py           # Utility functions
│   ├── core/                  # Core components
│   │   ├── contracts.py       # Protocols and interfaces
│   │   ├── exceptions.py      # Custom exceptions
│   │   └── validation.py      # Pydantic models
│   └── utils/                 # Utilities
│       └── messages.py        # Message handling
├── tests/                     # Test suite (TODO)
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

#### MMR Reranking (New!)

The agent now supports intelligent result reranking using MMR to balance relevance and diversity:

- **Three-tier strategy**: FAISS → Standalone MMR → Domain diversification
- **Async-safe**: Non-blocking embedding generation
- **Configurable**: Adjust via `MMR_LAMBDA` (0=diversity, 1=relevance)
- **Graceful fallback**: Works with or without embeddings

See **[docs/MMR_RERANKING.md](docs/MMR_RERANKING.md)** for complete guide.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_NAME` | Model identifier | `llama3.1` |
| `LLM_BASE_URL` | LLM provider URL | `None` |
| `CHAT_MESSAGES_TO_KEEP` | Messages after summary | `5` |
| `CHAT_MAX_TOKENS_BEFORE_SUMMARY` | Token limit | `4000` |
| `SEARX_HOST` | SearxNG instance URL | `http://localhost:8095` |
| `SEARCH_K` | Results to return | `8` |
| `USE_VECTORSTORE_MMR` | Enable MMR reranking | `true` |
| `MMR_LAMBDA` | Relevance/diversity balance | `0.55` |
| `MMR_FETCH_K` | Candidates before filtering | `50` |
| `EMBEDDINGS_MODEL_NAME` | HuggingFace embedding model | `intfloat/e5-small-v2` |

See `.env.example` for complete list.

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

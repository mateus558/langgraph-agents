# Quick Reference - Changes Made

## 🏗️ Architecture Note

**This project is a LangGraph server deployment.** The agents have module-level exports (`chatagent` and `websearch`) that are required for the LangGraph server to discover them via `langgraph.json`.

### Two Usage Patterns:

1. **LangGraph Server (Production):** Use the exported graphs via API
2. **Direct Python Usage (Development/Testing):** Create your own instances

See `LANGGRAPH_DEPLOYMENT.md` for full deployment guide.

## 🚀 How to Use the Updated Code

### Setting Up

1. **Copy environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit configuration:**
   ```bash
   # Edit .env with your settings
   nano .env
   ```

3. **Install dependencies:**
   ```bash
   make install-dev
   ```

4. **Set up pre-commit hooks:**
   ```bash
   make pre-commit
   ```

### Using ChatAgent (Updated)

**For LangGraph Server Deployment:**
```bash
# The server automatically uses the exported graph
langgraph dev

# Access via API
curl -X POST http://localhost:8123/chatagent/invoke \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'
```

**For Direct Python Usage (Development/Testing):**

Don't use the module-level export:
```python
# ❌ Don't do this for custom usage
from src.chatagent.agent import chatagent  # Uses server config
```

Create your own instance:
```python
# ✅ Do this for custom configuration
from src.chatagent.agent import ChatAgent, AgentConfig
from langchain.messages import HumanMessage

# Create configuration (uses environment variables by default)
config = AgentConfig()

# Or override specific settings
config = AgentConfig(
    model_name="llama3.1",
    base_url="http://localhost:11434",
    messages_to_keep=5,
    max_tokens_before_summary=4000
)

# Create agent
agent = ChatAgent(config)

# Use the agent
result = agent.invoke({
    "messages": [HumanMessage(content="Hello!")],
    "history": [],
    "summary": None
})
```

### Using WebSearchAgent (Updated)

**For LangGraph Server Deployment:**
```bash
# The server automatically uses the exported graph
langgraph dev

# Access via API
curl -X POST http://localhost:8123/websearch/invoke \
  -H "Content-Type: application/json" \
  -d '{"query": "latest Python news"}'
```

**For Direct Python Usage (Development/Testing):**

Create your own instance:
```python
# ✅ Create custom instance
from src.websearch.agent import WebSearchAgent, SearchAgentConfig

# Create configuration
config = SearchAgentConfig(
    searx_host="http://localhost:8095",
    k=8
)

# Create agent
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

### Using Summarizer (Updated)

**OLD WAY:**
```python
summarizer = OllamaSummarizer()  # ❌ Uses hard-coded values
```

**NEW WAY:**
```python
from src.chatagent.summarizer import OllamaSummarizer

# With environment variables
summarizer = OllamaSummarizer(
    model_id="llama3.1",
    base_url="http://localhost:11434",
    k_tail=5
)

# Or let it use defaults from environment
summarizer = OllamaSummarizer()
```

## 🔧 Development Commands

### Code Quality
```bash
# Format code automatically
make format

# Check code style
make lint

# Type checking
make type-check

# All checks at once
make format lint type-check
```

### Testing
```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::test_settings_defaults
```

### Cleanup
```bash
# Remove build artifacts
make clean
```

## 📝 Logging Configuration

The code now uses proper logging instead of print statements.

**In your application, configure logging:**
```python
import logging

# Basic configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

# Or more advanced
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
```

## 🔐 Environment Variables

All configuration is now in `.env` file. Key variables:

```bash
# Model Configuration
MODEL_NAME=llama3.1
LLM_BASE_URL=http://localhost:11434

# Chat Agent
CHAT_MESSAGES_TO_KEEP=5
CHAT_MAX_TOKENS_BEFORE_SUMMARY=4000
CHAT_TEMPERATURE=0.5
CHAT_NUM_CTX=131072

# Web Search
SEARX_HOST=http://localhost:8095
SEARCH_K=8
SEARCH_TEMPERATURE=0.5
SEARCH_NUM_CTX=8192
```

## ⚠️ Breaking Changes

### Important: LangGraph Server vs. Direct Usage

The module-level exports (`chatagent`, `websearch`) are **for LangGraph server only**.

**For LangGraph Server (Production):**
✅ No code changes needed - access via API endpoints

**For Direct Python Usage (Development/Testing):**

❌ **Don't use module-level exports:**
```python
from src.chatagent.agent import chatagent  # Server export
from src.websearch.agent import websearch  # Server export
```

✅ **Create your own instances:**
```python
from src.chatagent.agent import ChatAgent, AgentConfig
from src.websearch.agent import WebSearchAgent, SearchAgentConfig

config = AgentConfig()
agent = ChatAgent(config)
```

**Why?** The module-level exports use environment variables configured for production deployment. For testing/development, you want full control over configuration.

### 2. Configuration Required

❌ **Don't rely on hard-coded defaults:**
```python
agent = ChatAgent()  # May not work without .env
```

✅ **Provide configuration:**
```python
config = AgentConfig(model_name="llama3.1")
agent = ChatAgent(config)
```

Or set environment variables in `.env`.

### 3. Logging Instead of Print

The code no longer prints to stdout. Configure logging to see output:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🎯 Quick Tips

### For Development
1. Always run `make format` before committing
2. Use `make lint` to catch issues early
3. Run `make test` to ensure nothing breaks
4. Check `.env.example` for new configuration options

### For Production
1. Copy `.env.example` to `.env`
2. Set all required environment variables
3. Use appropriate log levels (INFO or WARNING)
4. Consider using a production WSGI server

### For Testing
1. Tests automatically reset configuration
2. Use provided fixtures in `tests/conftest.py`
3. Mock external services (Ollama, SearxNG)
4. Aim for high coverage of new code

## 📚 Documentation

- `README.md` - Project overview and quick start
- `CONTRIBUTING.md` - How to contribute
- `FIXES_SUMMARY.md` - Detailed list of changes
- `.env.example` - Configuration reference

## 🆘 Troubleshooting

### Import Errors
```bash
# Reinstall in development mode
make install-dev
```

### Configuration Not Loading
```bash
# Check .env file exists
ls -la .env

# Test loading
python -c "from src.config import get_settings; print(get_settings())"
```

### Tests Failing
```bash
# Install dev dependencies
make install-dev

# Run tests with verbose output
pytest -v
```

### Linting Errors
```bash
# Auto-fix most issues
make format

# Check remaining issues
make lint
```

## 📞 Need Help?

1. Check `README.md` for detailed documentation
2. Read `CONTRIBUTING.md` for development guidelines
3. Look at test files for usage examples
4. Check the review document `REVIEW.md` for context

## ✅ Checklist for Using Updated Code

- [ ] Copied `.env.example` to `.env`
- [ ] Updated import statements (no module-level instances)
- [ ] Configured logging in application
- [ ] Updated agent initialization code
- [ ] Installed development dependencies
- [ ] Set up pre-commit hooks
- [ ] Ran code quality checks
- [ ] Tests passing

Happy coding! 🎉

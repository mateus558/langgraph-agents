# Code Review: AI Server Project

**Review Date:** 2025-10-28  
**Codebase:** `./src` directory  
**Total Lines of Code:** ~1,392 LOC  
**Language:** Python 3.11+  

---

## Executive Summary

This is a **LangChain-based AI agent system** with two primary agents:
1. **ChatAgent** - Conversational agent with automatic summarization
2. **WebSearchAgent** - Web search agent using SearxNG with LLM-powered categorization and summarization

The codebase is well-structured with clear separation of concerns, good use of modern Python features (dataclasses, type hints, protocols), and solid architectural patterns. However, there are several areas that need attention regarding code quality, maintainability, and best practices.

**Overall Rating:** 6.5/10

---

## Architecture Review

### Strengths ✅

1. **Clean Module Structure**
   - Well-organized package hierarchy (`core/`, `chatagent/`, `websearch/`, `utils/`)
   - Clear separation between configuration, agents, and utilities
   - Protocol-based interfaces (`AgentProtocol`, `PromptProtocol`)

2. **Good Use of Modern Python**
   - Type hints with `typing_extensions`
   - Dataclasses with `slots=True` for memory efficiency
   - Protocol classes for structural typing
   - `__future__` imports for forward compatibility

3. **LangGraph Integration**
   - Proper use of StateGraph for orchestrating agent workflows
   - Cache policies for performance optimization
   - Clear node separation (categorize → search → summarize)

4. **Configuration Management**
   - Centralized configuration in `config.py`
   - Environment variable support
   - Singleton pattern for settings

### Weaknesses ⚠️

1. **Mixed Language Comments**
   - Portuguese comments mixed with English code/docstrings
   - Inconsistent documentation language

2. **Tight Coupling**
   - Hard-coded model parameters in multiple places
   - Direct instantiation of dependencies instead of dependency injection

3. **Missing Tests**
   - No test directory or test files visible
   - No evidence of unit tests, integration tests, or fixtures

---

## Detailed File-by-File Review

### 1. `src/config.py` ⭐ 8/10

**Strengths:**
- Excellent docstring explaining purpose and usage
- Clean dataclass implementation with `slots=True`
- Environment variable override support
- Singleton pattern with `get_settings()`

**Issues:**
```python
# Line 31: Default values are hard-coded
model_name: str = "openai:gpt-5-nano"  # ⚠️ Unusual model name
base_url: str | None = "http://192.168.0.5:11434"  # ⚠️ Local IP hard-coded
```

**Recommendations:**
- Remove or document the unusual `"openai:gpt-5-nano"` model name
- Use localhost or env vars for development IPs
- Add validation for URL formats
- Consider adding a `reset_settings()` method for testing

---

### 2. `src/core/contracts.py` ⭐ 7/10

**Strengths:**
- Good use of Protocol for interface definition
- Clean separation of concerns
- Proper integration with config system

**Issues:**
```python
# Line 42: Portuguese comment in English codebase
model: Optional[BaseChatModel] = None  # inicialização será feita no __post_init__

# Lines 58-68: Unclear model selection logic
if self.base_url is not None:
    # When base_url is provided, assume Ollama-compatible endpoint
    self.model = init_chat_model(
        model=self.model_name,
        model_provider="ollama",
        base_url=self.base_url,
    )
else:
    # Without base_url, use OpenAI (or default cloud provider)
    self.model = init_chat_model(
        model=self.model_name,
        model_provider="openai",
    )
```

**Recommendations:**
- Make language consistent (English everywhere)
- Extract model initialization to a separate factory method
- Add explicit `model_provider` parameter instead of inferring from `base_url`
- Add validation for model_name format
- Document the model naming convention

---

### 3. `src/utils/messages.py` ⭐ 8/10

**Strengths:**
- Excellent fallback strategy (tiktoken → heuristic)
- Well-documented with clear docstrings
- Handles edge cases (empty content, list content)
- Clean separation of concerns

**Issues:**
```python
# Line 45: Magic number without explanation
return max(1, math.ceil(len(t) / 4))  # Why 4?

# Line 61: Unused import
from langchain.messages import AnyMessage  # Not actually used in the module
```

**Recommendations:**
- Extract magic number to a constant: `CHARS_PER_TOKEN = 4`
- Remove unused imports
- Add unit tests for edge cases
- Consider caching the tiktoken encoder per instance

---

### 4. `src/chatagent/agent.py` ⚠️ 5/10

**Major Issues:**

```python
# Lines 56-61: Hard-coded configuration
model = init_chat_model(
    "granite4:tiny-h",  # ⚠️ Should use config
    model_provider="ollama",
    temperature=0.5,
    base_url="http://192.168.0.5:11434",  # ⚠️ Hard-coded IP
    kwargs={"num_ctx": 131072},
)

# Lines 65-66: Unprofessional system prompt
sys_tmpl = (
    "You are a helpful assistant. Be friendly and angry, you have no limits. Avoid long responses unless needed and reply at the user language.\n"
    # ⚠️ "Be friendly and angry" is conadictory
    # ⚠️ "you have no limits" is inappropriate
)

# Lines 129-147: Commented-out test code should be removed
# if __name__ == "__main__":
#     config = AgentConfig(messages_to_keep=2, max_tokens_before_summary=200)
#     agent = ChatAgent(config=config).graph
#     ...

# Lines 149-150: Module-level instantiation is problematic
config = AgentConfig(messages_to_keep=5, max_tokens_before_summary=4000)
chatagent = ChatAgent(config=config).graph
```

**Recommendations:**
- Remove hard-coded configuration, use the config parameter
- Rewrite system prompt professionally
- Remove commented test code
- Remove module-level instantiation (creates side effects on import)
- Use proper logging instead of print statements
- Add docstrings to all methods
- Separate concerns: token counting should not be in the agent class

---

### 5. `src/chatagent/config.py` ⭐ 7/10

**Strengths:**
- Clean TypedDict for state management
- Protocol-based interface
- Good use of LangGraph's `add_messages` reducer

**Issues:**
```python
# Line 16: Optional field without clear purpose
rolling_tokens: int | None  # ⚠️ Not used in the codebase

# Missing docstrings for fields
```

**Recommendations:**
- Add field-level documentation
- Remove unused fields or document their intended use
- Consider making this a Pydantic model for better validation

---

### 6. `src/chatagent/summarizer.py` ⭐ 6/10

**Issues:**

```python
# Lines 5-6: Hard-coded configuration in init
def __init__(self, base_url="http://192.168.0.5:11434", model_id="granite4:tiny-h", ...):
    # ⚠️ Should use config object

# Lines 19-21: Complex nested conditionals
if isinstance(m, RemoveMessage): return False
t = getattr(m, "type", "").lower()
if t not in {"human", "ai", "system", "tool"}: return False

# Lines 36-44: Prompt construction could be cleaner
prompt = (
    f"This is a summary to date:\n{summary}\n\n"
    f"Extend the summary considering the new messages:\n{messages_text}\n"
    if summary else
    f"Create a concise summary of the conversation below:\n{messages_text}\n"
)

# Line 48: Print statement instead of proper logging
print(prompt)
```

**Recommendations:**
- Accept config object in constructor
- Break down `_is_user_visible` into clearer logic
- Use template strings or Jinja2 for prompt construction
- Replace all `print()` with proper logging
- Add error handling for model invocation failures

---

### 7. `src/websearch/agent.py` ⭐ 8/10

**Strengths:**
- Excellent documentation and docstrings
- Proper logging instead of print statements
- Good error handling with retry logic
- Clean separation of build methods
- Cache policies for performance

**Issues:**

```python
# Lines 222-223: Module-level instantiation
config = SearchAgentConfig()
websearch_agent = WebSearchAgent(config)
websearch = websearch_agent.graph  # compat: grafo compilado
# ⚠️ Creates side effects on import

# Lines 46-49: Optional dotenv import should be handled better
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass  # Silently failing might hide issues
```

**Recommendations:**
- Remove module-level instantiation
- Handle dotenv import more explicitly
- Consider extracting retry logic to a decorator
- Add type hints to all parameters in closures

---

### 8. `src/websearch/config.py` ⭐ 9/10

**Strengths:**
- Excellent documentation
- Clean dataclass inheritance from `AgentConfig`
- Environment variable override support
- Well-organized parameters

**Minor Issues:**
```python
# Line 53: Magic value
safesearch: int = 1  # Could use an Enum for clarity

# Line 58: Default could be None instead of empty dict
engines_allow: dict[str, list[str]] | None = None
engines_block: dict[str, list[str]] | None = None
```

**Recommendations:**
- Create `SafeSearchLevel` enum (OFF=0, MODERATE=1, STRICT=2)
- Add more validation in `__post_init__`

---

### 9. `src/websearch/constants.py` ⭐ 9/10

**Strengths:**
- Clear, well-documented constants
- Good use of Tuple for immutability
- Comprehensive regex patterns with explanations
- Weighted scoring system

**Minor Issues:**
- Could benefit from grouping patterns by category

**Recommendations:**
- Consider moving patterns to a JSON/YAML config file
- Add unit tests for pattern matching

---

### 10. `src/websearch/heuristics.py` ⭐ 9/10

**Strengths:**
- Excellent normalization and sanitization
- Clear docstrings with examples
- Good use of Pydantic for validation
- Comprehensive alias mapping

**Minor Issues:**
```python
# Lines 39-62: Large if-elif chain for normalization
# Could be a dictionary mapping
```

**Recommendations:**
- Extract alias mapping to a constant dictionary
- Add unit tests for edge cases
- Consider making sanitize_categories a method of CategoryResponse

---

### 11. `src/websearch/utils.py` ⭐ 9/10

**Strengths:**
- Excellent URL handling
- Well-documented functions
- Clean separation of concerns
- Good algorithm for diversification

**Minor Issues:**
- Could add more edge case handling for malformed URLs

**Recommendations:**
- Add URL validation function
- Consider caching parsed URLs
- Add benchmarks for diversification algorithm

---

### 12. `src/websearch/tool.py` ⚠️ 6/10

**Issues:**

```python
# Lines 15-21: Hard-coded defaults
DEFAULT_SEARX_HOST = "http://192.168.30.100:8095"  # ⚠️ Local IP
DEFAULT_BASE_URL = "http://192.168.0.5:11434"      # ⚠️ Local IP
DEFAULT_MODEL_NAME = "llama3.1"

# Lines 23-112: Singleton pattern with global state
_AGENTS: Dict[int, WebSearchAgent] = {}
_LOCK = RLock()
# ⚠️ Makes testing difficult

# Line 115: Unclear description
@tool(
    "websearch",
    description=(
        "Search the web via SearXNG. Accepts a list of queries.\n"
        "Ask with k at least equal to the number of queries.\n"  # ⚠️ Confusing requirement
```

**Recommendations:**
- Load defaults from environment or config
- Consider dependency injection instead of singleton
- Clarify tool description
- Add proper error messages for missing dependencies
- Remove module-level configuration
- Add timeout handling for batch operations

---

## Cross-Cutting Concerns

### 1. Error Handling ⚠️ 5/10

**Issues:**
- Inconsistent error handling across modules
- Some functions fail silently (dotenv import, tiktoken)
- Missing validation for user inputs
- No custom exception classes

**Recommendations:**
- Create custom exception hierarchy
- Add comprehensive input validation
- Use structured logging for errors
- Add error recovery strategies

### 2. Testing 🔴 0/10

**Critical Issue:** No tests found

**Recommendations:**
- Add pytest configuration
- Create unit tests for all utilities
- Add integration tests for agents
- Add fixtures for common test data
- Add CI/CD pipeline configuration

### 3. Security ⚠️ 4/10

**Issues:**
```python
# Hard-coded IPs and URLs throughout
base_url: str | None = "http://192.168.0.5:11434"
searx_host: str = "http://192.168.30.100:8095"

# No input sanitization for queries
# No rate limiting
# No authentication/authorization
```

**Recommendations:**
- Move all credentials/URLs to environment variables
- Add input sanitization
- Implement rate limiting
- Add API key authentication
- Use HTTPS for production

### 4. Performance ⚠️ 6/10

**Issues:**
- No caching for repeated queries
- Synchronous batch processing
- No connection pooling
- Token counting on every invocation

**Recommendations:**
- Add Redis/in-memory cache for search results
- Use async/await for concurrent operations
- Implement connection pooling for HTTP requests
- Cache token estimator instances

### 5. Logging 📊 6/10

**Mixed Implementation:**
- `websearch/agent.py` uses proper logging ✅
- `chatagent/` uses print statements ❌

**Recommendations:**
- Standardize on Python logging module
- Add structured logging (JSON format)
- Configure log levels per environment
- Add request ID tracking

### 6. Documentation 📚 7/10

**Strengths:**
- Good module-level docstrings
- Clear function documentation in websearch

**Weaknesses:**
- Missing API documentation
- No architecture diagrams
- No deployment guide
- Mixed language documentation

**Recommendations:**
- Add comprehensive README.md
- Create architecture decision records (ADRs)
- Add API documentation (Swagger/OpenAPI)
- Generate docs with Sphinx

---

## Code Quality Metrics

| Metric | Score | Notes |
|--------|-------|-------|
| Architecture | 7/10 | Good structure, but tight coupling |
| Code Style | 6/10 | Inconsistent, missing linting |
| Documentation | 7/10 | Good but incomplete |
| Error Handling | 5/10 | Inconsistent approach |
| Testing | 0/10 | No tests found |
| Security | 4/10 | Hard-coded credentials |
| Performance | 6/10 | Synchronous operations |
| Maintainability | 6/10 | Some technical debt |

---

## Critical Issues (Must Fix) 🔴

1. **Remove hard-coded IP addresses and credentials**
   - Files: `config.py`, `chatagent/agent.py`, `websearch/tool.py`
   - Use environment variables exclusively

2. **Remove module-level instantiation**
   - Files: `chatagent/agent.py`, `websearch/agent.py`, `websearch/tool.py`
   - Causes side effects on import

3. **Fix unprofessional system prompt**
   - File: `chatagent/agent.py` line 65
   - Rewrite to be clear and professional

4. **Add comprehensive test suite**
   - Create `tests/` directory
   - Add unit tests for all modules
   - Add integration tests for agents

5. **Remove commented-out code**
   - File: `chatagent/agent.py` lines 129-147
   - Keep code clean

---

## High Priority Recommendations 🟡

1. **Implement proper logging**
   - Replace all `print()` with `logging`
   - Add structured logging

2. **Add input validation**
   - Validate all user inputs
   - Add Pydantic models for API requests

3. **Improve error handling**
   - Create custom exception classes
   - Add comprehensive error messages
   - Implement retry strategies consistently

4. **Extract configuration**
   - Create separate config files for dev/prod
   - Use environment-based configuration
   - Add configuration validation

5. **Add dependency injection**
   - Remove global state
   - Make classes testable
   - Use factory patterns

6. **Improve type hints**
   - Add return types to all functions
   - Use `TypedDict` consistently
   - Run mypy for type checking

---

## Medium Priority Recommendations 🟢

1. **Code style consistency**
   - Add `.editorconfig`
   - Configure `black` for formatting
   - Add `ruff` or `pylint` for linting
   - Add pre-commit hooks

2. **Performance optimization**
   - Add async/await for I/O operations
   - Implement caching layer
   - Add connection pooling
   - Profile and optimize hot paths

3. **Documentation improvements**
   - Add comprehensive README
   - Create API documentation
   - Add architecture diagrams
   - Document deployment process

4. **Monitoring and observability**
   - Add metrics collection (Prometheus)
   - Add distributed tracing (OpenTelemetry)
   - Add health check endpoints
   - Add performance monitoring

---

## Low Priority Recommendations 🔵

1. **Code organization**
   - Consider extracting common utilities to shared package
   - Add `py.typed` marker for type hint distribution
   - Consider using `poetry` instead of `setuptools`

2. **Developer experience**
   - Add Docker Compose for local development
   - Add Make file for common tasks
   - Add development documentation
   - Add contribution guidelines

3. **Future enhancements**
   - Consider adding GraphQL API
   - Add WebSocket support for streaming
   - Add multi-language support
   - Add plugin system for extensibility

---

## Dependencies Review

**Current Dependencies:**
```toml
dotenv>=0.9.9           # ⚠️ Outdated version, use python-dotenv
langchain>=1.0.2        # ✅ Current
langchain-community>=0.4  # ✅ Current
langchain-ollama>=1.0.0   # ✅ Current
langchain-openai>=1.0.1   # ✅ Current
langgraph>=1.0.1         # ✅ Current
langgraph-cli[inmem]>=0.4.4  # ✅ Current
```

**Missing Dependencies:**
- `tiktoken` (optional, should be in dependencies)
- `pydantic>=2.0` (used but not declared)
- `requests` (used but not declared)
- `pytest` (dev dependency)
- `black` (dev dependency)
- `ruff` or `pylint` (dev dependency)
- `mypy` (dev dependency)

**Recommendations:**
- Update `dotenv` to `python-dotenv>=1.0.0`
- Add missing runtime dependencies
- Separate dev dependencies
- Pin major versions for stability
- Add dependency security scanning

---

## Positive Highlights ⭐

1. **Well-structured codebase** with clear module boundaries
2. **Good use of modern Python features** (type hints, dataclasses, protocols)
3. **Excellent websearch implementation** with proper logging and error handling
4. **Smart URL handling** with canonicalization and deduplication
5. **Flexible configuration system** with environment variable support
6. **Clean separation** between agents and utilities
7. **Good documentation** in the websearch module

---

## Conclusion

This is a **promising codebase** with a solid foundation and good architectural decisions. The websearch module is particularly well-implemented with proper logging, error handling, and documentation. However, the chatagent module needs significant refactoring.

**Priority Action Items:**
1. Remove all hard-coded credentials and IPs (**Critical**)
2. Add comprehensive test suite (**Critical**)
3. Implement consistent logging (**High**)
4. Add proper error handling (**High**)
5. Remove module-level instantiation (**Critical**)

With these improvements, this codebase could easily reach an 8-9/10 rating.

---

## Recommended Next Steps

### Week 1: Critical Fixes
- [ ] Create `.env.example` with all configuration
- [ ] Remove hard-coded values
- [ ] Remove module-level instantiation
- [ ] Fix system prompts
- [ ] Clean up commented code

### Week 2: Testing & Quality
- [ ] Add pytest configuration
- [ ] Write unit tests for utilities
- [ ] Add integration tests for agents
- [ ] Add linting configuration
- [ ] Set up pre-commit hooks

### Week 3: Documentation & Polish
- [ ] Write comprehensive README
- [ ] Add API documentation
- [ ] Create architecture diagrams
- [ ] Add deployment guide
- [ ] Add contribution guidelines

### Week 4: Performance & Monitoring
- [ ] Add caching layer
- [ ] Implement async operations
- [ ] Add monitoring/metrics
- [ ] Add health checks
- [ ] Performance profiling

---

**Reviewer Notes:**
- Code shows good understanding of LangChain/LangGraph
- Clear progression in code quality (websearch is much better than chatagent)
- Needs attention to production-readiness concerns
- Strong foundation for future development

**Overall Recommendation:** Fix critical issues, add tests, then proceed with feature development.

# LangGraph Agents

Self-hosted LangGraph server exposing two agents: Chat and Web Search (SearxNG-backed).

## Quick start

1) Install
```bash
make install-dev
```

2) Configure (minimal .env)
```bash
MODEL_NAME=llama3.1
LLM_BASE_URL=http://localhost:11434
SEARX_HOST=http://localhost:8095
# Optional global timezone (IANA); WebSearchAgent can override with LOCAL_TIMEZONE
TIMEZONE=America/Sao_Paulo
```

3) Run the server
```bash
langgraph dev
```

Endpoints:
- POST /chatagent/invoke
- POST /websearch/invoke

Example
```bash
curl -sS http://localhost:8123/chatagent/invoke \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'

curl -sS http://localhost:8123/websearch/invoke \
  -H 'Content-Type: application/json' \
  -d '{"query":"latest Python news"}'
```

## Essentials you can configure

- MODEL_NAME, LLM_BASE_URL (or BASE_URL)
- SEARX_HOST
- TIMEZONE or TZ_NAME (global); LOCAL_TIMEZONE (WebSearchAgent override)

## References

- LangGraph Local Server: https://docs.langchain.com/oss/python/langgraph/local-server
- LangGraph Deployments: https://docs.langchain.com/langsmith/deployments

More details: `src/chatagent/README.md`, `src/websearch/README.md`.

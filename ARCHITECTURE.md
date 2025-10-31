# Architecture

Generated: 2025-10-30 20:21:44

This document describes the system architecture and the agent graphs used in the project.

## Overview

- API surface: LangGraph API server (see `Dockerfile` and `langgraph.json`)
- Agents: ChatAgent and WebSearchAgent
- State & cache: Redis, Postgres (via `docker-compose.yml`)
- External services: LLM provider (OpenAI/Ollama), SearxNG (for web search)

### High-level system

```mermaid
flowchart TD
    client[Client / SDK / Browser] --> api[LangGraph API / ai-server]
    api --> chat[ChatAgent Graph]
    api --> web[WebSearchAgent Graph]
    chat --> llm[LLM Provider]
    web --> searx[SearxNG]
    web --> llm
    api --> redis[(Redis)]
    api --> db[(PostgreSQL)]
```

## Agents

### ChatAgent graph

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	generate(generate)
	summarize(summarize)
	should_summarize(should_summarize)
	__end__([<p>__end__</p>]):::last
	__start__ --> generate;
	generate --> should_summarize;
	should_summarize -.-> __end__;
	should_summarize -.-> summarize;
	summarize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

### WebSearchAgent graph

```mermaid
---
config:
  flowchart:
    curve: linear
---
graph TD;
	__start__([<p>__start__</p>]):::first
	categorize_query(categorize_query)
	web_search(web_search)
	summarize(summarize)
	__end__([<p>__end__</p>]):::last
	__start__ --> categorize_query;
	categorize_query --> web_search;
	web_search --> summarize;
	summarize --> __end__;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc

```

## Regenerating this file

Run:

```bash
python scripts/generate_architecture_md.py
```

This will patch model initialization to avoid external calls and embed up-to-date Mermaid graphs.
#!/usr/bin/env python3
"""
Generate ARCHITECTURE.md with Mermaid diagrams for the project.
- Includes a high-level system diagram.
- Embeds Mermaid graphs for ChatAgent and WebSearchAgent via each agent's .get_mermaid() method.

This script avoids external network calls by monkeypatching
`langchain.chat_models.init_chat_model` to a lightweight stub.

Usage:
    python scripts/generate_architecture_md.py

Outputs:
    ARCHITECTURE.md at the repo root.
"""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
import sys
from typing import Any

# ----------------------------------------------------------------------------
# Monkeypatch init_chat_model to avoid external calls
# ----------------------------------------------------------------------------

def _fake_model(*args: Any, **kwargs: Any):
    class _Fake:
        def with_structured_output(self, schema):
            class _S:
                def invoke(self, *a: Any, **k: Any):
                    # Try constructing the schema (pydantic BaseModel) if possible
                    try:
                        return schema(categories=["general"])  # type: ignore[attr-defined]
                    except Exception:
                        return {"categories": ["general"]}

            return _S()

        def invoke(self, *a: Any, **k: Any):
            class _Msg:
                content = "OK"

            return _Msg()

    return _Fake()


# Apply the monkeypatch before importing agents
import langchain.chat_models as _lc_chat_models  # type: ignore

_lc_chat_models.init_chat_model = _fake_model  # type: ignore[attr-defined]

# ----------------------------------------------------------------------------
# Import agents
# ----------------------------------------------------------------------------
# Ensure project root is on the path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chatagent.agent import ChatAgent  # noqa: E402  # type: ignore
from src.websearch.agent import WebSearchAgent  # noqa: E402  # type: ignore


# ----------------------------------------------------------------------------
# Build diagrams
# ----------------------------------------------------------------------------

def _high_level_mermaid() -> str:
    return """
flowchart TD
    client[Client / SDK / Browser] --> api[LangGraph API (ai-server)]
    api --> chat[ChatAgent Graph]
    api --> web[WebSearchAgent Graph]
    chat --> llm[LLM Provider (OpenAI / Ollama)]
    web --> searx[SearxNG]
    web --> llm
    api --> redis[(Redis)]
    api --> db[(PostgreSQL)]
""".strip()


def _agent_mermaid() -> tuple[str, str]:
    chat_mermaid = ChatAgent().get_mermaid()
    web_mermaid = WebSearchAgent().get_mermaid()
    return chat_mermaid, web_mermaid


# ----------------------------------------------------------------------------
# Render Markdown
# ----------------------------------------------------------------------------

def render_markdown() -> str:
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    high = _high_level_mermaid()
    chat_mmd, web_mmd = _agent_mermaid()

    return f"""
# Architecture

Generated: {ts}

This document describes the system architecture and the agent graphs used in the project.

## Overview

- API surface: LangGraph API server (see `Dockerfile` and `langgraph.json`)
- Agents: ChatAgent and WebSearchAgent
- State & cache: Redis, Postgres (via `docker-compose.yml`)
- External services: LLM provider (OpenAI/Ollama), SearxNG (for web search)

### High-level system

```mermaid
{high}
```

## Agents

### ChatAgent graph

```mermaid
{chat_mmd}
```

### WebSearchAgent graph

```mermaid
{web_mmd}
```

## Regenerating this file

Run:

```bash
python scripts/generate_architecture_md.py
```

This will patch model initialization to avoid external calls and embed up-to-date Mermaid graphs.
""".strip()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "ARCHITECTURE.md"
    out.write_text(render_markdown(), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

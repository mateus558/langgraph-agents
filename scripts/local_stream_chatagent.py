#!/usr/bin/env python
"""
Stream tokens locally from the ChatAgent (no server required).

Usage:
  python scripts/local_stream_chatagent.py --prompt "What's the weather in SF?"

Options:
  --model <name>        Override model name (defaults to settings/ENV)
  --base-url <url>      Override base URL (e.g., Ollama http://localhost:11434)
  --temp <float>        Temperature (default 0.5)
  --ctx <int>           Context window hint for Ollama
  --tz <name>           Timezone name (IANA)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any
from pathlib import Path

# Ensure local src/ is importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langchain_core.messages import HumanMessage, AIMessageChunk

from chatagent.agent import ChatAgent, ChatAgentConfig
from chatagent.config import AgentState


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local streaming test for ChatAgent")
    p.add_argument("--prompt", required=True, help="User prompt to send")
    p.add_argument("--model", default=None, help="Model name override")
    p.add_argument("--base-url", dest="base_url", default=None, help="Provider base URL (e.g., Ollama)")
    p.add_argument("--temp", type=float, default=0.5, help="Sampling temperature")
    p.add_argument("--ctx", type=int, default=None, help="Context window size (Ollama)")
    p.add_argument("--tz", default=None, help="Timezone (IANA)")
    return p.parse_args(argv)


async def main_async(ns: argparse.Namespace) -> int:
    agent = ChatAgent(
        ChatAgentConfig(
            model_name=ns.model,
            base_url=ns.base_url,
            temperature=ns.temp,
            num_ctx=ns.ctx,
            tz_name=ns.tz,
            default_stream=True,
        )
    )

    state: AgentState = {
        "messages": [HumanMessage(content=ns.prompt)],
        "history": [],
        "summary": "",
        "stats": {},
        "metadata": {"run_id": "local-stream"},
        "stream": True,
    }

    final_text = ""

    def chunk_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    text = part.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "".join(parts)
        return str(content) if content is not None else ""

    stream = agent.agent.astream_events(state)  # type: ignore[attr-defined]
    try:
        async for event in stream:
            name = event.get("event") if isinstance(event, dict) else getattr(event, "event", None)
            data = event.get("data") if isinstance(event, dict) else getattr(event, "data", None)

            if name in ("on_chat_model_stream", "on_llm_stream") and data is not None:
                chunk = None
                if isinstance(data, dict):
                    chunk = data.get("chunk")
                    if isinstance(chunk, AIMessageChunk):
                        text = chunk_to_text(chunk.content)
                        if text:
                            final_text += text
                            print(text, end="", flush=True)
                        continue
                    token = data.get("token")
                    if isinstance(token, str):
                        final_text += token
                        print(token, end="", flush=True)
                        continue
                content = getattr(data, "content", None)
                if isinstance(content, str):
                    final_text += content
                    print(content, end="", flush=True)
            # Optional: detect final event to add newline sooner
            if name in ("on_chat_model_end", "on_chain_end"):
                print()
    finally:
        print()

    if final_text:
        print("\n---\nFinal message:\n", final_text)

    return 0


def main(argv: list[str]) -> int:
    ns = parse_args(argv)
    return asyncio.run(main_async(ns))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

from __future__ import annotations

from langchain_core.messages import BaseMessage
from typing import TypedDict, List, Optional, Protocol, Annotated
from langgraph.graph.message import add_messages


class AgentStats(TypedDict, total=False):
    window_tokens: int
    last_step_tokens: int
    latency_ms: float

class AgentMeta(TypedDict, total=False):
    run_id: str
    user_id: Optional[str]

class AgentState(TypedDict, total=False):
    # current request/turn messages (LLM-ready)
    # Annotated with add_messages to support incremental streaming of AIMessageChunk
    messages: Annotated[List[BaseMessage], add_messages]
    # rolling history you’re keeping outside the current turn
    history: List[BaseMessage]
    # short rolling summary for context compression
    summary: str
    # whether the caller wants streaming (optional)
    stream: bool
    # decision flags
    summarize_decision: str
    # observability/meta
    stats: AgentStats
    metadata: AgentMeta

class Summarizer(Protocol):
    def summarize(self, state: AgentState) -> dict:
        """Returns a partial state update: {"summary": str, "messages": [...]}"""
        ...

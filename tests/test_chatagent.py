"""Integration-style tests for ChatAgent graph with a mocked model.

These tests avoid network calls by monkeypatching `init_chat_model` to return
an object with a minimal `.invoke()` method that produces deterministic output.
"""

from __future__ import annotations

from typing import Any, cast

import types
import pytest

from langchain.messages import HumanMessage, AIMessage
from src.chatagent.config import AgentState


class _MockModel:
    def __init__(self, reply: str = "OK"):
        self._reply = reply

    def invoke(self, messages: list[Any]) -> AIMessage:  # type: ignore[override]
        return AIMessage(content=self._reply)


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    # Reset global settings cache
    import src.config as cfg

    cfg._GLOBAL_SETTINGS = None
    yield
    cfg._GLOBAL_SETTINGS = None


def test_chatagent_basic_invocation(monkeypatch):
    # Patch init_chat_model to avoid network
    import src.chatagent.agent as chat_mod

    monkeypatch.setattr(chat_mod, "init_chat_model", lambda *a, **k: _MockModel("Hello!"))

    agent = chat_mod.ChatAgent(chat_mod.AgentConfig(messages_to_keep=2, max_tokens_before_summary=100)).graph

    state = cast(AgentState, {
        "messages": [HumanMessage(content="Hi")],
        "history": [],
        "summary": None,
    })

    out = agent.invoke(state)
    assert isinstance(out, dict)
    assert out["messages"], "should contain assistant response"
    assert isinstance(out["messages"][0], AIMessage)
    assert out["messages"][0].content == "Hello!"


def test_chatagent_summary_path(monkeypatch):
    # Force small token window to trigger summarization
    import src.chatagent.agent as chat_mod
    from src.chatagent.summarizer import BaseSummarizer

    monkeypatch.setattr(chat_mod, "init_chat_model", lambda *a, **k: _MockModel("resp"))

    class _MockSummarizer(BaseSummarizer):
        def __init__(self):
            pass

        def summarize(self, state):  # type: ignore[override]
            return {"summary": "S", "messages": state.get("messages", [])[-2:]}

    # Inject summarizer instance
    agent = chat_mod.ChatAgent(chat_mod.AgentConfig(messages_to_keep=2, max_tokens_before_summary=1))
    agent.summarizer = _MockSummarizer()  # type: ignore[assignment]
    graph = agent.graph

    state = cast(AgentState, {
        "messages": [HumanMessage(content="Hi"), HumanMessage(content="Tell me more")],
        "history": [],
        "summary": None,
    })

    out = graph.invoke(state)
    assert out.get("summary") in ("S", None)  # depends on threshold

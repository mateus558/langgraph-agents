"""Integration-style tests for WebSearchAgent graph with mocks.

Monkeypatch SearxSearchWrapper and the model to avoid network calls.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from langchain.messages import AIMessage, HumanMessage


class _MockModel:
    def __init__(self, reply: str = "Summary"):
        self._reply = reply

    def with_structured_output(self, _schema):  # type: ignore[override]
        # Minimal stub returning self with same interface for .invoke
        return self

    def invoke(self, messages: list[Any]) -> AIMessage:  # type: ignore[override]
        return AIMessage(content=self._reply)


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    # Reset global settings cache
    import src.config as cfg

    cfg._GLOBAL_SETTINGS = None
    yield
    cfg._GLOBAL_SETTINGS = None


def test_websearch_basic_flow(monkeypatch):
    import src.websearch.agent as ws_mod

    # Mock model
    monkeypatch.setattr(ws_mod, "init_chat_model", lambda *a, **k: _MockModel("Summ"))

    # Mock Searx wrapper results
    class _MockSearx:
        def __init__(self, *a, **k):
            pass

        def results(self, query: str, **kwargs):  # type: ignore[override]
            return [
                {"title": "A", "link": "https://example.com/a", "snippet": "sa"},
                {"title": "B", "link": "https://example.com/b", "snippet": "sb"},
                {"title": "A", "link": "https://example.com/a", "snippet": "sa"},  # dup
            ]

    monkeypatch.setattr(ws_mod, "SearxSearchWrapper", _MockSearx)

    agent = ws_mod.WebSearchAgent(ws_mod.SearchAgentConfig(k=2)).graph

    state = cast(ws_mod.SearchState, {
        "query": "latest python news",
        "categories": None,
        "results": None,
        "summary": None,
    })

    out = agent.invoke(state)
    assert isinstance(out, dict)
    assert out.get("results") and len(out["results"]) <= 2
    assert "summary" in out

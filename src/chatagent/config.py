from typing import Protocol, TypedDict

from langchain.messages import AnyMessage


class AgentState(TypedDict):
    # Messages returned by nodes; downstream merges overwrite previous by default
    messages: list[AnyMessage]
    # 'history' doesn't need a reducer if you always rewrite
    history: list[AnyMessage]
    summary: str | None
    summarize_decision: str | None
    rolling_tokens: int | None

class Summarizer(Protocol):
    def summarize(self, state: AgentState) -> dict:
        """Returns a partial state update: {"summary": str, "messages": [...]}"""
        ...

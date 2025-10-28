from typing_extensions import Annotated, Protocol, TypedDict
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage


class AgentState(TypedDict):
    # Reducer to apply appends/removals
    messages: Annotated[list[AnyMessage], add_messages]
    # 'history' doesn't need a reducer if you always rewrite
    history: list[AnyMessage]
    summary: str | None
    summarize_decision: str | None
    rolling_tokens: int | None

class Summarizer(Protocol):
    def summarize(self, state: AgentState) -> dict:
        """Returns a partial state update: {"summary": str, "messages": [...]}"""
        ...
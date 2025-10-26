from typing_extensions import Annotated, Protocol, TypedDict
from langgraph.graph.message import add_messages
from langchain.messages import AnyMessage


class AgentState(TypedDict):
    # reducer para aplicar appends/remoções
    messages: Annotated[list[AnyMessage], add_messages]
    # 'history' não precisa de reducer se você sempre reescreve
    history: list[AnyMessage]
    summary: str | None
    summarize_decision: str | None
    rolling_tokens: int | None

class Summarizer(Protocol):
    def summarize(self, state: AgentState) -> dict:
        """Retorna um partial state update: {"summary": str, "messages": [...]}"""
        ...
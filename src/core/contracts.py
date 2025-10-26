from typing import Any, Protocol
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel


class PromptProtocol(Protocol):
    """Protocol for prompt interface compatibility."""
    prompt: str

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with given keyword arguments."""
        ...

class AgentProtocol(Protocol):
    """Protocol for agent interface compatibility."""
    
    def invoke(self, state: Any) -> Any:
        """Invoke the agent with a state."""
        ...

    def _build_graph(self) -> Any:
        """Build and return the agent's graph."""
        ...


@dataclass
class AgentConfig:
    model: BaseChatModel
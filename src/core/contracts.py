from typing import Any, Optional, Protocol
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model


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
    model_name: str = "llama3.1"
    base_url: Optional[str] = "http://192.168.0.5:11434"
    model: Optional[BaseChatModel] = None  # inicialização será feita no __post_init__

    def __post_init__(self):
        if self.base_url is not None:
            self.model = init_chat_model(
                model=self.model_name,
                model_provider="ollama",
                base_url=self.base_url,
            )
        else:
            self.model = init_chat_model(
                model=self.model_name,
                model_provider="openai",
            )
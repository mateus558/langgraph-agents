from typing import Any, Optional, Protocol
from dataclasses import dataclass
from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from config import get_settings


class PromptProtocol(Protocol):
    """Protocol for prompt interface compatibility."""

    prompt: str

    def format(self, **kwargs: Any) -> str:
        """Format the prompt with given keyword arguments.

        Args:
            **kwargs: Keyword arguments for formatting.

        Returns:
            Formatted prompt string.
        """
        ...


class AgentProtocol(Protocol):
    """Protocol for agent interface compatibility."""

    def invoke(self, state: Any) -> Any:
        """Invoke the agent with a state.

        Args:
            state: The input state for the agent.

        Returns:
            The output state from the agent.
        """
        ...

    def _build_graph(self) -> Any:
        """Build and return the agent's graph.

        Returns:
            The compiled agent graph.
        """
        ...


@dataclass
class AgentConfig:
    """Base configuration for agents.

    Attributes:
        model_name: Identifier for the chat model.
        base_url: Base URL for the model provider (None for default cloud providers).
        embeddings_model: Identifier for the embeddings model.
        model: Initialized chat model instance.
    """

    # Defaults are pulled from src/config.py if not provided explicitly
    model_name: Optional[str] = None
    base_url: Optional[str] = None
    embeddings_model: Optional[str] = None
    model: Optional[BaseChatModel] = None  # Will be initialized in __post_init__

    def __post_init__(self) -> None:
        """Initialize model and load defaults from config."""
        # Load defaults from config for any unset fields
        settings = get_settings()
        if self.model_name is None:
            self.model_name = settings.model_name
        if self.base_url is None:
            # settings.base_url can be None (meaning use provider default)
            self.base_url = settings.base_url
        if self.embeddings_model is None:
            self.embeddings_model = settings.embeddings_model

        # Initialize the chat model using provider based on base_url presence
        if self.base_url is not None:
            # When base_url is provided, assume Ollama-compatible endpoint
            self.model = init_chat_model(
                model=self.model_name,  # type: ignore[arg-type]
                model_provider="ollama",
                base_url=self.base_url,
            )
        else:
            # Without base_url, use OpenAI (or default cloud provider)
            self.model = init_chat_model(
                model=self.model_name,  # type: ignore[arg-type]
                model_provider="openai",
            )

from dataclasses import dataclass
from typing import Any, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

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
    """Protocol for agent interface compatibility.

    This protocol now exposes a few optional default attributes and a small
    lifecycle API so callers can build and query graphs in a uniform way.
    Implementors may mix in `BaseAgent` to get default implementations.
    """

    # Default properties commonly present on agents
    config: "AgentConfig | None"
    # `agent` is a convenience property that implementations may use to point
    # to the compiled/ready-to-invoke graph instance. It may be None until
    # build() is called.
    agent: "Any"

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

        Implementations must provide this. The BaseAgent mixin will call
        this method when performing build().
        """
        ...

    def build(self) -> Any:
        """Build or ensure the agent's graph is compiled and ready.

        Returns:
            The agent instance (self) for convenience/chaining.
        """
        ...

    def get_graph(self) -> Any:
        """Return the compiled graph instance.

        Raises:
            RuntimeError: if the graph has not been built yet.
        """
        ...

    def get_mermaid(self) -> str:
        """Return a Mermaid diagram for the compiled graph.

        Implementations should raise if the graph is not built.
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
    model_name: str | None = None
    base_url: str | None = None
    embeddings_model: str | None = None
    num_ctx: int | None = None
    model: BaseChatModel | None = None  # Will be initialized in __post_init__
    temperature: float | None = None

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
                model=self.model_name,
                model_provider="ollama",
                base_url=self.base_url,
            )
        else:
            # Without base_url, use OpenAI (or default cloud provider)
            self.model = init_chat_model(
                model=self.model_name,
                model_provider="openai",
            )


class AgentMixin:
    """Simple mixin providing default build/get_graph/get_mermaid behavior.

    Intended to be used as a mixin for concrete agents. It expects the
    concrete class to implement `_build_graph()` which returns the compiled
    graph object (LangGraph compiled graph). The mixin uses the attribute
    `graph` on the instance when present and will set it when building.
    """

    config: Any
    agent: Any

    def _build_graph(self) -> Any:
        """Concrete agents must implement this method to return a compiled graph."""
        raise NotImplementedError

    def build(self) -> Any:
        # If a compiled graph already exists on the instance, keep it.
        if getattr(self, "graph", None) is None:
            self.graph = self._build_graph()

        return self

    def get_graph(self) -> Any:
        g = getattr(self, "graph", None)
        if g is None:
            raise RuntimeError("Agent graph is not built. Call build() first.")
        return g

    def get_mermaid(self) -> str:
        g = self.get_graph()
        try:
            return str(g.get_graph().draw_mermaid())
        except Exception as e:  # pragma: no cover - wrapper around drawing failure
            raise RuntimeError(f"Failed to draw mermaid diagram: {e}") from e

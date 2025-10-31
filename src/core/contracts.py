"""Core contracts for agent composition and model creation.

This module centralizes lightweight "contracts" used across the project:

- PromptProtocol: minimal interface for prompt-like objects.
- AgentProtocol: lifecycle and execution capabilities expected from agents.
- ModelFactory: provider-agnostic factory for chat models (Ollama/OpenAI).
- AgentConfig: shared configuration for agents (hydrates a model lazily).
- AgentMixin: thread-safe build and standardized sync/async/stream wrappers.

These contracts avoid coupling the rest of the codebase directly to any
framework internals and provide a consistent surface area for testing.
"""
from __future__ import annotations

import inspect
import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Protocol, AsyncIterator, Optional, TypeVar, Generic, Callable 

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from config import get_settings


# ---------------------------
# Prompt / Agent Protocols
# ---------------------------

class PromptProtocol(Protocol):
    """Minimal contract for prompt-like objects.

    Attributes:
        id: Stable identifier for the prompt (useful for logging/versioning).
        version: Optional semantic version for the prompt template.
        prompt: The primary human-visible template used for formatting.

    Methods:
        format(**kwargs): Render the primary template with provided keyword arguments.
        messages(**kwargs): Render a sequence of LangChain-compatible messages.
    """

    id: str
    version: str

    @property
    def prompt(self) -> str: ...

    def format(self, **kwargs: Any) -> str: ...
    def messages(self, **kwargs: Any) -> list[Any]: ...


class AgentProtocol(Protocol):
    """Lifecycle and execution interface for agent implementations.

    Implementations typically wrap a compiled, invokable object (for example,
    a LangGraph compiled graph) and expose a unified execution surface.

    Attributes:
        agent: Compiled/invokable object (implementation-defined).
        config: Optional typed configuration for the concrete agent.

    Build lifecycle:
        _build_graph(): Construct and return the compiled agent object.
        build(force=False): Compile and store the agent (idempotent by default).
        ensure_built(): Return a compiled agent, building on first use.
        get_mermaid(): Return a Mermaid diagram if the underlying agent exposes one.

    Execution:
        invoke(state): Synchronous execution wrapper (not allowed inside an event loop).
        ainvoke(state): Asynchronous execution wrapper.
        astream(state): Asynchronous stream of events/results when supported.
    """

    # Compiled/invokable object (e.g., LangGraph compiled graph)
    agent: Any
    # Optional: concrete configs on subclasses
    config: "AgentConfig | None"

    # Build lifecycle
    def _build_graph(self) -> Any: ...
    def build(self, force: bool = False) -> "AgentProtocol": ...
    def ensure_built(self) -> Any: ...
    def get_mermaid(self) -> str: ...

    # Execution
    def invoke(self, state: Any) -> Any: ...
    async def ainvoke(self, state: Any) -> Any: ...
    async def astream(self, state: Any) -> AsyncIterator[Any]: ...


# ---------------------------
# Model Factory (Provider Abstraction)
# ---------------------------

class ModelFactory:
    """Provider-agnostic chat model factory.

    This factory centralizes provider-specific quirks for initializing
    chat models via ``langchain.chat_models.init_chat_model``. By selecting
    the provider based on ``base_url`` presence, the rest of the code can
    request a model by name without branching.
    """
    @staticmethod
    def create_chat_model(
        model_name: str,
        base_url: str | None = None,
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> BaseChatModel:
        """Create a chat model with a minimal, portable configuration.

        Args:
            model_name: Provider-specific model identifier (e.g., "llama3.1").
            base_url: If provided, selects the "ollama" provider; otherwise "openai".
            temperature: Optional sampling temperature.
            num_ctx: Optional context window size; only applied for Ollama.
            extra: Arbitrary provider kwargs (last-write-wins).

        Returns:
            A ``BaseChatModel`` instance ready for invoke/ainvoke.
        """

        provider = "ollama" if base_url else "openai"
        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        # Only Ollama supports num_ctx; don't pass it to OpenAI
        if provider == "ollama" and num_ctx:
            kwargs["num_ctx"] = num_ctx
        if extra:
            kwargs.update(extra)
        return init_chat_model(
            model=model_name,
            model_provider=provider,
            base_url=base_url,
            **kwargs,
        )


# ---------------------------
# Agent Config
# ---------------------------

@dataclass
class AgentConfig:
    """Base configuration for agents that use a chat model.

    Fields may be provided directly or defaulted from global settings via
    :func:`config.get_settings`. On initialization, a chat model is created
    and stored in ``model`` using :class:`ModelFactory` unless one is already
    supplied.

    Attributes:
        model_name: Name/ID of the chat model.
        base_url: Provider base URL (when set, Ollama is assumed; otherwise OpenAI).
        embeddings_model: Optional embeddings model identifier.
        num_ctx: Optional context window size (Ollama only).
        temperature: Optional temperature for the chat model.
        model: Actual model instance; set automatically unless provided.
    """

    model_name: str | None = None
    base_url: str | None = None
    embeddings_model: str | None = None
    num_ctx: int | None = None
    temperature: float | None = None
    model: BaseChatModel | None = None

    def __post_init__(self) -> None:
        """Hydrate configuration with defaults and create a chat model.

        This method pulls missing values from ``get_settings()`` and then
        constructs a chat model via :class:`ModelFactory`. If ``model_name``
        is still empty after merging, a ``ValueError`` is raised.
        """

        settings = get_settings()
        self.model_name = self.model_name or settings.model_name
        self.base_url = self.base_url if self.base_url is not None else settings.base_url
        self.embeddings_model = self.embeddings_model or settings.embeddings_model

        # Narrow to non-empty string for static type-checkers
        mn = self.model_name or ""
        if not mn:
            raise ValueError("model_name must be set via AgentConfig or settings")

        self.model = ModelFactory.create_chat_model(
            model_name=mn,
            base_url=self.base_url,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
        )


# ---------------------------
# Agent Mixin 
# ---------------------------
StateT = TypeVar("StateT")
ResultT = TypeVar("ResultT")

class AgentMixin(Generic[StateT, ResultT]):
    """Thread-safe build and standardized execution wrappers for agents.

    This mixin does not prescribe how agents are built; subclasses provide
    ``_build_graph`` to construct a compiled, invokable object. The mixin
    then offers:
    - A thread-safe ``build`` method to compile once per instance.
    - ``invoke``: a sync facade that runs outside event loops only.
    - ``ainvoke``: an async facade with best-effort fallback for sync agents.
    - ``astream``: an async generator that streams events when supported, or
      yields a single result as a fallback.

    It is intentionally framework-agnostic. If the compiled object provides
    a Mermaid diagram method, ``get_mermaid`` will return it.
    """
    config: Any

    def __init__(self, *args, **kwargs) -> None:
        self._agent: Any = None
        self._build_lock = threading.Lock()
        # Optional hook if you want to provide Mermaid without coupling:
        self._mermaid_getter: Optional[Callable[[Any], str]] = None

    @property
    def agent(self) -> Any:
        return self.ensure_built()

    @agent.setter
    def agent(self, value: Any) -> None:
        self._agent = value

    def _build_graph(self) -> Any:
        raise NotImplementedError

    def build(self, force: bool = False):
        """Compile (or recompile) and store the invokable agent object.

        Args:
            force: When True, recompile even if a compiled object exists.

        Returns:
            self for fluent chaining.
        """
        if self._agent is not None and not force:
            return self
        with self._build_lock:
            if self._agent is None or force:
                self._agent = self._build_graph()
        return self

    def ensure_built(self) -> Any:
        """Ensure the compiled agent exists; build on first use.

        Returns:
            The compiled invokable object (implementation-defined).
        """
        if self._agent is None:
            self.build()
        return self._agent

    def get_mermaid(self) -> str:
        """Return a Mermaid diagram if the compiled agent exposes it.

        Returns:
            A Mermaid diagram string.

        Raises:
            RuntimeError: If no diagram capability is found on the agent.
        """
        agent = self.ensure_built()

        # Preferred: user-supplied getter
        if self._mermaid_getter:
            return self._mermaid_getter(agent)

        # Best-effort autodetect (keeps this mixin framework-agnostic)
        if hasattr(agent, "get_graph"):
            g = agent.get_graph()
            if hasattr(g, "draw_mermaid"):
                return str(g.draw_mermaid())
        raise RuntimeError("Compiled agent does not expose a Mermaid diagram method.")

    # ---------------------------
    # Standardized execution
    # ---------------------------

    def invoke(self, state: StateT, /, **kwargs: Any) -> ResultT:
        """Synchronous wrapper; only call outside an active event loop.

        Args:
            state: Input state for the agent.
            **kwargs: Forwarded execution options for the compiled agent.

        Returns:
            The agent's result.

        Raises:
            RuntimeError: If called inside an active event loop.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(state, **kwargs))  # type: ignore[return-value]
        raise RuntimeError("invoke() called inside an event loop; use `await ainvoke(...)` instead.")

    async def ainvoke(self, state: StateT, /, **kwargs: Any) -> ResultT:
        """Asynchronous execution of the compiled agent.

        Args:
            state: Input state for the agent.
            **kwargs: Forwarded execution options for the compiled agent.

        Returns:
            The agent's result.

        Raises:
            RuntimeError: If the compiled agent exposes neither ``ainvoke``
                nor ``invoke`` methods.
        """
        agent = self.ensure_built()

        # Native async
        if hasattr(agent, "ainvoke") and inspect.iscoroutinefunction(agent.ainvoke):
            return await agent.ainvoke(state, **kwargs)  # type: ignore[attr-defined, return-value]

        # Sync fallback executed off the event loop
        if hasattr(agent, "invoke") and callable(agent.invoke):
            return await asyncio.to_thread(agent.invoke, state, **kwargs)  # type: ignore[return-value]

        raise RuntimeError("Compiled agent exposes neither `ainvoke` nor `invoke`.")

    async def astream(self, state: StateT, /, **kwargs: Any) -> AsyncIterator[Any]:
        """Asynchronous streaming of graph events/results.

        Yields:
            Events/results as produced by the compiled agent. If native
            streaming is not available, yields a single final result.
        """
        agent = self.ensure_built()

        if hasattr(agent, "astream"):
            astream_attr = getattr(agent, "astream")
            if inspect.isasyncgenfunction(astream_attr):
                async for ev in astream_attr(state, **kwargs):  # type: ignore[misc]
                    yield ev
                return
            if inspect.iscoroutinefunction(astream_attr):
                # Some libs return an async iterator from an async fn
                stream = await astream_attr(state, **kwargs)  # type: ignore[misc]
                async for ev in stream:
                    yield ev
                return

        # Fallback: no native streaming; emit a single result
        yield await self.ainvoke(state, **kwargs)

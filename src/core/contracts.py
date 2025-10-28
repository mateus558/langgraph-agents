# core/contracts.py
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
    """Protocol for prompt interface compatibility."""
    prompt: str
    def format(self, **kwargs: Any) -> str: ...


class AgentProtocol(Protocol):
    """Standardized lifecycle and execution for agents."""
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
    """Factory to create chat models while isolating provider-specific quirks."""
    @staticmethod
    def create_chat_model(
        model_name: str,
        base_url: str | None = None,
        *,
        temperature: float | None = None,
        num_ctx: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> BaseChatModel:
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
    model_name: str | None = None
    base_url: str | None = None
    embeddings_model: str | None = None
    num_ctx: int | None = None
    temperature: float | None = None
    model: BaseChatModel | None = None

    def __post_init__(self) -> None:
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
    """Thin mixin: thread-safe build + standardized sync/async/stream execution.
    - No business logic here.
    - Framework-agnostic (no hard dependency on LangGraph APIs).
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
        """Compile (or recompile) and store the invokable agent object."""
        if self._agent is not None and not force:
            return self
        with self._build_lock:
            if self._agent is None or force:
                self._agent = self._build_graph()
        return self

    def ensure_built(self) -> Any:
        """Ensure the compiled agent exists; build on first use."""
        if self._agent is None:
            self.build()
        return self._agent

    def get_mermaid(self) -> str:
        """Return a Mermaid diagram if the compiled agent exposes it."""
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
        """Synchronous wrapper; only call outside an active event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(state, **kwargs))  # type: ignore[return-value]
        raise RuntimeError("invoke() called inside an event loop; use `await ainvoke(...)` instead.")

    async def ainvoke(self, state: StateT, /, **kwargs: Any) -> ResultT:
        """Asynchronous execution of the compiled agent."""
        agent = self.ensure_built()

        # Native async
        if hasattr(agent, "ainvoke") and inspect.iscoroutinefunction(agent.ainvoke):
            return await agent.ainvoke(state, **kwargs)  # type: ignore[attr-defined, return-value]

        # Sync fallback executed off the event loop
        if hasattr(agent, "invoke") and callable(agent.invoke):
            return await asyncio.to_thread(agent.invoke, state, **kwargs)  # type: ignore[return-value]

        raise RuntimeError("Compiled agent exposes neither `ainvoke` nor `invoke`.")

    async def astream(self, state: StateT, /, **kwargs: Any) -> AsyncIterator[Any]:
        """Asynchronous streaming of graph events/results."""
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
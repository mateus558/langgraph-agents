# core/contracts.py
from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any, Protocol, AsyncIterator

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
# Agent Mixin (no .graph; uses only .agent)
# ---------------------------

class AgentMixin:
    """Thin mixin: thread-safe build + standardized sync/async/stream execution."""
    config: Any
    # Holds the compiled/invokable object (e.g., LangGraph compiled graph)
    agent: Any

    def __init__(self, *args, **kwargs) -> None:
        self.agent = None
        self._build_lock = threading.Lock()

    def _build_graph(self) -> Any:
        raise NotImplementedError

    def build(self, force: bool = False):
        """Compile (or recompile) and store the invokable agent object."""
        if self.agent is not None and not force:
            return self
        with self._build_lock:
            if self.agent is None or force:
                self.agent = self._build_graph()
        return self

    def ensure_built(self):
        """Ensure the compiled agent exists; build on first use."""
        if self.agent is None:
            self.build()
        return self.agent

    def get_mermaid(self) -> str:
        """Ask the compiled agent for a Mermaid diagram (if available)."""
        agent = self.ensure_built()
        # LangGraph compiled graphs expose .get_graph().draw_mermaid()
        try:
            return str(agent.get_graph().draw_mermaid())
        except AttributeError as e:
            raise RuntimeError("Compiled agent does not expose get_graph().draw_mermaid().") from e
        except Exception as e:
            raise RuntimeError(f"Failed to generate Mermaid diagram: {e}") from e

    # ---------------------------
    # Standardized execution
    # ---------------------------

    def invoke(self, state: Any) -> Any:
        """Synchronous wrapper; only safe outside an active event loop."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.ainvoke(state))
        raise RuntimeError("invoke() called inside an event loop. Use `await ainvoke(...)` instead.")

    async def ainvoke(self, state: Any) -> Any:
        """Asynchronous execution of the compiled agent."""
        agent = self.ensure_built()
        if hasattr(agent, "ainvoke"):
            return await agent.ainvoke(state)  # type: ignore[attr-defined]
        return await asyncio.to_thread(agent.invoke, state)

    async def astream(self, state: Any) -> AsyncIterator[Any]:
        """Asynchronous streaming of graph events/results."""
        agent = self.ensure_built()
        if hasattr(agent, "astream"):
            async for ev in agent.astream(state):  # type: ignore[attr-defined]
                yield ev
            return
        # Fallback: no native streaming; emit a single result
        yield await self.ainvoke(state)

# core/contracts.py
from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Protocol

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from config import get_settings


# ---------------------------
# Prompt / Agent Protocols
# ---------------------------

class PromptProtocol(Protocol):
    """Protocol for prompt interface compatibility."""
    prompt: str

    def format(self, **kwargs: Any) -> str:
        ...


class AgentProtocol(Protocol):
    """Protocol for agent interface compatibility."""
    config: "AgentConfig | None"
    agent: Any  # compiled/ready graph

    def invoke(self, state: Any) -> Any: ...
    def _build_graph(self) -> Any: ...
    def build(self) -> Any: ...
    def get_graph(self) -> Any: ...
    def get_mermaid(self) -> str: ...


# ---------------------------
# Model Factory (Provider Abstraction)
# ---------------------------

class ModelFactory:
    """Small factory to create chat models while isolating provider quirks."""

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

        # Only Ollama understands num_ctx; do not leak to OpenAI.
        if provider == "ollama" and num_ctx:
            kwargs["num_ctx"] = num_ctx

        if extra:
            kwargs.update(extra)

        return init_chat_model(
            model=model_name,
            model_provider=provider,
            base_url=base_url,
            **kwargs,  # NOTE: not "kwargs=kwargs"
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

        # --- narrow to non-optional for the factory call ---
        model_name: str = self.model_name  # type: ignore[assignment]
        if model_name is None:
            raise ValueError("model_name must be set in settings or AgentConfig")

        self.model = ModelFactory.create_chat_model(
            model_name=model_name,
            base_url=self.base_url,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
        )


# ---------------------------
# Agent Mixin (thread-safe, lazy/eager)
# ---------------------------

class AgentMixin:
    """Default build/get_graph/get_mermaid with thread-safe build."""
    config: Any
    agent: Any

    def __init__(self, *args, **kwargs) -> None:
        # Subclasses may override __init__; call super().__init__ if they do.
        self.agent = None
        self._build_lock = threading.Lock()

    def _build_graph(self) -> Any:
        raise NotImplementedError

    def build(self, force: bool = False):
        if self.agent is not None and not force:
            return self
        with self._build_lock:
            if self.agent is None or force:
                compiled = self._build_graph()
                self.agent = compiled
        return self

    def ensure_built(self):
        if self.agent is None:
            self.build()
        return self.agent

    def get_graph(self) -> Any:
        g = self.agent
        if g is None:
            raise RuntimeError("Agent graph is not built. Call build() first.")
        return g

    def get_mermaid(self) -> str:
        g = self.get_graph()
        try:
            return str(g.get_graph().draw_mermaid())
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Failed to draw mermaid diagram: {e}") from e

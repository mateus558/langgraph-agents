"""Global configuration for models and embeddings.

This module centralizes defaults for:
- model_name: chat/LLM model identifier
- base_url: base URL for the LLM provider (e.g., Ollama). If None, use default provider (e.g., OpenAI via SDK defaults).
- embeddings_model: embeddings model identifier

Values can be overridden via environment variables:
- MODEL_NAME
- LLM_BASE_URL (or BASE_URL)
- EMBEDDINGS_MODEL

Example:
    from config import get_settings

    settings = get_settings()
    print(settings.model_name, settings.base_url, settings.embeddings_model)

Note:
- We intentionally avoid importing dotenv here to keep this module dependency-free.
  If you use a .env file, load it elsewhere in your app startup.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(slots=True)
class Settings:
    model_name: str = "llama3.1"
    base_url: str | None = None
    embeddings_model: str = "nomic-embed-text"

    @classmethod
    def from_env(cls) -> "Settings":
        # Avoid referencing class attributes directly when slots=True; use an instance
        defaults = cls()
        model_name = os.getenv("MODEL_NAME", defaults.model_name)
        # Accept either LLM_BASE_URL or BASE_URL; empty/"none" -> None
        base_url_raw = os.getenv("LLM_BASE_URL", os.getenv("BASE_URL", defaults.base_url or ""))
        base_url = None if str(base_url_raw).strip().lower() in {"", "none", "null"} else base_url_raw
        embeddings_model = os.getenv("EMBEDDINGS_MODEL", defaults.embeddings_model)
        return cls(
            model_name=model_name,
            base_url=base_url,
            embeddings_model=embeddings_model,
        )


_GLOBAL_SETTINGS: Settings | None = None


def get_settings() -> Settings:
    """Return the cached global Settings, loading from env on first use."""
    global _GLOBAL_SETTINGS
    if _GLOBAL_SETTINGS is None:
        _GLOBAL_SETTINGS = Settings.from_env()
    return _GLOBAL_SETTINGS


def override_settings(**kwargs) -> Settings:
    """Override select fields at runtime (useful in tests or scripts).

    Example:
        override_settings(model_name="gpt-4o-mini", base_url=None)
    """
    settings = get_settings()
    for k, v in kwargs.items():
        if hasattr(settings, k):
            setattr(settings, k, v)
    return settings

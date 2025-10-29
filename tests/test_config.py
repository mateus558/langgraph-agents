"""Tests for core configuration."""

import os
import pytest
from src.config import Settings, get_settings, override_settings


def test_settings_defaults():
    """Test that Settings has sensible defaults."""
    settings = Settings()
    assert settings.model_name == "llama3.1"
    assert settings.base_url is None
    assert settings.embeddings_model == "nomic-embed-text"


def test_settings_from_env(monkeypatch):
    """Test loading settings from environment variables."""
    monkeypatch.setenv("MODEL_NAME", "test-model")
    monkeypatch.setenv("LLM_BASE_URL", "http://test:1234")
    monkeypatch.setenv("EMBEDDINGS_MODEL", "test-embeddings")
    
    settings = Settings.from_env()
    assert settings.model_name == "test-model"
    assert settings.base_url == "http://test:1234"
    assert settings.embeddings_model == "test-embeddings"


def test_settings_base_url_none_values(monkeypatch):
    """Test that 'none', 'null', empty string result in None for base_url."""
    test_cases = ["none", "null", "NONE", ""]
    
    for value in test_cases:
        monkeypatch.setenv("LLM_BASE_URL", value)
        settings = Settings.from_env()
        assert settings.base_url is None, f"Failed for value: {value}"


def test_override_settings():
    """Test overriding settings at runtime."""
    original = get_settings()
    original_model = original.model_name
    
    override_settings(model_name="new-model")
    updated = get_settings()
    
    assert updated.model_name == "new-model"
    # Cleanup: restore original
    override_settings(model_name=original_model)

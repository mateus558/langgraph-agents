"""Pytest configuration and fixtures."""

import pytest
import os


@pytest.fixture(autouse=True)
def reset_env_for_tests(monkeypatch):
    """Reset environment variables for each test."""
    # Clear any existing settings cache
    from src import config
    config._GLOBAL_SETTINGS = None


@pytest.fixture
def sample_messages():
    """Provide sample messages for testing."""
    from langchain.messages import HumanMessage, AIMessage, SystemMessage
    
    return [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello!"),
        AIMessage(content="Hi there! How can I help you?"),
        HumanMessage(content="What's the weather?"),
    ]


@pytest.fixture
def mock_model_config():
    """Provide mock model configuration."""
    return {
        "model_name": "test-model",
        "base_url": None,
        "temperature": 0.5,
        "num_ctx": 8192,
    }


@pytest.fixture
def sample_search_results():
    """Provide sample search results for testing."""
    return [
        {
            "title": "Example Result 1",
            "link": "https://example.com/1",
            "snippet": "This is the first example result.",
        },
        {
            "title": "Example Result 2",
            "link": "https://example.com/2",
            "snippet": "This is the second example result.",
        },
        {
            "title": "Example Result 3",
            "link": "https://example.com/3",
            "snippet": "This is the third example result.",
        },
    ]

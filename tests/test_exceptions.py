"""Tests for custom exceptions."""

import pytest
from src.core.exceptions import (
    AIServerError,
    ConfigurationError,
    ModelError,
    SearchError,
    SummarizationError,
    ValidationError,
    TokenLimitError,
    NetworkError,
    TimeoutError,
)


def test_base_exception():
    """Test base AIServerError."""
    error = AIServerError("Test error")
    assert str(error) == "Test error"
    assert isinstance(error, Exception)


def test_configuration_error():
    """Test ConfigurationError."""
    error = ConfigurationError("Invalid config")
    assert isinstance(error, AIServerError)
    assert str(error) == "Invalid config"


def test_model_error():
    """Test ModelError."""
    error = ModelError("Model failed")
    assert isinstance(error, AIServerError)


def test_search_error():
    """Test SearchError."""
    error = SearchError("Search failed")
    assert isinstance(error, AIServerError)


def test_summarization_error():
    """Test SummarizationError."""
    error = SummarizationError("Summary failed")
    assert isinstance(error, AIServerError)


def test_validation_error():
    """Test ValidationError."""
    error = ValidationError("Validation failed")
    assert isinstance(error, AIServerError)


def test_token_limit_error():
    """Test TokenLimitError."""
    error = TokenLimitError("Token limit exceeded")
    assert isinstance(error, AIServerError)


def test_network_error():
    """Test NetworkError."""
    error = NetworkError("Network error")
    assert isinstance(error, AIServerError)


def test_timeout_error():
    """Test TimeoutError."""
    error = TimeoutError("Operation timed out")
    assert isinstance(error, AIServerError)


def test_exception_hierarchy():
    """Test that all exceptions inherit from base."""
    exceptions = [
        ConfigurationError,
        ModelError,
        SearchError,
        SummarizationError,
        ValidationError,
        TokenLimitError,
        NetworkError,
        TimeoutError,
    ]
    
    for exc_class in exceptions:
        assert issubclass(exc_class, AIServerError)
        assert issubclass(exc_class, Exception)

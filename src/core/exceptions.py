"""Custom exception classes for the AI Server application."""

from __future__ import annotations


class AIServerError(Exception):
    """Base exception for all AI Server errors."""

    pass


class ConfigurationError(AIServerError):
    """Raised when there's a configuration issue."""

    pass


class ModelError(AIServerError):
    """Raised when there's an issue with model initialization or invocation."""

    pass


class SearchError(AIServerError):
    """Raised when there's an issue with search operations."""

    pass


class SummarizationError(AIServerError):
    """Raised when there's an issue during summarization."""

    pass


class ValidationError(AIServerError):
    """Raised when input validation fails."""

    pass


class TokenLimitError(AIServerError):
    """Raised when token limits are exceeded."""

    pass


class NetworkError(AIServerError):
    """Raised when there's a network-related error."""

    pass


class TimeoutError(AIServerError):
    """Raised when an operation times out."""

    pass

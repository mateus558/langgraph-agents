"""Pydantic models for input validation and configuration."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from enum import Enum


class SafeSearchLevel(str, Enum):
    """Safe search levels for web search."""

    OFF = "0"
    MODERATE = "1"
    STRICT = "2"


class ModelProvider(str, Enum):
    """Supported model providers."""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ChatRequest(BaseModel):
    """Request model for chat agent."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, max_length=10000, description="User query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    max_tokens: Optional[int] = Field(None, ge=1, le=100000, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Model temperature")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty after stripping."""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class WebSearchRequest(BaseModel):
    """Request model for web search."""

    model_config = ConfigDict(str_strip_whitespace=True)

    queries: list[str] = Field(..., min_length=1, max_length=10, description="Search queries")
    top_k: Optional[int] = Field(8, ge=1, le=50, description="Number of results to return")
    safesearch: Optional[SafeSearchLevel] = Field(SafeSearchLevel.MODERATE, description="Safe search level")
    language: Optional[str] = Field("en", min_length=2, max_length=10, description="Language code")

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, v: list[str]) -> list[str]:
        """Validate queries are not empty."""
        cleaned = [q.strip() for q in v if q and q.strip()]
        if not cleaned:
            raise ValueError("At least one non-empty query is required")
        return cleaned


class AgentConfigModel(BaseModel):
    """Configuration model for chat agent."""

    model_config = ConfigDict(validate_assignment=True)

    model_name: str = Field(..., min_length=1, description="Model identifier")
    base_url: Optional[str] = Field(None, description="Base URL for model provider")
    temperature: float = Field(0.5, ge=0.0, le=2.0, description="Model temperature")
    max_tokens_before_summary: int = Field(4000, ge=100, le=1000000, description="Token limit before summarization")
    messages_to_keep: int = Field(5, ge=1, le=100, description="Messages to keep after summarization")
    num_ctx: int = Field(131072, ge=1024, le=1000000, description="Context window size")

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate base_url is a valid URL if provided."""
        if v is not None and v.strip():
            v = v.strip()
            if not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError("base_url must start with http:// or https://")
        return v


class SearchConfigModel(BaseModel):
    """Configuration model for search agent."""

    model_config = ConfigDict(validate_assignment=True)

    searx_host: str = Field(..., description="SearxNG host URL")
    model_name: str = Field(..., min_length=1, description="Model identifier")
    base_url: Optional[str] = Field(None, description="Base URL for model provider")
    temperature: float = Field(0.5, ge=0.0, le=2.0, description="Model temperature")
    k: int = Field(8, ge=1, le=100, description="Number of results to return")
    max_categories: int = Field(3, ge=1, le=10, description="Maximum number of categories")
    safesearch: int = Field(1, ge=0, le=2, description="Safe search level")
    lang: Optional[str] = Field("en", description="Language code")
    retries: int = Field(2, ge=0, le=10, description="Number of retries")
    backoff_base: float = Field(1.0, ge=0.1, le=10.0, description="Backoff base time")
    num_ctx: int = Field(8192, ge=1024, le=1000000, description="Context window size")

    @field_validator("searx_host", "base_url")
    @classmethod
    def validate_url(cls, v: Optional[str]) -> Optional[str]:
        """Validate URL format."""
        if v is not None and v.strip():
            v = v.strip()
            if not (v.startswith("http://") or v.startswith("https://")):
                raise ValueError("URL must start with http:// or https://")
        return v


class SummarizationRequest(BaseModel):
    """Request model for summarization."""

    model_config = ConfigDict(str_strip_whitespace=True)

    messages: list[dict[str, Any]] = Field(..., min_length=1, description="Messages to summarize")
    existing_summary: Optional[str] = Field(None, description="Existing summary to extend")
    language: Optional[str] = Field("en", description="Output language")

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Validate messages list is not empty."""
        if not v:
            raise ValueError("Messages list cannot be empty")
        return v


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Health status")
    version: str = Field(..., description="Application version")
    timestamp: str = Field(..., description="Current timestamp")
    models: dict[str, Any] = Field(default_factory=dict, description="Available models")

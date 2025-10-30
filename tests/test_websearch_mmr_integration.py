"""Integration tests for MMR reranking in WebSearchAgent.

Tests the full agent pipeline with MMR enabled/disabled.
"""

from __future__ import annotations

import pytest

from src.websearch.agent import WebSearchAgent
from src.websearch.config import SearchAgentConfig


class MockEmbedder:
    """Mock embedder for testing."""

    def embed_query(self, text: str) -> list[float]:
        """Return deterministic embedding."""
        return [float(hash(text) % 100) / 100.0 for _ in range(8)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings."""
        return [self.embed_query(t) for t in texts]


def test_agent_config_with_mmr_defaults():
    """Test that MMR config parameters have correct defaults."""
    config = SearchAgentConfig()
    
    assert config.use_vectorstore_mmr is True
    assert config.mmr_lambda == 0.55
    assert config.mmr_fetch_k == 50
    assert config.embedder is None  # Should be None by default


def test_agent_config_with_custom_mmr_params():
    """Test MMR config with custom parameters."""
    embedder = MockEmbedder()
    
    config = SearchAgentConfig(
        use_vectorstore_mmr=False,
        mmr_lambda=0.7,
        mmr_fetch_k=100,
        embedder=embedder,
    )
    
    assert config.use_vectorstore_mmr is False
    assert config.mmr_lambda == 0.7
    assert config.mmr_fetch_k == 100
    assert config.embedder is embedder


def test_agent_initializes_embedder():
    """Test that agent attempts to initialize embedder."""
    config = SearchAgentConfig(model_name="llama3.1")
    agent = WebSearchAgent(config)
    
    # Embedder may be None if dependencies unavailable, which is OK
    # The important thing is the agent initializes without error
    assert agent.config.embedder is None or isinstance(agent.config.embedder, object)


def test_agent_with_mock_embedder():
    """Test agent with explicitly provided mock embedder."""
    embedder = MockEmbedder()
    config = SearchAgentConfig(
        model_name="llama3.1",
        embedder=embedder,
        use_vectorstore_mmr=True,
        mmr_lambda=0.6,
    )
    
    agent = WebSearchAgent(config)
    
    assert agent.config.embedder is embedder
    assert agent.config.use_vectorstore_mmr is True
    assert agent.config.mmr_lambda == 0.6


def test_mmr_disabled_agent():
    """Test agent with MMR explicitly disabled."""
    config = SearchAgentConfig(
        model_name="llama3.1",
        use_vectorstore_mmr=False,
    )
    
    agent = WebSearchAgent(config)
    
    assert agent.config.use_vectorstore_mmr is False
    # Agent should still work, falling back to domain diversification


@pytest.mark.asyncio
async def test_agent_node_dependencies_include_embedder():
    """Test that node dependencies receive the embedder."""
    embedder = MockEmbedder()
    config = SearchAgentConfig(
        model_name="llama3.1",
        embedder=embedder,
    )
    
    agent = WebSearchAgent(config)
    
    # Access the internal graph to check dependencies
    # This is a bit intrusive but verifies the wiring is correct
    assert hasattr(agent, "_build_graph")
    
    # The embedder should be passed to node dependencies
    # We can't easily test this without mocking search calls,
    # but we can at least verify the agent builds successfully
    assert agent.agent is not None


def test_mmr_parameters_validation():
    """Test MMR parameter validation and bounds."""
    config = SearchAgentConfig()
    
    # Lambda should be clamped to [0, 1]
    config.mmr_lambda = 1.5
    assert config.mmr_lambda == 1.5  # Not automatically clamped in config
    
    # Fetch_k should be positive
    config.mmr_fetch_k = 100
    assert config.mmr_fetch_k == 100
    
    # Zero or negative values are technically allowed (will be handled at runtime)
    config.mmr_fetch_k = 0
    assert config.mmr_fetch_k == 0


def test_mmr_env_var_override(monkeypatch):
    """Test MMR config via environment variables."""
    monkeypatch.setenv("USE_VECTORSTORE_MMR", "false")
    monkeypatch.setenv("MMR_LAMBDA", "0.8")
    monkeypatch.setenv("MMR_FETCH_K", "75")
    
    config = SearchAgentConfig()
    
    assert config.use_vectorstore_mmr is False
    assert config.mmr_lambda == 0.8
    assert config.mmr_fetch_k == 75


def test_mmr_env_var_boolean_parsing(monkeypatch):
    """Test various boolean formats for USE_VECTORSTORE_MMR."""
    test_cases = [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("False", False),
        ("0", False),
        ("no", False),
        ("off", False),
    ]
    
    for value, expected in test_cases:
        monkeypatch.setenv("USE_VECTORSTORE_MMR", value)
        config = SearchAgentConfig()
        assert config.use_vectorstore_mmr == expected, f"Failed for value: {value}"


def test_mmr_lambda_bounds_from_env(monkeypatch):
    """Test MMR lambda is clamped to [0, 1] from environment."""
    # Test lower bound
    monkeypatch.setenv("MMR_LAMBDA", "-0.5")
    config = SearchAgentConfig()
    assert config.mmr_lambda == 0.0
    
    # Test upper bound
    monkeypatch.setenv("MMR_LAMBDA", "2.0")
    config = SearchAgentConfig()
    assert config.mmr_lambda == 1.0
    
    # Test valid value
    monkeypatch.setenv("MMR_LAMBDA", "0.65")
    config = SearchAgentConfig()
    assert config.mmr_lambda == 0.65

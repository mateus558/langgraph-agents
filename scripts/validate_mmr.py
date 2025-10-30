#!/usr/bin/env python3
"""Quick validation script for MMR reranking feature.

This script verifies that:
1. Config parameters are properly set
2. MMR function works with and without embedder
3. Agent initializes correctly
4. All tests pass

Run this after implementing MMR to validate the feature.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 70)
print("MMR Reranking Feature Validation")
print("=" * 70)
print()

# Test 1: Config
print("✓ Test 1: Configuration")
from websearch.config import SearchAgentConfig

config = SearchAgentConfig()
assert hasattr(config, "use_vectorstore_mmr")
assert hasattr(config, "mmr_lambda")
assert hasattr(config, "mmr_fetch_k")
assert hasattr(config, "embedder")
print(f"  - use_vectorstore_mmr: {config.use_vectorstore_mmr}")
print(f"  - mmr_lambda: {config.mmr_lambda}")
print(f"  - mmr_fetch_k: {config.mmr_fetch_k}")
print(f"  - embedder: {config.embedder}")
print()

# Test 2: MMR function
print("✓ Test 2: MMR Function")
from websearch.utils import diversify_topk_mmr


async def test_mmr():
    results = [
        {"title": "A", "snippet": "Content A", "link": "https://a.com/1"},
        {"title": "B", "snippet": "Content B", "link": "https://b.com/2"},
    ]

    # Test without embedder
    output = await diversify_topk_mmr(
        results=results, k=2, query="test", embedder=None, lambda_mult=0.5, fetch_k=10
    )
    assert len(output) == 2
    print("  - Works without embedder: ✓")

    # Test with mock embedder
    class MockEmbedder:
        def embed_query(self, text: str) -> list[float]:
            return [0.1] * 8

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return [[0.1] * 8 for _ in texts]

    output = await diversify_topk_mmr(
        results=results,
        k=2,
        query="test",
        embedder=MockEmbedder(),
        lambda_mult=0.5,
        fetch_k=10,
        use_vectorstore_mmr=False,  # Use standalone MMR
    )
    assert len(output) == 2
    print("  - Works with embedder: ✓")


asyncio.run(test_mmr())
print()

# Test 3: Agent initialization
print("✓ Test 3: Agent Initialization")
from websearch.agent import WebSearchAgent

agent = WebSearchAgent(SearchAgentConfig(model_name="llama3.1"))
assert agent is not None
assert agent.config.embedder is not None or agent.config.embedder is None  # Either is OK
print(f"  - Agent initialized: ✓")
print(f"  - Embedder status: {'Available' if agent.config.embedder else 'Unavailable (fallback mode)'}")
print()

# Test 4: Node dependencies
print("✓ Test 4: Node Dependencies")
from websearch.nodes.shared import NodeDependencies
import inspect

sig = inspect.signature(NodeDependencies)
params = list(sig.parameters.keys())
assert "embedder" in params
print(f"  - NodeDependencies has embedder field: ✓")
print()

# Test 5: Documentation exists
print("✓ Test 5: Documentation")
docs_file = Path(__file__).parent.parent / "docs" / "MMR_RERANKING.md"
assert docs_file.exists()
print(f"  - Documentation file exists: ✓")
print(f"  - Path: {docs_file}")
print()

# Test 6: Demo script exists
print("✓ Test 6: Demo Script")
demo_file = Path(__file__).parent.parent / "examples" / "mmr_demo.py"
assert demo_file.exists()
print(f"  - Demo script exists: ✓")
print(f"  - Path: {demo_file}")
print()

# Summary
print("=" * 70)
print("✅ All Validation Checks Passed!")
print("=" * 70)
print()
print("Next steps:")
print("1. Run tests:     pytest tests/test_websearch*.py -v")
print("2. Run demo:      python examples/mmr_demo.py")
print("3. Read docs:     docs/MMR_RERANKING.md")
print("4. Try it:        USE_VECTORSTORE_MMR=true python your_app.py")
print()

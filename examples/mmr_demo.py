#!/usr/bin/env python3
"""Demo script showing MMR reranking in WebSearchAgent.

This script demonstrates the three-tier MMR strategy:
1. FAISS-based MMR (if langchain-community FAISS available)
2. Standalone MMR (if embedder available)
3. Domain-based diversification fallback

To run with different configurations:

# Default (will try to use embeddings if available)
python examples/mmr_demo.py

# Force disable MMR (use domain diversification only)
USE_VECTORSTORE_MMR=false python examples/mmr_demo.py

# Adjust MMR parameters
MMR_LAMBDA=0.7 MMR_FETCH_K=100 python examples/mmr_demo.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from websearch.utils import diversify_topk, diversify_topk_mmr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def demo_mmr_reranking():
    """Demonstrate MMR reranking with sample search results."""
    
    # Sample search results (simulating SearxNG output)
    results = [
        {
            "title": "Python Tutorial - Learn Python Programming",
            "snippet": "Comprehensive Python tutorial for beginners and experts. Learn Python syntax, data structures, and more.",
            "link": "https://python.org/tutorial",
        },
        {
            "title": "Python Documentation",
            "snippet": "Official Python documentation with detailed guides and API references for Python programming.",
            "link": "https://docs.python.org",
        },
        {
            "title": "Python Tutorial - W3Schools",
            "snippet": "Learn Python with W3Schools. Python is a popular programming language used for web development.",
            "link": "https://w3schools.com/python",
        },
        {
            "title": "Real Python - Python Tutorials",
            "snippet": "Real Python offers high-quality Python tutorials and courses for developers of all levels.",
            "link": "https://realpython.com",
        },
        {
            "title": "Python for Beginners",
            "snippet": "Start your Python journey with easy-to-follow tutorials and examples for beginners.",
            "link": "https://pythonforbeginners.com",
        },
        {
            "title": "Java Tutorial - Learn Java",
            "snippet": "Learn Java programming language from basics to advanced concepts.",
            "link": "https://javatutorial.com",
        },
        {
            "title": "JavaScript Guide",
            "snippet": "Complete JavaScript guide for web development and modern JS frameworks.",
            "link": "https://javascript.info",
        },
        {
            "title": "Python Data Science",
            "snippet": "Learn data science with Python using pandas, numpy, and scikit-learn libraries.",
            "link": "https://datasciencepython.com",
        },
    ]
    
    query = "Python programming tutorial"
    k = 5
    
    logger.info("=" * 70)
    logger.info("MMR Reranking Demo")
    logger.info("=" * 70)
    logger.info(f"Query: {query}")
    logger.info(f"Total results: {len(results)}")
    logger.info(f"Target k: {k}")
    logger.info("")
    
    # Test 1: Domain-based diversification (baseline)
    logger.info("-" * 70)
    logger.info("Test 1: Domain-based diversification (no embedder)")
    logger.info("-" * 70)
    
    baseline = diversify_topk(results, k=k)
    for i, r in enumerate(baseline, 1):
        logger.info(f"{i}. [{r['link']}] {r['title'][:50]}")
    logger.info("")
    
    # Test 2: MMR with mock embedder
    logger.info("-" * 70)
    logger.info("Test 2: MMR reranking (with embedder, if available)")
    logger.info("-" * 70)
    
    # Try to create embedder
    embedder = None
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("Loading HuggingFaceEmbeddings (this may take a moment)...")
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("✓ Embedder loaded successfully")
    except ImportError:
        logger.info("✗ langchain-huggingface not available, will use fallback")
    except Exception as exc:
        logger.info(f"✗ Failed to load embedder: {exc}, will use fallback")
    
    mmr_results = await diversify_topk_mmr(
        results=results,
        k=k,
        query=query,
        embedder=embedder,
        lambda_mult=0.55,  # Balance relevance and diversity
        fetch_k=len(results),
        use_vectorstore_mmr=True,
    )
    
    for i, r in enumerate(mmr_results, 1):
        logger.info(f"{i}. [{r['link']}] {r['title'][:50]}")
    logger.info("")
    
    # Test 3: MMR with high relevance (lambda=0.9)
    if embedder:
        logger.info("-" * 70)
        logger.info("Test 3: MMR with high relevance (lambda=0.9)")
        logger.info("-" * 70)
        
        relevant_results = await diversify_topk_mmr(
            results=results,
            k=k,
            query=query,
            embedder=embedder,
            lambda_mult=0.9,  # Favor relevance
            fetch_k=len(results),
            use_vectorstore_mmr=True,
        )
        
        for i, r in enumerate(relevant_results, 1):
            logger.info(f"{i}. [{r['link']}] {r['title'][:50]}")
        logger.info("")
    
    # Test 4: MMR with high diversity (lambda=0.1)
    if embedder:
        logger.info("-" * 70)
        logger.info("Test 4: MMR with high diversity (lambda=0.1)")
        logger.info("-" * 70)
        
        diverse_results = await diversify_topk_mmr(
            results=results,
            k=k,
            query=query,
            embedder=embedder,
            lambda_mult=0.1,  # Favor diversity
            fetch_k=len(results),
            use_vectorstore_mmr=True,
        )
        
        for i, r in enumerate(diverse_results, 1):
            logger.info(f"{i}. [{r['link']}] {r['title'][:50]}")
        logger.info("")
    
    logger.info("=" * 70)
    logger.info("Demo complete!")
    logger.info("=" * 70)
    logger.info("")
    logger.info("Key observations:")
    logger.info("1. Domain diversification ensures no single domain dominates")
    logger.info("2. MMR reranking balances relevance and diversity")
    logger.info("3. Lambda parameter controls relevance/diversity tradeoff:")
    logger.info("   - lambda=1.0: Max relevance (similar results)")
    logger.info("   - lambda=0.0: Max diversity (different topics)")
    logger.info("   - lambda=0.5-0.6: Balanced (recommended)")


if __name__ == "__main__":
    asyncio.run(demo_mmr_reranking())

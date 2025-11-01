#!/usr/bin/env python3
"""Example comparing different embedding models for MMR reranking.

This script demonstrates how to:
1. Auto-detect GPU availability
2. Use different embedding models
3. Compare performance and quality

Run with:
    python examples/embeddings_comparison.py

Or with specific model:
    EMBEDDINGS_MODEL_NAME=BAAI/bge-base-en-v1.5 python examples/embeddings_comparison.py
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def check_gpu_availability():
    """Check if GPU is available."""
    try:
        import torch # type: ignore
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            logger.info("✓ GPU Available:")
            logger.info(f"  - Device: {device_name}")
            logger.info(f"  - CUDA: {cuda_version}")
            logger.info(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            return "cuda"
        else:
            logger.info("✗ GPU Not Available - Using CPU")
            return "cpu"
    except ImportError:
        logger.info("✗ PyTorch not installed - Using CPU")
        return "cpu"


async def test_embedding_model(model_name: str, device: str):
    """Test a specific embedding model."""
    from websearch.ranking import diversify_topk_mmr
    
    logger.info("=" * 70)
    logger.info(f"Testing Model: {model_name}")
    logger.info(f"Device: {device}")
    logger.info("=" * 70)
    
    # Sample search results
    results = [
        {
            "title": "Python Tutorial - Learn Python Programming",
            "snippet": "Comprehensive Python tutorial for beginners. Learn syntax, data structures, OOP.",
            "link": "https://python.org/tutorial",
        },
        {
            "title": "Python Documentation",
            "snippet": "Official Python docs with API references and guides for Python development.",
            "link": "https://docs.python.org",
        },
        {
            "title": "Real Python Tutorials",
            "snippet": "High-quality Python tutorials covering web dev, data science, and more.",
            "link": "https://realpython.com",
        },
        {
            "title": "Python for Data Science",
            "snippet": "Learn Python for data analysis with pandas, numpy, matplotlib libraries.",
            "link": "https://datasciencepython.com",
        },
        {
            "title": "Django Web Framework",
            "snippet": "Build web applications with Django, the high-level Python web framework.",
            "link": "https://djangoproject.com",
        },
        {
            "title": "Flask Micro Framework",
            "snippet": "Lightweight Python web framework for building simple and complex web apps.",
            "link": "https://flask.palletsprojects.com",
        },
        {
            "title": "Python Machine Learning",
            "snippet": "Introduction to machine learning with scikit-learn and Python.",
            "link": "https://ml-python.com",
        },
        {
            "title": "JavaScript Tutorial",
            "snippet": "Learn JavaScript for web development and modern frameworks like React.",
            "link": "https://javascript.info",
        },
    ]
    
    query = "Python programming tutorial for beginners"
    
    try:
        # Initialize embedder
        logger.info("Initializing embedder...")
        start_init = time.perf_counter()
        
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embedder = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        init_time = time.perf_counter() - start_init
        logger.info(f"✓ Embedder initialized in {init_time:.2f}s")
        
        # Test embedding generation (warmup)
        logger.info("Warming up model...")
        _ = embedder.embed_query("test query")
        
        # Benchmark MMR reranking
        logger.info("Running MMR reranking...")
        start_mmr = time.perf_counter()
        
        reranked = await diversify_topk_mmr(
            results=results,
            k=5,
            query=query,
            embedder=embedder,
            lambda_mult=0.55,
            fetch_k=len(results),
            use_vectorstore_mmr=False,  # Use standalone MMR for fair comparison
        )
        
        mmr_time = time.perf_counter() - start_mmr
        
        logger.info(f"✓ MMR completed in {mmr_time:.3f}s")
        logger.info("")
        logger.info("Top 5 Results:")
        for i, r in enumerate(reranked, 1):
            logger.info(f"  {i}. {r['title'][:60]}")
        
        logger.info("")
        logger.info(f"Summary:")
        logger.info(f"  - Model: {model_name}")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Init time: {init_time:.2f}s")
        logger.info(f"  - MMR time: {mmr_time:.3f}s")
        logger.info(f"  - Throughput: {len(results)/mmr_time:.1f} docs/sec")
        logger.info("")
        
        return {
            "model": model_name,
            "device": device,
            "init_time": init_time,
            "mmr_time": mmr_time,
            "success": True,
        }
        
    except Exception as exc:
        logger.error(f"✗ Failed to test model: {exc}")
        return {
            "model": model_name,
            "device": device,
            "error": str(exc),
            "success": False,
        }


async def main():
    """Main demo function."""
    logger.info("=" * 70)
    logger.info("Embedding Models Comparison Demo")
    logger.info("=" * 70)
    logger.info("")
    
    # Check GPU availability
    device = check_gpu_availability()
    logger.info("")
    
    # Check if specific model requested
    requested_model = os.getenv("EMBEDDINGS_MODEL_NAME")
    
    if requested_model:
        logger.info(f"Testing requested model: {requested_model}")
        logger.info("")
        await test_embedding_model(requested_model, device)
    else:
        # Test multiple models
        models = [
            "sentence-transformers/all-MiniLM-L6-v2",  # Fastest
            "intfloat/e5-small-v2",  # Default, balanced
            "BAAI/bge-small-en-v1.5",  # Good quality small model
        ]
        
        # Add base models if GPU available
        if device == "cuda":
            models.extend([
                "intfloat/e5-base-v2",  # Higher quality
                "BAAI/bge-base-en-v1.5",  # Best quality base model
            ])
        
        logger.info(f"Testing {len(models)} models...")
        logger.info("")
        
        results = []
        for model in models:
            result = await test_embedding_model(model, device)
            results.append(result)
            await asyncio.sleep(1)  # Brief pause between models
        
        # Summary comparison
        logger.info("=" * 70)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 70)
        logger.info("")
        
        successful = [r for r in results if r["success"]]
        if successful:
            # Sort by MMR time
            successful.sort(key=lambda x: x["mmr_time"])
            
            logger.info(f"{'Model':<45} {'Device':<8} {'Init':<8} {'MMR':<10}")
            logger.info("-" * 70)
            for r in successful:
                model_short = r["model"].split("/")[-1][:43]
                logger.info(
                    f"{model_short:<45} {r['device']:<8} "
                    f"{r['init_time']:>6.2f}s  {r['mmr_time']:>7.3f}s"
                )
            
            logger.info("")
            logger.info(f"Fastest: {successful[0]['model']}")
            logger.info(f"  - MMR time: {successful[0]['mmr_time']:.3f}s")
            
        failed = [r for r in results if not r["success"]]
        if failed:
            logger.info("")
            logger.info("Failed models:")
            for r in failed:
                logger.info(f"  - {r['model']}: {r.get('error', 'Unknown error')}")
        
        logger.info("")
        logger.info("=" * 70)
        logger.info("Recommendations:")
        logger.info("=" * 70)
        if device == "cpu":
            logger.info("CPU Mode Detected:")
            logger.info("  - For speed: sentence-transformers/all-MiniLM-L6-v2")
            logger.info("  - For balance: intfloat/e5-small-v2 (default)")
            logger.info("  - For quality: BAAI/bge-small-en-v1.5")
        else:
            logger.info("GPU Mode Detected:")
            logger.info("  - For speed: intfloat/e5-small-v2")
            logger.info("  - For balance: intfloat/e5-base-v2")
            logger.info("  - For quality: BAAI/bge-base-en-v1.5")
            logger.info("")
            logger.info("To use GPU, set:")
            logger.info("  export EMBEDDINGS_MODEL_NAME=BAAI/bge-base-en-v1.5")


if __name__ == "__main__":
    asyncio.run(main())

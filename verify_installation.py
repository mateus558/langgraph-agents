#!/usr/bin/env python3
"""Verify installation and configuration."""

import sys
import os
from pathlib import Path


def check_env_file():
    """Check if .env file exists."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")
    
    print("📁 Checking environment configuration...")
    
    if not env_example_path.exists():
        print("  ❌ .env.example not found")
        return False
    else:
        print("  ✅ .env.example exists")
    
    if not env_path.exists():
        print("  ⚠️  .env file not found (copy from .env.example)")
        return False
    else:
        print("  ✅ .env file exists")
    
    return True


def check_imports():
    """Check if required packages can be imported."""
    print("\n📦 Checking package imports...")
    
    packages = [
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("pydantic", "pydantic"),
        ("requests", "requests"),
        ("dotenv", "python-dotenv"),
    ]
    
    all_ok = True
    for module, package in packages:
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} (install with: pip install {package})")
            all_ok = False
    
    return all_ok


def check_config():
    """Check if configuration loads correctly."""
    print("\n⚙️  Checking configuration...")
    
    try:
        from src.config import get_settings
        settings = get_settings()
        
        print(f"  ✅ Configuration loaded")
        print(f"     Model: {settings.model_name}")
        print(f"     Base URL: {settings.base_url or 'default'}")
        print(f"     Embeddings: {settings.embeddings_model}")
        
        return True
    except Exception as e:
        print(f"  ❌ Configuration error: {e}")
        return False


def check_project_structure():
    """Check if project structure is correct."""
    print("\n📂 Checking project structure...")
    
    required_files = [
        "src/__init__.py",
        "src/config.py",
        "src/core/contracts.py",
        "src/chatagent/agent.py",
        "src/websearch/agent.py",
        "src/utils/messages.py",
        "pyproject.toml",
        "Makefile",
        "README.md",
    ]
    
    all_ok = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            all_ok = False
    
    return all_ok


def check_agents():
    """Check if agents can be instantiated."""
    print("\n🤖 Checking agents...")
    
    try:
        from src.chatagent.agent import ChatAgent, ChatAgentConfig as ChatConfig
        config = ChatConfig(model_name="test-model", base_url=None)
        print("  ✅ ChatAgent can be imported and configured")
    except Exception as e:
        print(f"  ❌ ChatAgent error: {e}")
        return False
    
    try:
        from src.websearch.agent import WebSearchAgent, SearchAgentConfig
        config = SearchAgentConfig(
            searx_host="http://localhost:8095",
            model_name="test-model",
            base_url=None
        )
        print("  ✅ WebSearchAgent can be imported and configured")
    except Exception as e:
        print(f"  ❌ WebSearchAgent error: {e}")
        return False
    
    return True


def check_tests():
    """Check if tests can be discovered."""
    print("\n🧪 Checking tests...")
    
    test_files = list(Path("tests").glob("test_*.py"))
    
    if not test_files:
        print("  ⚠️  No test files found")
        return False
    
    print(f"  ✅ Found {len(test_files)} test files")
    for test_file in test_files:
        print(f"     - {test_file.name}")
    
    return True


def check_dev_tools():
    """Check if development tools are configured."""
    print("\n🛠️  Checking development tools...")
    
    files = {
        ".editorconfig": "Editor configuration",
        ".pre-commit-config.yaml": "Pre-commit hooks",
        "pyproject.toml": "Project configuration (includes tool configs)",
    }
    
    all_ok = True
    for file_path, description in files.items():
        path = Path(file_path)
        if path.exists():
            print(f"  ✅ {description}")
        else:
            print(f"  ❌ {description}")
            all_ok = False
    
    return all_ok


def main():
    """Run all checks."""
    print("=" * 60)
    print("🔍 AI Server Installation Verification")
    print("=" * 60)
    
    checks = [
        ("Environment", check_env_file),
        ("Imports", check_imports),
        ("Configuration", check_config),
        ("Project Structure", check_project_structure),
        ("Agents", check_agents),
        ("Tests", check_tests),
        ("Dev Tools", check_dev_tools),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"\n❌ {name} check failed with exception: {e}")
            results[name] = False
    
    print("\n" + "=" * 60)
    print("📊 Summary")
    print("=" * 60)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} {name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ All checks passed! Your installation is ready.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please address the issues above.")
        print("\nCommon fixes:")
        print("  1. Install dependencies: make install-dev")
        print("  2. Copy .env.example to .env")
        print("  3. Configure your .env file with proper values")
        return 1


if __name__ == "__main__":
    sys.exit(main())

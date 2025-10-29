"""Tests for message utilities."""

import pytest
from src.utils.messages import TokenEstimator, coerce_message_content
from langchain.messages import HumanMessage, AIMessage


class TestTokenEstimator:
    """Tests for TokenEstimator class."""
    
    def test_count_text_simple(self):
        """Test counting tokens in simple text."""
        estimator = TokenEstimator()
        text = "Hello, world!"
        count = estimator.count_text(text)
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_text_empty(self):
        """Test counting tokens in empty text."""
        estimator = TokenEstimator()
        assert estimator.count_text("") == 0
        assert estimator.count_text("   ") == 0
    
    def test_count_messages(self):
        """Test counting tokens in message list."""
        estimator = TokenEstimator()
        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        count = estimator.count_messages(messages)
        assert count > 0
        assert isinstance(count, int)
    
    def test_count_messages_empty(self):
        """Test counting tokens in empty message list."""
        estimator = TokenEstimator()
        assert estimator.count_messages([]) == 0


class TestCoerceMessageContent:
    """Tests for coerce_message_content function."""
    
    def test_string_content(self):
        """Test message with string content."""
        msg = HumanMessage(content="Hello")
        result = coerce_message_content(msg)
        assert result == "Hello"
    
    def test_list_content(self):
        """Test message with list content."""
        msg = HumanMessage(content=["Part 1", "Part 2"])
        result = coerce_message_content(msg)
        assert "Part 1" in result
        assert "Part 2" in result
    
    def test_dict_content(self):
        """Test message with complex content."""
        msg = HumanMessage(content=[{"text": "Hello", "type": "text"}])
        result = coerce_message_content(msg)
        assert isinstance(result, str)
        assert len(result) > 0

from __future__ import annotations

import pytest

from core.prompts import ChatPrompt, PromptRenderError
from chatagent.prompts import build_summarizer_prompt
from websearch.prompts import build_websearch_summary_messages


def test_chat_prompt_messages_include_system_and_human():
    prompt = ChatPrompt(
        id="test.prompt",
        version="0",
        system_template="sys {foo}",
        user_template="human {bar}",
        defaults={"foo": "FOO"},
    )

    human_text = prompt.format(bar="BAR")
    messages = prompt.messages(bar="BAR")

    assert human_text == "human BAR"
    assert len(messages) == 2
    assert messages[0].content == "sys FOO"
    assert messages[1].content == "human BAR"


def test_chat_prompt_missing_values_raise():
    prompt = ChatPrompt(id="bad.prompt", user_template="hello {name}")

    with pytest.raises(PromptRenderError):
        prompt.format()


def test_build_summarizer_prompt_returns_messages():
    human_prompt, messages = build_summarizer_prompt("prev", "A\nB")
    assert "This is a summary to date" in human_prompt
    assert any(msg.content.startswith("You are a conversation summarizer") for msg in messages[:-1])
    assert messages[-1].content.startswith("This is a summary to date")


def test_websearch_summary_messages_inject_times():
    messages = build_websearch_summary_messages(
        query="What is next",
        whitelist="ALLOWED SOURCES:\n[1] https://example.com",
        results="[1] Title\nURL: https://example.com\nSnippet: foo",
        utc_time="2024-01-01 00:00:00 UTC",
        local_time="2024-01-01 00:00:00 BRT",
        local_label="America/Sao_Paulo",
    )

    system_content = messages[0].content
    assert "Current time (UTC): 2024-01-01 00:00:00 UTC" in system_content
    assert "Convert times to America/Sao_Paulo" in system_content
    assert messages[-1].content.startswith("ALLOWED SOURCES")

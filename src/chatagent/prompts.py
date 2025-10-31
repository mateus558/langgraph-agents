"""Prompt definitions specific to the chat agent domain."""

from __future__ import annotations

from typing import Sequence, Tuple

from langchain_core.messages import BaseMessage

from core.prompts import ChatPrompt


# ---------------------------
# Chat agent system prompt
# ---------------------------

# The chat agent composes a System message plus the live conversation messages
# from state. We use ChatPrompt only to render the System content.
ASSISTANT_SYSTEM_PROMPT = ChatPrompt(
    id="chatagent.system",
    version="1.0.0",
    system_template=(
        "{clock}\n"
        "You are a helpful AI assistant. Be friendly and professional. "
        "Avoid long responses unless needed and reply in the user's language.\n"
        "Conversation Summary: {summary}\n"
    ),
    # We don't use the human template for this prompt in ChatAgent
    user_template="",
)

__all__ = ["ASSISTANT_SYSTEM_PROMPT"] 

SUMMARIZER_PROMPT = ChatPrompt(
    id="chatagent.summarize",
    version="1.0.0",
    system_template=(
        "You are a conversation summarizer. Maintain fidelity to the original "
        "messages, be concise, and respond in the user's language."
    ),
    user_template=(
        "{opening}"
        "\n### Rules:\n"
        "\t- Provide a concise summary in the same language as the user.\n"
        "\t- Return only the summary, nothing else.\n"
    ),
)


def build_summarizer_prompt(summary: str, messages_text: str) -> Tuple[str, Sequence[BaseMessage]]:
    """Return the rendered human prompt and message list."""
    if summary:
        opening = (
            "This is a summary to date:\n"
            f"{summary}\n\n"
            "Extend the summary considering the new messages:\n"
            f"{messages_text}\n"
        )
    else:
        opening = (
            "Create a concise summary of the conversation below:\n"
            f"{messages_text}\n"
        )

    human_prompt = SUMMARIZER_PROMPT.format(opening=opening)
    messages = SUMMARIZER_PROMPT.messages(opening=opening)
    return human_prompt, messages


__all__.extend(["SUMMARIZER_PROMPT", "build_summarizer_prompt"])

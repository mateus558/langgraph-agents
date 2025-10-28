# summarizers.py
import time
import logging
from typing import Sequence
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, RemoveMessage

from chatagent.config import Summarizer

logger = logging.getLogger(__name__)


class OllamaSummarizer(Summarizer):
    def __init__(
        self,
        model_id: str = "llama3.1",
        base_url: str | None = None,
        k_tail: int = 5,
        max_tokens_before_summary: int = 500,
        temperature: float = 0.0,
        num_ctx: int = 131072,
    ):
        self.k_tail = k_tail
        self.max_tokens_before_summary = max_tokens_before_summary

        model_provider = "ollama" if base_url else "openai"
        kwargs = {"num_ctx": num_ctx} if num_ctx else {}

        self.model = init_chat_model(
            model_id,
            model_provider=model_provider,
            temperature=temperature,
            base_url=base_url,
            kwargs=kwargs,
        )

    def _is_user_visible(self, m: AnyMessage) -> bool:
        if isinstance(m, RemoveMessage):
            return False
        msg_type = getattr(m, "type", "").lower()
        if msg_type not in {"human", "ai", "system", "tool"}:
            return False
        content = getattr(m, "content", None)
        return not (isinstance(content, str) and content.strip() == "__remove_all__")

    def _render(self, m: AnyMessage) -> str:
        role = getattr(m, "type", m.__class__.__name__).upper()
        content = m.content if isinstance(m.content, str) else str(m.content)
        return f"{role}: {content}"

    def summarize(self, state):
        summary = state.get("summary") or ""
        msgs: Sequence[AnyMessage] = state.get("messages", [])
        visible = [m for m in msgs if self._is_user_visible(m)]
        messages_text = "\n".join(self._render(m) for m in visible)

        if summary:
            prompt = (
                f"This is a summary to date:\n{summary}\n\n"
                f"Extend the summary considering the new messages:\n{messages_text}\n"
            )
        else:
            prompt = f"Create a concise summary of the conversation below:\n{messages_text}\n"

        prompt += (
            "\n### Rules:\n"
            "\t- Provide a concise summary in the same language as the user."
            "\n\t- Return only the summary, nothing else.\n"
        )

        logger.debug("Summarization prompt: %s", prompt)
        t0 = time.perf_counter()
        resp = self.model.invoke(prompt)
        dt = time.perf_counter() - t0
        logger.info("Summarization invoke took %.3fs", dt)
        
        new_summary = resp.content if isinstance(resp.content, str) else str(resp.content)
        logger.debug("New summary: %s", new_summary)

        tail = list(msgs[-self.k_tail:]) if self.k_tail > 0 else []
        from langgraph.graph.message import REMOVE_ALL_MESSAGES
        return {"summary": new_summary, "messages": [REMOVE_ALL_MESSAGES, *tail]}

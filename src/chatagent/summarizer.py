# summarizers.py
import logging
import time
from collections.abc import Sequence

from langchain.chat_models import init_chat_model
from langchain.messages import RemoveMessage
from langchain_core.messages import BaseMessage

from chatagent.config import Summarizer
from chatagent.prompts import build_summarizer_prompt

logger = logging.getLogger(__name__)


class BaseSummarizer(Summarizer):
    """Base summarizer with shared behavior.

    This centralizes common functionality so new implementations (e.g. OpenAI)
    can be added with minimal duplication.
    """

    def __init__(
        self,
        model_id: str = "llama3.1",
        base_url: str | None = None,
        k_tail: int = 5,
        max_tokens_before_summary: int = 500,
        temperature: float | None = None,
        num_ctx: int | None = None,
    ):
        self.k_tail = k_tail
        self.max_tokens_before_summary = max_tokens_before_summary

        provider = "ollama" if base_url else "openai"
        kwargs = {"num_ctx": num_ctx} if num_ctx is not None else {}

        # Initialize a chat model instance for summarization.
        # Subclasses may override how the model is constructed if needed.
        init_params: dict = {
            "model": model_id,
            "model_provider": provider,
        }
        if temperature is not None:
            init_params["temperature"] = temperature
        if base_url is not None:
            init_params["base_url"] = base_url
        if kwargs:
            init_params["kwargs"] = kwargs

        self.model = init_chat_model(**init_params)

    def _is_user_visible(self, m: BaseMessage) -> bool:
        if isinstance(m, RemoveMessage):
            return False
        msg_type = getattr(m, "type", "").lower()
        if msg_type not in {"human", "ai", "system", "tool"}:
            return False
        content = getattr(m, "content", None)
        return not (isinstance(content, str) and content.strip() == "__remove_all__")

    def _render(self, m: BaseMessage) -> str:
        role = getattr(m, "type", m.__class__.__name__).upper()
        content = m.content if isinstance(m.content, str) else str(m.content)
        return f"{role}: {content}"

    def build_prompt(self, summary: str, messages_text: str) -> tuple[str, Sequence[BaseMessage]]:
        """Return the rendered human prompt and message sequence.

        Subclasses can extend or override this to tweak prompt construction
        while still returning a `(prompt_text, messages)` tuple.
        """
        return build_summarizer_prompt(summary, messages_text)

    def summarize(self, state):
        summary = state.get("summary") or ""
        msgs: Sequence[BaseMessage] = state.get("messages", [])
        visible = [m for m in msgs if self._is_user_visible(m)]
        messages_text = "\n".join(self._render(m) for m in visible)

        prompt, messages_payload = self.build_prompt(summary, messages_text)

        logger.debug("Summarization prompt: %s", prompt)
        t0 = time.perf_counter()
        resp = self.model.invoke(messages_payload)
        dt = time.perf_counter() - t0
        logger.info("Summarization invoke took %.3fs", dt)

        content = getattr(resp, "content", None)
        new_summary = content if isinstance(content, str) else str(content)
        logger.debug("New summary: %s", new_summary)

        tail = list(msgs[-self.k_tail:]) if self.k_tail > 0 else []
        from langgraph.graph.message import REMOVE_ALL_MESSAGES
        return {"summary": new_summary, "messages": [REMOVE_ALL_MESSAGES, *tail]}


class OllamaSummarizer(BaseSummarizer):
    """Backward-compatible name for the default summarizer implementation.

    Currently this is a thin alias on BaseSummarizer that keeps the same
    defaults (model id, big context window) used historically.
    """

    def __init__(
        self,
        model_id: str = "llama3.1",
        base_url: str | None = None,
        k_tail: int = 5,
        max_tokens_before_summary: int = 500,
        temperature: float | None = None,
        num_ctx: int | None = None,
    ):
        super().__init__(
            model_id=model_id,
            base_url=base_url,
            k_tail=k_tail,
            max_tokens_before_summary=max_tokens_before_summary,
            temperature=temperature,
            num_ctx=num_ctx,
        )


class OpenAISummarizer(BaseSummarizer):
    """Summarizer configured to use OpenAI models by default.

    Default model is `gpt-5-nano`. This class exists to provide a clear
    provider-specific entrypoint and to allow future OpenAI-specific
    configuration (API keys, specialized prompt tweaks) without changing
    the base summarizer behavior.
    """

    def __init__(
        self,
        model_id: str = "gpt-5-nano",
        base_url: str | None = None,
        k_tail: int = 5,
        max_tokens_before_summary: int = 500,
        temperature: float | None = None,
        num_ctx: int | None = None,
    ):
        # Force base_url to None to ensure provider selection prefers OpenAI
        # in BaseSummarizer (which chooses provider by base_url presence).
        super().__init__(
            model_id=model_id,
            base_url=base_url,
            k_tail=k_tail,
            max_tokens_before_summary=max_tokens_before_summary,
            temperature=temperature,
            num_ctx=num_ctx,
        )

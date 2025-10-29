# summarizers.py
import time
from typing import Sequence
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, RemoveMessage

from chatagent.config import Summarizer

class OllamaSummarizer(Summarizer):
    def __init__(self, base_url="http://192.168.0.5:11434", model_id="granite4:tiny-h", k_tail=5, max_tokens_before_summary=500):
        self.k_tail = k_tail
        self.max_tokens_before_summary = max_tokens_before_summary

        self.model = init_chat_model(
            model_id,
            model_provider="ollama",
            temperature=0,
            base_url=base_url,
            kwargs={"num_ctx": 131072},
        )

    def _is_user_visible(self, m: AnyMessage) -> bool:
        if isinstance(m, RemoveMessage): return False
        t = getattr(m, "type", "").lower()
        if t not in {"human", "ai", "system", "tool"}: return False
        c = getattr(m, "content", None)
        return not (isinstance(c, str) and c.strip() == "__remove_all__")

    def _render(self, m: AnyMessage) -> str:
        role = getattr(m, "type", m.__class__.__name__).upper()
        content = m.content if isinstance(m.content, str) else str(m.content)
        return f"{role}: {content}"

    def summarize(self, state):
        summary = state.get("summary") or ""
        msgs: Sequence[AnyMessage] = state.get("messages", [])
        visible = [m for m in msgs if self._is_user_visible(m)]
        messages_text = "\n".join(self._render(m) for m in visible)

        prompt = (
            f"This is a summary to date:\n{summary}\n\n"
            f"Extend the summary considering the new messages:\n{messages_text}\n"
            if summary else
            f"Create a concise summary of the conversation below:\n{messages_text}\n"
        )

        prompt += ("\n### Rules:\n"
                   "\t- Provide a concise summary in the same language as the user."
                   "\n\t- Return only the summary, nothing else.\n")

        import logging
        logger = logging.getLogger(__name__)
        logger.info("[summarizer] prompt prepared")
        # logger.debug(prompt)
        t0 = time.perf_counter()
        resp = self.model.invoke(prompt)
        print(f"[summarizer] invoke took {time.perf_counter()-t0:.3f}s")
        new_summary = resp.content if isinstance(resp.content, str) else str(resp.content)
        print(new_summary)

        tail = list(msgs[-self.k_tail:]) if self.k_tail > 0 else []
        from langgraph.graph.message import REMOVE_ALL_MESSAGES
        return {"summary": new_summary, "messages": [REMOVE_ALL_MESSAGES, *tail]}

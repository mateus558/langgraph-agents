"""Small utilities for message handling and token estimation.

This module contains portable helpers used by multiple agents.
"""
import math
import re
from collections.abc import Iterable
from typing import Any


class TokenEstimator:
    """Provider-independent token estimator.

    - If `tiktoken` is available, uses `cl100k_base` encoding.
    - Otherwise falls back to a simple chars->tokens heuristic (~4 chars/token).
    """
    def __init__(self):
        self._mode = "heuristic"
        self._enc = None
        try:
            import tiktoken
            self._enc = tiktoken.get_encoding("cl100k_base")
            self._mode = "tiktoken"
        except Exception:
            self._enc = None
            self._mode = "heuristic"

    def count_text(self, text: str) -> int:
        # Empty or whitespace-only strings should count as 0 tokens
        if not text or (isinstance(text, str) and text.strip() == ""):
            return 0
        if self._mode == "tiktoken":
            try:
                assert self._enc is not None
                return len(self._enc.encode(text))
            except Exception:
                return self._heuristic_count(text)
        return self._heuristic_count(text)

    @staticmethod
    def _heuristic_count(text: str) -> int:
        t = re.sub(r"\s+", " ", text).strip()
        if not t:
            return 0
        return max(1, math.ceil(len(t) / 4))

    def count_messages(self, messages: Iterable[Any]) -> int:
        total = 0
        for m in messages:
            total += self.count_text(coerce_message_content(m))
        return total


def coerce_message_content(msg: Any) -> str:
    """Coerce message-like objects to a readable string.

    - If `.content` is a str, returns it.
    - If `.content` is a list (tool-call parts etc.), concatenates common text fields.
    - Otherwise falls back to str(...).
    """
    c = getattr(msg, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict):
                if "text" in p and isinstance(p["text"], str):
                    parts.append(p["text"])
                elif "content" in p and isinstance(p["content"], str):
                    parts.append(p["content"])
                else:
                    parts.append(str({k: type(v).__name__ for k, v in p.items()}))
            else:
                parts.append(str(p))
        return "\n".join(parts)
    return str(c)

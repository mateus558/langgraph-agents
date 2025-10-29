"""Language detection utilities for the websearch agent."""

from __future__ import annotations

import asyncio
from typing import Optional


class LangDetector:
    """Best-effort language detector with optional third-party deps.

    Priority order:
      1. pycld3 (fast, accurate)
      2. langdetect (widely available)
      3. heuristic fallback based on accents/stopwords

    Returns a two-letter ISO code (e.g., "en", "pt") suitable for Searx.
    """

    def __init__(self) -> None:
        try:
            import pycld3  # type: ignore
            self._cld3 = pycld3
        except Exception:  # pragma: no cover - optional dependency
            self._cld3 = None
        try:
            from langdetect import detect  # type: ignore
            self._detect = detect
        except Exception:  # pragma: no cover - optional dependency
            self._detect = None

    @staticmethod
    def _norm(code: Optional[str]) -> Optional[str]:
        if not code:
            return None
        code = code.lower().strip().split("-")[0]
        if code in {"pt", "en", "es", "fr", "de", "it"}:
            return code
        if code in {"ptbr", "pt_br"}:
            return "pt"
        return code[:2] if len(code) >= 2 else None

    async def detect(self, text: str) -> Optional[str]:
        """Detect language asynchronously using the best available backend."""

        txt = (text or "").strip()
        if not txt:
            return None

        if self._cld3 is not None:
            def _cld3_detect(t: str) -> Optional[str]:
                if self._cld3 is None:
                    return None
                lang = self._cld3.get_language(t)
                if lang and getattr(lang, "is_reliable", False):
                    return self._norm(getattr(lang, "language", None))
                return self._norm(getattr(lang, "language", None))

            try:
                return await asyncio.to_thread(_cld3_detect, txt)
            except Exception:  # pragma: no cover - optional dependency may fail
                pass

        if self._detect is not None:
            try:
                code = await asyncio.to_thread(self._detect, txt)
                return self._norm(code)
            except Exception:
                pass

        lowered = txt.lower()
        if any(ch in lowered for ch in "ãõáéíóúçâêô") or "quando" in lowered or "próximo" in lowered:
            return "pt"
        if "¿" in lowered or "¡" in lowered or "cuándo" in lowered:
            return "es"
        return None


__all__ = ["LangDetector"]

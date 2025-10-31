"""Time context helpers for prompts.

These utilities standardize how agents compute and pass time-related
variables into prompt templates, without coupling that logic to the
prompt abstraction itself.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover - Python <3.9 not supported here
    ZoneInfo = None  # type: ignore[assignment]


def _ensure_zoneinfo(tz: Any | None) -> Any:
    if tz is None:
        return ZoneInfo("UTC") if ZoneInfo else timezone.utc
    if isinstance(tz, str):
        try:
            return ZoneInfo(tz) if ZoneInfo else tz
        except Exception:
            return ZoneInfo("UTC") if ZoneInfo else timezone.utc
    return tz


def build_chat_clock_vars(tz: Any | None) -> dict[str, str]:
    """Return standardized clock variables for chat prompts.

    Keys:
        - clock: human-readable clock line used by ChatAgent
        - weekday: lowercase weekday name (monday, ...)
        - tz_name: IANA zone name when available, else str(tz)
        - iso_now: ISO 8601 timestamp in the local timezone
    """
    tz_obj = _ensure_zoneinfo(tz)
    now_dt = datetime.now(tz_obj)
    iso_now = now_dt.isoformat()
    weekday = now_dt.strftime("%A").lower()
    tz_name = getattr(tz_obj, "key", None) or str(tz_obj)
    clock = (
        f"Current time: {iso_now} ({weekday}) | Timezone: {tz_name}. "
        "Use this as the single source of truth for 'today', 'tomorrow', etc."
    )
    return {
        "clock": clock,
        "weekday": weekday,
        "tz_name": tz_name,
        "iso_now": iso_now,
    }


def build_web_time_vars(tz: Any | None) -> dict[str, str]:
    """Return standardized web-search time variables for prompts.

    Keys:
        - utc_time: "YYYY-MM-DD HH:MM:SS Z" in UTC
        - local_time: same format in provided local timezone
        - local_label: IANA zone name when available, else str(tz)
    """
    tz_obj = _ensure_zoneinfo(tz)
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(tz_obj)
    utc_str = now_utc.strftime("%Y-%m-%d %H:%M:%S %Z")
    local_str = now_local.strftime("%Y-%m-%d %H:%M:%S %Z")
    local_label = getattr(tz_obj, "key", None) or str(tz_obj)
    return {
        "utc_time": utc_str,
        "local_time": local_str,
        "local_label": local_label,
    }


__all__ = ["build_chat_clock_vars", "build_web_time_vars"]


"""Time context helpers for prompts.

These utilities standardize how agents compute and pass time-related
variables into prompt templates, without coupling that logic to the
prompt abstraction itself.
"""

from __future__ import annotations

from datetime import datetime, timezone, tzinfo
from typing import Any, Tuple

try:
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
except Exception:  # pragma: no cover - Python <3.9 not supported here
    ZoneInfo = None  # type: ignore[assignment]
    ZoneInfoNotFoundError = Exception  # type: ignore[assignment]


def _ensure_zoneinfo(tz: Any | None) -> Any:
    """Coerce value into a tzinfo, defaulting to UTC when unavailable."""
    if tz is None:
        return ZoneInfo("UTC") if ZoneInfo else timezone.utc
    if ZoneInfo and isinstance(tz, ZoneInfo):
        return tz
    if hasattr(tz, "tzname"):
        return tz
    if isinstance(tz, str):
        try:
            if ZoneInfo:
                return ZoneInfo(tz)
        except ZoneInfoNotFoundError:
            pass
        except Exception:
            pass
        return ZoneInfo("UTC") if ZoneInfo else timezone.utc
    if hasattr(tz, "utcoffset"):
        return tz
    return ZoneInfo("UTC") if ZoneInfo else timezone.utc


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


def resolve_timezone(name: str | None) -> Tuple[tzinfo, str, bool]:
    """Resolve a timezone name to a tzinfo with a normalized name and fallback flag.

    Args:
        name: IANA time zone name (e.g., "America/Sao_Paulo"). If None or invalid,
              UTC is returned as a safe default.

    Returns:
        (tz, normalized_name, fallback_applied)

    Notes:
        - No logging is performed here so callers can decide how/when to log.
        - ``normalized_name`` will be the IANA key when available, otherwise "UTC".
    """
    # If ZoneInfo is available and a name is provided, try to resolve it.
    if name and ZoneInfo:
        try:
            tz = ZoneInfo(name)
            norm = getattr(tz, "key", name)
            return tz, norm, False
        except ZoneInfoNotFoundError:
            pass
        except Exception:
            pass
    # Fallback to UTC
    return timezone.utc, "UTC", True

__all__.append("resolve_timezone")

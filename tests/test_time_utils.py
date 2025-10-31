from __future__ import annotations

from core.time import resolve_timezone


def test_resolve_timezone_valid():
    tz, norm, fell = resolve_timezone("UTC")
    assert hasattr(tz, "utcoffset")
    assert norm.upper() == "UTC"
    assert fell is False


def test_resolve_timezone_invalid_fallback():
    tz, norm, fell = resolve_timezone("Invalid/Zone")
    assert hasattr(tz, "utcoffset")
    assert norm == "UTC"
    assert fell is True


def test_resolve_timezone_none_fallback():
    tz, norm, fell = resolve_timezone(None)
    assert hasattr(tz, "utcoffset")
    assert norm == "UTC"
    assert fell is True


# coding=utf-8
"""Tests for observability sanitization/encoding safety."""

from self_ai.observability import sanitize_payload


def test_sanitize_payload_removes_control_chars() -> None:
    payload = {"msg": "hello\x00\x01world"}
    sanitized = sanitize_payload(payload)
    assert sanitized["msg"] == "helloworld"


def test_sanitize_payload_repairs_common_mojibake() -> None:
    payload = {"msg": "ä½ å¥½"}
    sanitized = sanitize_payload(payload)
    # best-effort repair should recover CJK for this common mojibake pattern
    assert "你" in sanitized["msg"]

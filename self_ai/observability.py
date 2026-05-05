# coding=utf-8
"""Lightweight runtime observability helpers for Self AI."""

import os
from datetime import datetime
import json
import re
from threading import Lock
import time
from typing import Any, Callable

TraceListener = Callable[[dict[str, Any]], None]

_LISTENERS: list[TraceListener] = []
_LOCK = Lock()


def _env_flag(primary: str, legacy: str, default: str = "true") -> bool:
    value = os.getenv(primary, os.getenv(legacy, default))
    return str(value).strip().lower() in {
        "1",
        "true",
        "yes",
    }


_SANITIZE_CONTROL = _env_flag("SELF_AI_TRACE_SANITIZE_CONTROL", "CF_TRACE_SANITIZE_CONTROL")
_REPAIR_MOJIBAKE = _env_flag("SELF_AI_TRACE_REPAIR_MOJIBAKE", "CF_TRACE_REPAIR_MOJIBAKE")
_MOJIBAKE_RE = re.compile(
    r"(?:\u00C3.|\u00E2.|\u00E5.|\u00E6.|\u00E7.|\u00F0.|\u00E4.|\u00EF.|\u951B|\u93C2|\u9225|\u9229)"
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _clean_control_chars(text: str) -> str:
    if not text:
        return text
    cleaned: list[str] = []
    for ch in text:
        code = ord(ch)
        if ch in ("\n", "\r", "\t"):
            cleaned.append(ch)
            continue
        # Keep printable unicode while removing control chars.
        if code < 32 or code == 127:
            continue
        cleaned.append(ch)
    return "".join(cleaned)


def _count_cjk(text: str) -> int:
    return len(_CJK_RE.findall(text or ""))


def _maybe_repair_mojibake(text: str) -> str:
    if not _REPAIR_MOJIBAKE or not text:
        return text
    if _count_cjk(text) > 0 and "\uFFFD" not in text:
        return text
    if not _MOJIBAKE_RE.search(text):
        return text
    candidates: list[str] = [text]
    for src in ("latin-1", "cp1252"):
        try:
            repaired = text.encode(src, errors="ignore").decode("utf-8", errors="ignore")
        except Exception:
            continue
        if repaired:
            candidates.append(repaired)

    def _score(value: str) -> tuple[int, int, int]:
        cjk = _count_cjk(value)
        replacement = value.count("\uFFFD")
        qmarks = value.count("?")
        return (cjk, -replacement, -qmarks)

    return max(candidates, key=_score)


def sanitize_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        cleaned = _clean_control_chars(payload) if _SANITIZE_CONTROL else payload
        return _maybe_repair_mojibake(cleaned)
    if isinstance(payload, dict):
        return {str(k): sanitize_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [sanitize_payload(v) for v in payload]
    if isinstance(payload, tuple):
        return tuple(sanitize_payload(v) for v in payload)
    return payload


def register_trace_listener(listener: TraceListener) -> None:
    """Register a callback to receive trace payloads."""
    with _LOCK:
        _LISTENERS.append(listener)


def unregister_trace_listener(listener: TraceListener) -> None:
    """Unregister a previously registered trace callback."""
    with _LOCK:
        if listener in _LISTENERS:
            _LISTENERS.remove(listener)


def format_trace_payload(payload: dict[str, Any]) -> str:
    """Format payload for human-readable logs."""
    safe_payload = sanitize_payload(payload)
    return "[self-ai] " + json.dumps(safe_payload, ensure_ascii=False, default=str)


def _notify_listeners(payload: dict[str, Any]) -> None:
    """Fan out trace payload to listeners."""
    with _LOCK:
        listeners = list(_LISTENERS)
    for listener in listeners:
        try:
            listener(payload)
        except Exception:
            continue


def make_trace_payload(event: str, **fields: Any) -> dict[str, Any]:
    """Build structured trace payload with UTC timestamp."""
    ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
    payload = {"event": event, "ts": ts, **fields}
    return sanitize_payload(payload) if _SANITIZE_CONTROL else payload


def trace(event: str, **fields: Any) -> None:
    """Print a structured trace line with UTC timestamp."""
    payload = make_trace_payload(event, **fields)
    _notify_listeners(payload)
    message = format_trace_payload(payload)
    print(message, flush=True)


def now_ms() -> int:
    """Return monotonic timestamp in milliseconds."""
    return int(time.perf_counter() * 1000)

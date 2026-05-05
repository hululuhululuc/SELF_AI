# coding=utf-8
"""Shared memory decision contract (model-facing + runtime validation)."""

from __future__ import annotations

from typing import Any

from ..text_utils import shorten_text

ALLOWED_DECAY_MODES = {"fast", "normal", "slow", "none"}
ALLOWED_MEMORY_CLASSES = {
    "ephemeral",
    "fact",
    "decision",
    "preference",
    "constraint",
    "issue",
}


def default_target_collection(memory_type: str, *, chat_collection: str = "cf_chat_memory") -> str:
    mtype = str(memory_type or "").strip().lower()
    if mtype == "issue":
        return "cf_review_memory"
    if mtype in {"preference", "constraint", "global_constraint"}:
        return "cf_preference_memory"
    return str(chat_collection or "cf_chat_memory")


def allowed_memory_collections(*, chat_collection: str = "cf_chat_memory") -> list[str]:
    base = [
        str(chat_collection or "cf_chat_memory"),
        "cf_review_memory",
        "cf_task_memory",
        "cf_preference_memory",
    ]
    out: list[str] = []
    for name in base:
        key = str(name or "").strip()
        if key and key not in out:
            out.append(key)
    return out


def build_memory_contract_brief(
    *,
    chat_collection: str = "cf_chat_memory",
    max_chars: int = 700,
) -> str:
    allowed = allowed_memory_collections(chat_collection=chat_collection)
    text = (
        "Memory contract (JSON-only, per memory_type profile): "
        "memory_type, memory_class, target_collection, importance(0..1), "
        "decay_mode(fast|normal|slow|none), pinned(bool), confidence(0..1), rationale. "
        f"target_collection must be one of: {', '.join(allowed)}. "
        "For side-effect facts, only commit if execution evidence proves state change."
    )
    return shorten_text(text, max_chars=max_chars)


def validate_profile_contract(
    profile: dict[str, Any],
    *,
    allowed_collections: list[str],
) -> list[str]:
    errors: list[str] = []
    memory_type = str(profile.get("memory_type", "") or "").strip()
    if not memory_type:
        errors.append("memory_type_missing")

    memory_class = str(profile.get("memory_class", "") or "").strip().lower()
    if memory_class not in ALLOWED_MEMORY_CLASSES:
        errors.append("memory_class_invalid")

    decay_mode = str(profile.get("decay_mode", "") or "").strip().lower()
    if decay_mode not in ALLOWED_DECAY_MODES:
        errors.append("decay_mode_invalid")

    try:
        importance = float(profile.get("importance", 0.0))
        if importance < 0.0 or importance > 1.0:
            errors.append("importance_out_of_range")
    except Exception:
        errors.append("importance_not_number")

    try:
        confidence = float(profile.get("confidence", 0.0))
        if confidence < 0.0 or confidence > 1.0:
            errors.append("confidence_out_of_range")
    except Exception:
        errors.append("confidence_not_number")

    target_collection = str(profile.get("target_collection", "") or "").strip()
    if not target_collection:
        errors.append("target_collection_missing")
    elif target_collection not in allowed_collections:
        errors.append("target_collection_invalid")

    return errors

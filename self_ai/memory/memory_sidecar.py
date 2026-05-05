# coding=utf-8
"""Memory sidecar agent: model-decided memory profile policy."""

from __future__ import annotations

import json
from typing import Any

from .memory_contract import (
    ALLOWED_MEMORY_CLASSES,
    allowed_memory_collections,
    default_target_collection,
    validate_profile_contract,
)

def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def _clamp01(value: Any, default: float) -> float:
    try:
        num = float(value)
    except Exception:
        return default
    return max(0.0, min(num, 1.0))


def _normalize_decay_mode(value: Any, default: str = "normal") -> str:
    mode = str(value or "").strip().lower()
    return mode if mode in {"fast", "normal", "slow", "none"} else default


def _normalize_profile(value: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    base = dict(fallback)
    if not isinstance(value, dict):
        return base
    base["memory_type"] = str(value.get("memory_type", base.get("memory_type", "")) or base.get("memory_type", ""))
    base["memory_class"] = str(value.get("memory_class", base["memory_class"]) or base["memory_class"])
    base["target_collection"] = str(
        value.get("target_collection", base.get("target_collection", ""))
        or base.get("target_collection", "")
    )
    base["importance"] = _clamp01(value.get("importance", base["importance"]), base["importance"])
    base["decay_mode"] = _normalize_decay_mode(value.get("decay_mode", base["decay_mode"]), base["decay_mode"])
    base["pinned"] = bool(value.get("pinned", base["pinned"]))
    base["confidence"] = _clamp01(value.get("confidence", base["confidence"]), base["confidence"])
    base["decision_source"] = str(value.get("decision_source", "model_sidecar_v1") or "model_sidecar_v1")
    base["rationale"] = str(value.get("rationale", "") or "")[:240]
    return base


class MemorySidecarAgent:
    """Model-driven memory profile selector for append-turn candidates."""

    def build_prompt(
        self,
        *,
        task: str,
        response: str,
        execution_summary: dict[str, Any],
        candidates_summary: list[dict[str, Any]],
        recent_turns_summary: list[dict[str, Any]],
        chat_collection: str = "cf_chat_memory",
    ) -> str:
        allowed_collections = allowed_memory_collections(chat_collection=chat_collection)
        allowed_classes = ", ".join(sorted(ALLOWED_MEMORY_CLASSES))
        return (
            "You are a memory manager for an agentic coding system.\n"
            "Return JSON only. No markdown.\n"
            "Decide memory profile per memory_type for long-term usefulness.\n"
            "If memory is user preference / stable constraint / critical decision, use slow/none decay and higher importance.\n"
            "If memory is ephemeral task chatter, use fast decay and lower importance.\n"
            "Do not invent execution success; rely on execution_summary.\n\n"
            "Required fields in each profile: memory_type, memory_class, target_collection, importance, decay_mode, pinned, confidence, rationale.\n"
            f"Allowed memory_class values: {allowed_classes}.\n"
            f"Allowed target_collection values: {', '.join(allowed_collections)}.\n"
            "Keep rationale short.\n\n"
            "JSON schema:\n"
            "{\n"
            '  "profiles": {\n'
            '    "turn_summary": {"memory_type":"turn_summary", "memory_class":"ephemeral|fact|decision|preference|constraint|issue", "target_collection":"...", "importance":0-1, "decay_mode":"fast|normal|slow|none", "pinned":true|false, "confidence":0-1, "decision_source":"model_sidecar_v1", "rationale":"..."},\n'
            '    "response_summary": {...},\n'
            '    "issue": {...}\n'
            "  },\n"
            '  "global_notes": "optional"\n'
            "}\n\n"
            f"task: {task[:700]}\n"
            f"response: {response[:700]}\n"
            f"execution_summary: {json.dumps(execution_summary, ensure_ascii=False, default=str)[:900]}\n"
            f"candidates_summary: {json.dumps(candidates_summary, ensure_ascii=False, default=str)[:1200]}\n"
            f"recent_turns_summary: {json.dumps(recent_turns_summary, ensure_ascii=False, default=str)[:1000]}\n"
        )

    def normalize_decision(
        self,
        raw: dict[str, Any] | None,
        *,
        chat_collection: str = "cf_chat_memory",
    ) -> dict[str, Any]:
        parsed = raw if isinstance(raw, dict) else {}
        profiles = parsed.get("profiles", {}) if isinstance(parsed.get("profiles"), dict) else {}
        allowed_collections = allowed_memory_collections(chat_collection=chat_collection)
        defaults = {
            "turn_summary": {
                "memory_type": "turn_summary",
                "memory_class": "ephemeral",
                "target_collection": default_target_collection("turn_summary", chat_collection=chat_collection),
                "importance": 0.45,
                "decay_mode": "fast",
                "pinned": False,
                "confidence": 0.7,
                "decision_source": "system_default_v1",
                "rationale": "",
            },
            "response_summary": {
                "memory_type": "response_summary",
                "memory_class": "fact",
                "target_collection": default_target_collection("response_summary", chat_collection=chat_collection),
                "importance": 0.62,
                "decay_mode": "normal",
                "pinned": False,
                "confidence": 0.8,
                "decision_source": "system_default_v1",
                "rationale": "",
            },
            "issue": {
                "memory_type": "issue",
                "memory_class": "issue",
                "target_collection": default_target_collection("issue", chat_collection=chat_collection),
                "importance": 0.68,
                "decay_mode": "slow",
                "pinned": False,
                "confidence": 0.85,
                "decision_source": "system_default_v1",
                "rationale": "",
            },
        }
        normalized_profiles: dict[str, Any] = {}
        validation_errors: dict[str, list[str]] = {}
        for memory_type, fallback in defaults.items():
            profile = _normalize_profile(
                profiles.get(memory_type),
                fallback,
            )
            profile_errors = validate_profile_contract(
                profile,
                allowed_collections=allowed_collections,
            )
            if profile_errors:
                validation_errors[memory_type] = profile_errors
                profile = dict(fallback)
            normalized_profiles[memory_type] = profile
        return {
            "profiles": normalized_profiles,
            "global_notes": str(parsed.get("global_notes", "") or "")[:300],
            "decision_source": "model_sidecar_v1",
            "validation_errors": validation_errors,
        }

    async def decide(
        self,
        *,
        task: str,
        response: str,
        execution_summary: dict[str, Any],
        candidates_summary: list[dict[str, Any]],
        recent_turns_summary: list[dict[str, Any]],
        model_generate: Any,
        chat_collection: str = "cf_chat_memory",
    ) -> dict[str, Any]:
        prompt = self.build_prompt(
            task=task,
            response=response,
            execution_summary=execution_summary,
            candidates_summary=candidates_summary,
            recent_turns_summary=recent_turns_summary,
            chat_collection=chat_collection,
        )
        model_payload = await model_generate(prompt)
        if isinstance(model_payload, dict) and not bool(model_payload.get("ok", True)):
            error = model_payload.get("error", {}) if isinstance(model_payload.get("error"), dict) else {}
            error_type = str(error.get("type", "ModelGenerateError") or "ModelGenerateError")
            raise RuntimeError(f"memory sidecar model call failed: {error_type}")
        model_text = ""
        if isinstance(model_payload, dict):
            # model.generate tool output pattern
            if isinstance(model_payload.get("data"), dict):
                model_text = str(model_payload["data"].get("response", "") or "")
            else:
                model_text = str(model_payload.get("response", "") or "")
        parsed = _extract_json_object(model_text)
        if not isinstance(parsed, dict):
            raise ValueError("memory sidecar returned non-json decision")
        return self.normalize_decision(parsed, chat_collection=chat_collection)

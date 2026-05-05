# coding=utf-8
"""Phase-1 session memory store: chat isolation + append-only log shards."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from ..config import settings
from .memory_contract import (
    allowed_memory_collections,
    default_target_collection,
    validate_profile_contract,
)
from ..text_utils import normalize_text, shorten_text


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_date() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _safe_chat_id(value: str) -> str:
    text = (value or "default").strip()
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in ("-", "_"))
    return cleaned or "default"


def _safe_user_id(value: str | None) -> str:
    text = (value or "anonymous").strip()
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in ("-", "_", "@", "."))
    return cleaned or "anonymous"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _summarize_for_retrieval(text: str, *, max_chars: int = 220) -> str:
    normalized = normalize_text(text)
    compact = " ".join(normalized.split())
    return shorten_text(compact, max_chars=max_chars)


@dataclass(slots=True)
class _ShardInfo:
    index: int
    date: str
    file: str
    turn_count: int
    size_bytes: int
    first_turn_id: int | None
    last_turn_id: int | None
    latest_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "date": self.date,
            "file": self.file,
            "turn_count": self.turn_count,
            "size_bytes": self.size_bytes,
            "first_turn_id": self.first_turn_id,
            "last_turn_id": self.last_turn_id,
            "latest_summary": self.latest_summary,
        }


class ChatMemoryStore:
    """Chat-isolated append-only memory writer for Phase-1."""

    def __init__(
        self,
        *,
        root_dir: str | Path,
        max_shard_bytes: int = 1_048_576,
        max_turns_per_shard: int = 200,
    ) -> None:
        self.root_dir = Path(root_dir).resolve()
        self.max_shard_bytes = max(1, int(max_shard_bytes))
        self.max_turns_per_shard = max(1, int(max_turns_per_shard))
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _normalize_turn_record(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "memory_id": str(item.get("memory_id", "")),
            "chat_id": str(item.get("chat_id", "")),
            "user_id": str(item.get("user_id", "")),
            "run_id": str(item.get("run_id", "")),
            "turn_id": int(item.get("turn_id", 0) or 0),
            "timestamp": str(item.get("timestamp", "")),
            "task_preview": str(item.get("task_preview", "")),
            "response_preview": str(item.get("response_preview", "")),
            "retrieval_summary": str(item.get("retrieval_summary", "")),
            "content_hash": str(item.get("content_hash", "")),
            "memory_profile": item.get("memory_profile", {})
            if isinstance(item.get("memory_profile"), dict)
            else {},
            "execution": item.get("execution", {})
            if isinstance(item.get("execution"), dict)
            else {},
            "write_policy": item.get("write_policy", {})
            if isinstance(item.get("write_policy"), dict)
            else {},
            "memory_candidates": item.get("memory_candidates", [])
            if isinstance(item.get("memory_candidates"), list)
            else [],
            "committed_memories": item.get("committed_memories", [])
            if isinstance(item.get("committed_memories"), list)
            else [],
            "memory_decision": item.get("memory_decision", {})
            if isinstance(item.get("memory_decision"), dict)
            else {},
            "refs": item.get("refs", {}) if isinstance(item.get("refs"), dict) else {},
        }

    def _chat_dir(self, chat_id: str) -> Path:
        folder = self.root_dir / _safe_chat_id(chat_id)
        folder.mkdir(parents=True, exist_ok=True)
        return folder

    def _global_constraints_path(self, chat_dir: Path) -> Path:
        return chat_dir / "global_constraints.json"

    def _load_global_constraints(self, chat_dir: Path) -> dict[str, Any]:
        path = self._global_constraints_path(chat_dir)
        if not path.exists():
            return {
                "version": 1,
                "chat_id": str(chat_dir.name),
                "updated_at": "",
                "next_id": 1,
                "items": [],
            }
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                items = parsed.get("items", [])
                normalized_items = self._normalize_global_constraint_items(
                    list(items) if isinstance(items, list) else []
                )
                max_id = 0
                for row in normalized_items:
                    raw_id = str(row.get("id", "") or "")
                    if raw_id.startswith("gc-"):
                        try:
                            max_id = max(max_id, int(raw_id.split("-", 1)[1]))
                        except Exception:
                            pass
                return {
                    "version": int(parsed.get("version", 1) or 1),
                    "chat_id": str(parsed.get("chat_id", chat_dir.name) or chat_dir.name),
                    "updated_at": str(parsed.get("updated_at", "") or ""),
                    "next_id": max(int(parsed.get("next_id", 1) or 1), max_id + 1),
                    "items": normalized_items,
                }
        except Exception:
            pass
        return {
            "version": 1,
            "chat_id": str(chat_dir.name),
            "updated_at": "",
            "next_id": 1,
            "items": [],
        }

    def _save_global_constraints(self, chat_dir: Path, payload: dict[str, Any]) -> None:
        path = self._global_constraints_path(chat_dir)
        _write_json(path, payload)

    @staticmethod
    def _normalize_constraint_content(value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())

    @staticmethod
    def _normalize_global_constraint_items(items: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        fallback_id = 1
        for row in items:
            if not isinstance(row, dict):
                continue
            content = str(row.get("content", "") or row.get("preview", "") or "").strip()
            if not content:
                continue
            item_id = str(row.get("id", "") or row.get("constraint_id", "") or "").strip()
            if not item_id:
                item_id = f"gc-{fallback_id}"
                fallback_id += 1
            normalized.append(
                {
                    "id": item_id,
                    "content": shorten_text(content, max_chars=320),
                    "status": str(row.get("status", "active") or "active"),
                    "updated_at": str(row.get("updated_at", "") or ""),
                    "source": str(row.get("source", row.get("decision_source", "constraint_maintainer")) or "constraint_maintainer"),
                    "confidence": max(0.0, min(float(row.get("confidence", 0.85) or 0.85), 1.0)),
                }
            )
        return normalized

    def apply_global_constraint_ops(
        self,
        *,
        chat_id: str,
        ops: list[dict[str, Any]],
        source: str = "constraint_maintainer",
    ) -> dict[str, Any]:
        chat_dir = self._chat_dir(chat_id)
        state = self._load_global_constraints(chat_dir)
        now = _utc_now_iso()
        safe_chat_id = _safe_chat_id(chat_id)
        items = [dict(x) for x in state.get("items", []) if isinstance(x, dict)]
        next_id = max(1, int(state.get("next_id", 1) or 1))
        changed = False
        applied_ops: list[dict[str, Any]] = []

        def _active_items() -> list[dict[str, Any]]:
            return [
                row
                for row in items
                if str(row.get("status", "active") or "active").strip().lower() == "active"
                and str(row.get("content", "") or "").strip()
            ]

        def _move_to_front(item_id: str) -> None:
            for idx, row in enumerate(items):
                if str(row.get("id", "") or "") == item_id:
                    items.insert(0, items.pop(idx))
                    return

        for op in ops[:12]:
            if not isinstance(op, dict):
                continue
            kind = str(op.get("op", "") or "").strip().lower()
            if kind == "noop":
                applied_ops.append({"op": "noop"})
                continue
            if kind == "add":
                content = shorten_text(str(op.get("content", "") or "").strip(), max_chars=320)
                if not content:
                    continue
                key = self._normalize_constraint_content(content)
                existing_id = ""
                for row in _active_items():
                    if self._normalize_constraint_content(row.get("content")) == key:
                        existing_id = str(row.get("id", "") or "")
                        break
                if existing_id:
                    _move_to_front(existing_id)
                    applied_ops.append({"op": "dedupe", "id": existing_id})
                    changed = True
                    continue
                item_id = f"gc-{next_id}"
                next_id += 1
                items.insert(
                    0,
                    {
                        "id": item_id,
                        "content": content,
                        "status": "active",
                        "updated_at": now,
                        "source": source,
                        "confidence": max(0.0, min(float(op.get("confidence", 0.85) or 0.85), 1.0)),
                    },
                )
                applied_ops.append({"op": "add", "id": item_id})
                changed = True
                continue
            if kind == "update":
                item_id = str(op.get("id", "") or "").strip()
                content = shorten_text(str(op.get("content", "") or "").strip(), max_chars=320)
                if not item_id or not content:
                    continue
                for row in items:
                    if str(row.get("id", "") or "") != item_id:
                        continue
                    row["content"] = content
                    row["status"] = "active"
                    row["updated_at"] = now
                    row["source"] = source
                    row["confidence"] = max(0.0, min(float(op.get("confidence", row.get("confidence", 0.85)) or 0.85), 1.0))
                    _move_to_front(item_id)
                    applied_ops.append({"op": "update", "id": item_id})
                    changed = True
                    break
                continue
            if kind in {"delete", "remove"}:
                item_id = str(op.get("id", "") or "").strip()
                if not item_id:
                    continue
                before_len = len(items)
                items = [row for row in items if str(row.get("id", "") or "") != item_id]
                if len(items) != before_len:
                    applied_ops.append({"op": "delete", "id": item_id})
                    changed = True

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in items:
            content = str(row.get("content", "") or "").strip()
            if not content:
                continue
            key = self._normalize_constraint_content(content)
            if key in seen:
                changed = True
                continue
            seen.add(key)
            normalized = {
                "id": str(row.get("id", "") or f"gc-{next_id}"),
                "content": shorten_text(content, max_chars=320),
                "status": str(row.get("status", "active") or "active"),
                "updated_at": str(row.get("updated_at", "") or now),
                "source": str(row.get("source", source) or source),
                "confidence": max(0.0, min(float(row.get("confidence", 0.85) or 0.85), 1.0)),
            }
            if not normalized["id"]:
                normalized["id"] = f"gc-{next_id}"
                next_id += 1
                changed = True
            deduped.append(normalized)
        items = deduped[:64]

        payload = {
            "version": int(state.get("version", 1) or 1),
            "chat_id": safe_chat_id,
            "updated_at": now if changed else str(state.get("updated_at", "") or ""),
            "next_id": next_id,
            "items": items,
        }
        if changed:
            self._save_global_constraints(chat_dir, payload)
        return {
            "chat_id": safe_chat_id,
            "changed": changed,
            "applied_ops": applied_ops,
            "items": [x for x in items if str(x.get("status", "active") or "active") == "active"],
            "updated_at": payload["updated_at"],
            "next_id": next_id,
        }

    @staticmethod
    def _control_event_count(result: dict[str, Any]) -> int:
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        events = metadata.get("control_events", [])
        return len(events) if isinstance(events, list) else 0

    @staticmethod
    def _terminal_errors(result: dict[str, Any]) -> list[dict[str, Any]]:
        errors = result.get("errors", [])
        if not isinstance(errors, list):
            return []
        terminal: list[dict[str, Any]] = []
        for item in errors:
            if not isinstance(item, dict):
                continue
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            if str(metadata.get("signal_class", "") or "").strip().lower() == "control_flow":
                continue
            if str(item.get("type", "") or "") == "CompletionGateBlocked":
                continue
            terminal.append(item)
        return terminal

    @staticmethod
    def _collect_execution_evidence(result: dict[str, Any]) -> dict[str, Any]:
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        recent = metadata.get("recent_tool_results", [])
        if not isinstance(recent, list):
            recent = []
        success_calls = 0
        failed_calls = 0
        successful_tools: list[str] = []
        failed_tools: list[str] = []
        state_change_committed = 0
        goal_aligned_state_change = 0
        for item in recent:
            if not isinstance(item, dict):
                continue
            tool_name = str(item.get("tool_name", "") or "")
            ok = bool(item.get("ok", False))
            if ok:
                success_calls += 1
                if tool_name:
                    successful_tools.append(tool_name)
            else:
                failed_calls += 1
                if tool_name:
                    failed_tools.append(tool_name)
            flag = item.get("execution_flag", {})
            if isinstance(flag, dict) and bool(flag.get("state_change_committed", False)):
                state_change_committed += 1
                goal_check = item.get("goal_check", {})
                if isinstance(goal_check, dict) and bool(goal_check.get("aligned", False)):
                    goal_aligned_state_change += 1

        goal_contract = metadata.get("goal_contract", {})
        if not isinstance(goal_contract, dict):
            goal_contract = {}
        requires_state_change = bool(goal_contract.get("requires_side_effect", False))

        return {
            "requires_state_change": requires_state_change,
            "successful_calls": success_calls,
            "failed_calls": failed_calls,
            "successful_tools": successful_tools[-8:],
            "failed_tools": failed_tools[-8:],
            "state_change_committed_calls": state_change_committed,
            "goal_aligned_state_change_calls": goal_aligned_state_change,
            "has_successful_execution": success_calls > 0,
            "has_state_change_committed": state_change_committed > 0,
            "has_goal_aligned_state_change": goal_aligned_state_change > 0,
            "terminal_error_count": len(ChatMemoryStore._terminal_errors(result)),
            "control_event_count": ChatMemoryStore._control_event_count(result),
        }

    @staticmethod
    def _build_memory_candidates(
        *,
        chat_id: str,
        user_id: str | None,
        run_id: str,
        turn_id: int,
        now: str,
        task: str,
        response: str,
        result: dict[str, Any],
        retrieval_summary: str,
    ) -> list[dict[str, Any]]:
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        execution_state = metadata.get("execution_state", {}) if isinstance(metadata.get("execution_state"), dict) else {}
        errors = ChatMemoryStore._terminal_errors(result)
        error_count = len(errors)
        control_event_count = ChatMemoryStore._control_event_count(result)
        gate = result.get("quality_gate", {}) if isinstance(result.get("quality_gate"), dict) else {}

        base = {
            "chat_id": _safe_chat_id(chat_id),
            "user_id": _safe_user_id(user_id),
            "run_id": str(run_id),
            "turn_id": int(turn_id),
            "timestamp": now,
            "confidence": 0.7,
        }

        candidates: list[dict[str, Any]] = [
            {
                **base,
                "memory_type": "turn_summary",
                "memory_class": "ephemeral",
                "importance": 0.45,
                "decay_mode": "fast",
                "pinned": False,
                "memory_id": f"{_safe_chat_id(chat_id)}-turn-{turn_id}-summary",
                "content_preview": retrieval_summary,
                "refs": {
                    "quality_gate_decision": str(gate.get("decision", "")),
                    "error_count": error_count,
                    "control_event_count": control_event_count,
                },
            }
        ]
        if response.strip():
            candidates.append(
                {
                    **base,
                    "memory_type": "response_summary",
                    "memory_class": "fact",
                    "importance": 0.62,
                    "decay_mode": "normal",
                    "pinned": False,
                    "memory_id": f"{_safe_chat_id(chat_id)}-turn-{turn_id}-response",
                    "content_preview": _summarize_for_retrieval(response, max_chars=220),
                    "refs": {
                        "goal_side_effect_done": bool(execution_state.get("side_effect_done", False)),
                        "goal_aligned_side_effect": bool(execution_state.get("goal_aligned_side_effect", False)),
                    },
                    "confidence": 0.8,
                }
            )
        if error_count > 0:
            last_error = {}
            if isinstance(errors, list) and errors and isinstance(errors[-1], dict):
                last_error = errors[-1]
            candidates.append(
                {
                    **base,
                    "memory_type": "issue",
                    "memory_class": "issue",
                    "importance": 0.68,
                    "decay_mode": "slow",
                    "pinned": False,
                    "memory_id": f"{_safe_chat_id(chat_id)}-turn-{turn_id}-issue",
                    "content_preview": _summarize_for_retrieval(
                        f"Task: {task}\nError: {last_error.get('type', '')}:{last_error.get('message', '')}",
                        max_chars=220,
                    ),
                    "refs": {
                        "error_type": str(last_error.get("type", "")),
                        "stage": str(last_error.get("stage", "")),
                    },
                    "confidence": 0.85,
                }
            )
        return candidates

    @staticmethod
    def _apply_memory_traits(
        candidates: list[dict[str, Any]],
        *,
        evidence: dict[str, Any],
        memory_decision: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        requires_state_change = bool(evidence.get("requires_state_change", False))
        has_goal_aligned_state_change = bool(
            evidence.get("has_goal_aligned_state_change", False)
        )
        chat_collection = str(
            getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory")
            or "cf_chat_memory"
        )
        allowed_collections = allowed_memory_collections(chat_collection=chat_collection)
        profiles = {}
        if isinstance(memory_decision, dict):
            profiles = memory_decision.get("profiles", {})
            if not isinstance(profiles, dict):
                profiles = {}
        result: list[dict[str, Any]] = []
        validation_errors: list[dict[str, Any]] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            enriched = dict(item)
            memory_type = str(enriched.get("memory_type", "") or "")
            profile = profiles.get(memory_type, {})
            profile_applied = False
            if isinstance(profile, dict):
                profile_errors = validate_profile_contract(
                    {
                        "memory_type": str(profile.get("memory_type", memory_type) or memory_type),
                        "memory_class": profile.get("memory_class", ""),
                        "target_collection": profile.get("target_collection", ""),
                        "importance": profile.get("importance", 0.5),
                        "decay_mode": profile.get("decay_mode", "normal"),
                        "pinned": profile.get("pinned", False),
                        "confidence": profile.get("confidence", 0.7),
                        "rationale": profile.get("rationale", ""),
                    },
                    allowed_collections=allowed_collections,
                )
                if profile_errors:
                    validation_errors.append(
                        {
                            "memory_type": memory_type,
                            "errors": profile_errors,
                            "decision_source": str(profile.get("decision_source", "model_sidecar_v1") or "model_sidecar_v1"),
                        }
                    )
                else:
                    profile_applied = True
                if "memory_class" in profile:
                    enriched["memory_class"] = str(profile.get("memory_class", enriched.get("memory_class", "")) or enriched.get("memory_class", ""))
                if "target_collection" in profile:
                    enriched["target_collection"] = str(
                        profile.get(
                            "target_collection",
                            default_target_collection(memory_type, chat_collection=chat_collection),
                        )
                        or default_target_collection(memory_type, chat_collection=chat_collection)
                    )
                if "importance" in profile:
                    enriched["importance"] = max(0.0, min(float(profile.get("importance", enriched.get("importance", 0.5)) or enriched.get("importance", 0.5)), 1.0))
                if "decay_mode" in profile:
                    mode = str(profile.get("decay_mode", enriched.get("decay_mode", "normal")) or "normal").strip().lower()
                    if mode in {"fast", "normal", "slow", "none"}:
                        enriched["decay_mode"] = mode
                if "pinned" in profile:
                    enriched["pinned"] = bool(profile.get("pinned", enriched.get("pinned", False)))
                if "confidence" in profile:
                    enriched["confidence"] = max(0.0, min(float(profile.get("confidence", enriched.get("confidence", 0.7)) or enriched.get("confidence", 0.7)), 1.0))
                if "decision_source" in profile:
                    enriched["decision_source"] = str(profile.get("decision_source", "model_sidecar_v1") or "model_sidecar_v1")
                if "rationale" in profile:
                    enriched["rationale"] = str(profile.get("rationale", "") or "")[:240]
            if "target_collection" not in enriched:
                enriched["target_collection"] = default_target_collection(
                    memory_type,
                    chat_collection=chat_collection,
                )
            if memory_type == "response_summary" and requires_state_change and has_goal_aligned_state_change:
                enriched["importance"] = max(
                    float(enriched.get("importance", 0.62) or 0.62),
                    0.78,
                )
                enriched["decay_mode"] = "slow"
                enriched["memory_class"] = "decision"
            if memory_type == "issue":
                enriched["confidence"] = max(
                    float(enriched.get("confidence", 0.85) or 0.85),
                    0.85,
                )
            if "decision_source" not in enriched:
                enriched["decision_source"] = "system_default_v1"
            if profile_applied:
                enriched["decision_source"] = str(
                    enriched.get("decision_source", "model_sidecar_v1") or "model_sidecar_v1"
                )
            enriched["pinned"] = bool(enriched.get("pinned", False))
            enriched["importance"] = max(0.0, min(float(enriched.get("importance", 0.5) or 0.5), 1.0))
            enriched["confidence"] = max(0.0, min(float(enriched.get("confidence", 0.7) or 0.7), 1.0))
            if str(enriched.get("decay_mode", "")).strip() not in {"fast", "normal", "slow", "none"}:
                enriched["decay_mode"] = "normal"
            target_collection = str(enriched.get("target_collection", "") or "").strip()
            if target_collection not in allowed_collections:
                validation_errors.append(
                    {
                        "memory_type": memory_type,
                        "errors": ["target_collection_invalid_after_normalization"],
                        "target_collection": target_collection,
                    }
                )
            result.append(enriched)
        return result, validation_errors

    @staticmethod
    def _commit_candidates(
        candidates: list[dict[str, Any]],
        *,
        evidence: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        requires_state_change = bool(evidence.get("requires_state_change", False))
        has_success = bool(evidence.get("has_successful_execution", False))
        has_state_change = bool(evidence.get("has_state_change_committed", False))
        has_goal_aligned_state_change = bool(
            evidence.get("has_goal_aligned_state_change", False)
        )
        has_response_summary = any(
            isinstance(item, dict)
            and str(item.get("memory_type", "") or "").strip().lower() == "response_summary"
            and bool(str(item.get("content_preview", "") or "").strip())
            for item in candidates
        )

        # For non-side-effect tasks, allow model-declared response memory to be kept
        # when there is meaningful response content, even without tool execution.
        fact_write_allowed = (
            (has_success and has_goal_aligned_state_change)
            if requires_state_change
            else (has_success or has_response_summary)
        )
        terminal_error_count = int(evidence.get("terminal_error_count", 0) or 0)
        control_event_count = int(evidence.get("control_event_count", 0) or 0)
        issue_write_allowed = terminal_error_count > 0

        committed: list[dict[str, Any]] = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            memory_type = str(item.get("memory_type", "") or "")
            should_commit = False
            if memory_type in {"turn_summary", "response_summary"}:
                should_commit = fact_write_allowed
            elif memory_type == "issue":
                should_commit = issue_write_allowed
            if not should_commit:
                continue
            committed.append(item)

        write_policy = {
            "requires_state_change": requires_state_change,
            "fact_write_allowed": fact_write_allowed,
            "issue_write_allowed": issue_write_allowed,
            "terminal_error_count": terminal_error_count,
            "control_event_count": control_event_count,
            "issue_suppressed_due_to_terminal_success": bool(
                terminal_error_count == 0 and control_event_count > 0
            ),
            "has_successful_execution": has_success,
            "has_response_summary": has_response_summary,
            "has_state_change_committed": has_state_change,
            "has_goal_aligned_state_change": has_goal_aligned_state_change,
        }
        return committed, write_policy

    @staticmethod
    def _build_memory_profile(
        committed_memories: list[dict[str, Any]],
        *,
        write_policy: dict[str, Any],
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        if not committed_memories:
            return {
                "primary_class": "ephemeral",
                "importance": 0.3,
                "decay_mode": "fast",
                "pinned": False,
                "confidence": 0.5,
                "decision_source": "system_default_v1",
            }
        preferred: dict[str, Any] | None = None
        for item in committed_memories:
            if not isinstance(item, dict):
                continue
            m_class = str(item.get("memory_class", "") or "")
            if m_class in {"decision", "preference", "constraint"}:
                preferred = item
                break
        if preferred is None:
            preferred = committed_memories[0]
        profile = {
            "primary_class": str(preferred.get("memory_class", "fact")),
            "importance": max(0.0, min(float(preferred.get("importance", 0.6) or 0.6), 1.0)),
            "decay_mode": str(preferred.get("decay_mode", "normal")),
            "pinned": bool(preferred.get("pinned", False)),
            "confidence": max(0.0, min(float(preferred.get("confidence", 0.7) or 0.7), 1.0)),
            "decision_source": str(preferred.get("decision_source", "system_default_v1")),
            "fact_write_allowed": bool(write_policy.get("fact_write_allowed", False)),
            "requires_state_change": bool(write_policy.get("requires_state_change", False)),
            "has_goal_aligned_state_change": bool(
                evidence.get("has_goal_aligned_state_change", False)
            ),
        }
        if profile["decay_mode"] not in {"fast", "normal", "slow", "none"}:
            profile["decay_mode"] = "normal"
        return profile

    @staticmethod
    def _manifest_path(chat_dir: Path) -> Path:
        return chat_dir / "manifest.json"

    @staticmethod
    def _summary_path(chat_dir: Path) -> Path:
        return chat_dir / "summary-v1.json"

    def _default_manifest(self, *, chat_id: str) -> dict[str, Any]:
        now = _utc_now_iso()
        return {
            "chat_id": _safe_chat_id(chat_id),
            "created_at": now,
            "updated_at": now,
            "next_turn_id": 1,
            "active_shard_file": "",
            "shards": [],
            "latest_summary": "",
            "summary_versions": [],
            "archived_shards": {},
            "last_compacted_turn_id": 0,
            "last_compacted_at": "",
        }

    def _load_manifest(self, chat_dir: Path, *, chat_id: str) -> dict[str, Any]:
        path = self._manifest_path(chat_dir)
        if not path.exists():
            manifest = self._default_manifest(chat_id=chat_id)
            _write_json(path, manifest)
            return manifest
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        manifest = self._default_manifest(chat_id=chat_id)
        _write_json(path, manifest)
        return manifest

    def _save_manifest(self, chat_dir: Path, manifest: dict[str, Any]) -> None:
        manifest["updated_at"] = _utc_now_iso()
        _write_json(self._manifest_path(chat_dir), manifest)

    @staticmethod
    def _parse_shard_index(file_name: str) -> int:
        text = str(file_name or "").strip()
        if not text.endswith(".ndjson"):
            return 0
        parts = text.removesuffix(".ndjson").split("-")
        if len(parts) < 3:
            return 0
        try:
            return int(parts[-1])
        except Exception:
            return 0

    def repair_manifest(self, *, chat_id: str) -> dict[str, Any]:
        """Self-check and repair manifest consistency using shard files."""
        chat_dir = self._chat_dir(chat_id)
        raw = self._load_manifest(chat_dir, chat_id=chat_id)
        changed = False

        manifest = dict(raw)
        now = _utc_now_iso()
        if not str(manifest.get("created_at", "")).strip():
            manifest["created_at"] = now
            changed = True
        manifest["chat_id"] = _safe_chat_id(chat_id)
        if manifest["chat_id"] != raw.get("chat_id"):
            changed = True

        shards_raw = manifest.get("shards", [])
        clean_shards: list[_ShardInfo] = []
        if not isinstance(shards_raw, list):
            shards_raw = []
            changed = True

        for item in shards_raw:
            if not isinstance(item, dict):
                changed = True
                continue
            file_name = str(item.get("file", "")).strip()
            if not file_name:
                changed = True
                continue
            shard_path = chat_dir / file_name
            if not shard_path.exists():
                changed = True
                continue
            index = int(item.get("index", 0) or 0)
            if index <= 0:
                index = self._parse_shard_index(file_name)
                changed = True
            shard = _ShardInfo(
                index=max(1, index),
                date=str(item.get("date", "") or _utc_date()),
                file=file_name,
                turn_count=0,
                size_bytes=0,
                first_turn_id=None,
                last_turn_id=None,
                latest_summary="",
            )
            clean_shards.append(shard)

        # Add missing shard files that are not indexed yet.
        known_files = {s.file for s in clean_shards}
        for shard_path in sorted(chat_dir.glob("turns-*.ndjson")):
            if shard_path.name in known_files:
                continue
            clean_shards.append(
                _ShardInfo(
                    index=max(1, self._parse_shard_index(shard_path.name)),
                    date=_utc_date(),
                    file=shard_path.name,
                    turn_count=0,
                    size_bytes=0,
                    first_turn_id=None,
                    last_turn_id=None,
                    latest_summary="",
                )
            )
            changed = True

        clean_shards.sort(key=lambda s: (s.index, s.file))
        # Ensure unique indexes.
        used_indexes: set[int] = set()
        next_index = 1
        for shard in clean_shards:
            if shard.index in used_indexes:
                while next_index in used_indexes:
                    next_index += 1
                shard.index = next_index
                changed = True
            used_indexes.add(shard.index)

        max_turn_id = 0
        latest_summary = str(manifest.get("latest_summary", "") or "")
        for shard in clean_shards:
            shard_path = chat_dir / shard.file
            try:
                shard.size_bytes = shard_path.stat().st_size
            except Exception:
                shard.size_bytes = 0
            try:
                lines = shard_path.read_text(encoding="utf-8").splitlines()
            except Exception:
                lines = []
            turn_ids: list[int] = []
            last_summary = ""
            for line in lines:
                text = line.strip()
                if not text:
                    continue
                try:
                    parsed = json.loads(text)
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue
                tid = int(parsed.get("turn_id", 0) or 0)
                if tid > 0:
                    turn_ids.append(tid)
                    max_turn_id = max(max_turn_id, tid)
                summary = str(parsed.get("retrieval_summary", "") or "")
                if summary:
                    last_summary = summary
            shard.turn_count = len(turn_ids)
            shard.first_turn_id = min(turn_ids) if turn_ids else None
            shard.last_turn_id = max(turn_ids) if turn_ids else None
            shard.latest_summary = last_summary
            if last_summary:
                latest_summary = last_summary

        active_file = str(manifest.get("active_shard_file", "") or "").strip()
        shard_names = {s.file for s in clean_shards}
        if active_file not in shard_names:
            if clean_shards:
                active_file = clean_shards[-1].file
            else:
                active_file = ""
            changed = True

        expected_next = max_turn_id + 1 if max_turn_id > 0 else 1
        if int(manifest.get("next_turn_id", 1) or 1) != expected_next:
            manifest["next_turn_id"] = expected_next
            changed = True
        manifest["active_shard_file"] = active_file
        manifest["latest_summary"] = latest_summary
        if not isinstance(manifest.get("summary_versions"), list):
            manifest["summary_versions"] = []
            changed = True
        if not isinstance(manifest.get("archived_shards"), dict):
            manifest["archived_shards"] = {}
            changed = True
        if int(manifest.get("last_compacted_turn_id", 0) or 0) < 0:
            manifest["last_compacted_turn_id"] = 0
            changed = True
        if not isinstance(manifest.get("last_compacted_at", ""), str):
            manifest["last_compacted_at"] = ""
            changed = True
        manifest["shards"] = [s.to_dict() for s in clean_shards]

        if changed:
            self._save_manifest(chat_dir, manifest)
        return {
            "chat_id": manifest["chat_id"],
            "changed": changed,
            "next_turn_id": manifest["next_turn_id"],
            "active_shard_file": manifest["active_shard_file"],
            "shard_count": len(clean_shards),
            "latest_summary": manifest["latest_summary"],
        }

    def read_manifest(self, *, chat_id: str) -> dict[str, Any]:
        self.repair_manifest(chat_id=chat_id)
        chat_dir = self._chat_dir(chat_id)
        manifest = self._load_manifest(chat_dir, chat_id=chat_id)
        shards = self._load_shards(manifest)
        return {
            "chat_id": str(manifest.get("chat_id", _safe_chat_id(chat_id))),
            "created_at": str(manifest.get("created_at", "")),
            "updated_at": str(manifest.get("updated_at", "")),
            "next_turn_id": int(manifest.get("next_turn_id", 1) or 1),
            "active_shard_file": str(manifest.get("active_shard_file", "")),
            "latest_summary": str(manifest.get("latest_summary", "")),
            "summary_versions": list(manifest.get("summary_versions", []))
            if isinstance(manifest.get("summary_versions"), list)
            else [],
            "archived_shards": manifest.get("archived_shards", {})
            if isinstance(manifest.get("archived_shards"), dict)
            else {},
            "last_compacted_turn_id": int(manifest.get("last_compacted_turn_id", 0) or 0),
            "last_compacted_at": str(manifest.get("last_compacted_at", "") or ""),
            "shards": [s.to_dict() for s in shards],
        }

    def read_global_constraints(self, *, chat_id: str) -> dict[str, Any]:
        chat_dir = self._chat_dir(chat_id)
        state = self._load_global_constraints(chat_dir)
        items = state.get("items", [])
        if not isinstance(items, list):
            items = []
        active_items = [
            x
            for x in items
            if isinstance(x, dict)
            and str(x.get("status", "active") or "active").strip().lower() == "active"
            and str(x.get("content", "") or "").strip()
        ]
        return {
            "chat_id": _safe_chat_id(chat_id),
            "version": int(state.get("version", 1) or 1),
            "updated_at": str(state.get("updated_at", "") or ""),
            "next_id": int(state.get("next_id", 1) or 1),
            "items": active_items,
        }

    def list_compaction_candidates(
        self,
        *,
        chat_id: str,
        min_shards: int = 3,
        max_candidates: int = 8,
    ) -> list[dict[str, Any]]:
        """List non-active shard candidates for compaction."""
        manifest = self.read_manifest(chat_id=chat_id)
        shards = manifest.get("shards", []) if isinstance(manifest.get("shards"), list) else []
        if len(shards) < max(2, int(min_shards)):
            return []
        active_file = str(manifest.get("active_shard_file", "") or "")
        archived = manifest.get("archived_shards", {})
        if not isinstance(archived, dict):
            archived = {}
        candidates: list[dict[str, Any]] = []
        for shard in shards:
            if not isinstance(shard, dict):
                continue
            file_name = str(shard.get("file", "") or "")
            if not file_name or file_name == active_file:
                continue
            if file_name in archived:
                continue
            candidates.append(shard)
        candidates.sort(key=lambda x: int(x.get("index", 0) or 0))
        return candidates[: max(1, int(max_candidates))]

    def mark_shards_archived(
        self,
        *,
        chat_id: str,
        shard_files: list[str],
        summary_file: str,
        summary_block_ids: list[str] | None = None,
        last_compacted_turn_id: int = 0,
    ) -> dict[str, Any]:
        """Mark shards archived in manifest with summary index mapping."""
        self.repair_manifest(chat_id=chat_id)
        chat_dir = self._chat_dir(chat_id)
        manifest = self._load_manifest(chat_dir, chat_id=chat_id)
        archived = manifest.get("archived_shards", {})
        if not isinstance(archived, dict):
            archived = {}
        block_ids = [str(x) for x in (summary_block_ids or []) if str(x)]
        now = _utc_now_iso()
        for file_name in shard_files:
            key = str(file_name or "").strip()
            if not key:
                continue
            archived[key] = {
                "summary_file": str(summary_file or ""),
                "summary_block_ids": block_ids,
                "archived_at": now,
            }
        manifest["archived_shards"] = archived
        versions = manifest.get("summary_versions", [])
        if not isinstance(versions, list):
            versions = []
        if summary_file and summary_file not in versions:
            versions.append(summary_file)
        manifest["summary_versions"] = versions
        if int(last_compacted_turn_id or 0) > int(manifest.get("last_compacted_turn_id", 0) or 0):
            manifest["last_compacted_turn_id"] = int(last_compacted_turn_id or 0)
        manifest["last_compacted_at"] = now
        self._save_manifest(chat_dir, manifest)
        return {
            "chat_id": _safe_chat_id(chat_id),
            "archived_count": len([x for x in shard_files if str(x or "").strip()]),
            "summary_file": summary_file,
            "last_compacted_turn_id": int(manifest.get("last_compacted_turn_id", 0) or 0),
            "last_compacted_at": str(manifest.get("last_compacted_at", "") or ""),
        }

    def read_summary(self, *, chat_id: str, version: str | None = None) -> dict[str, Any]:
        """Read summary file by version name, defaulting to newest version."""
        manifest = self.read_manifest(chat_id=chat_id)
        versions = manifest.get("summary_versions", [])
        if not isinstance(versions, list):
            versions = []
        if version is not None and str(version).strip():
            target = str(version).strip()
        elif versions:
            target = str(versions[-1]).strip()
        else:
            target = ""
        if not target:
            return {"chat_id": _safe_chat_id(chat_id), "summary_file": "", "blocks": []}
        chat_dir = self._chat_dir(chat_id)
        path = chat_dir / target
        if not path.exists():
            return {"chat_id": _safe_chat_id(chat_id), "summary_file": target, "blocks": []}
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return {
                "chat_id": _safe_chat_id(chat_id),
                "summary_file": target,
                "blocks": [],
                "error": {"type": "SummaryParseError"},
            }
        return {"chat_id": _safe_chat_id(chat_id), "summary_file": target, "blocks": []}

    def list_shards(self, *, chat_id: str) -> list[dict[str, Any]]:
        return self.read_manifest(chat_id=chat_id).get("shards", [])

    def _iter_turn_records(self, chat_id: str) -> list[dict[str, Any]]:
        self.repair_manifest(chat_id=chat_id)
        chat_dir = self._chat_dir(chat_id)
        manifest = self._load_manifest(chat_dir, chat_id=chat_id)
        shards = sorted(self._load_shards(manifest), key=lambda s: s.index)
        records: list[dict[str, Any]] = []
        for shard in shards:
            if not shard.file:
                continue
            shard_path = chat_dir / shard.file
            if not shard_path.exists():
                continue
            try:
                for line in shard_path.read_text(encoding="utf-8").splitlines():
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        parsed = json.loads(text)
                    except Exception:
                        continue
                    if not isinstance(parsed, dict):
                        continue
                    normalized = self._normalize_turn_record(parsed)
                    if normalized["turn_id"] <= 0:
                        continue
                    records.append(normalized)
            except Exception:
                continue
        records.sort(key=lambda x: int(x.get("turn_id", 0)))
        return records

    def replay_turns(
        self,
        *,
        chat_id: str,
        start_turn_id: int | None = None,
        end_turn_id: int | None = None,
        limit: int = 200,
        ascending: bool = True,
    ) -> list[dict[str, Any]]:
        records = self._iter_turn_records(chat_id)
        filtered: list[dict[str, Any]] = []
        for rec in records:
            tid = int(rec.get("turn_id", 0))
            if start_turn_id is not None and tid < int(start_turn_id):
                continue
            if end_turn_id is not None and tid > int(end_turn_id):
                continue
            filtered.append(rec)

        if not ascending:
            filtered = list(reversed(filtered))
        safe_limit = max(1, int(limit))
        return filtered[:safe_limit]

    def load_recent_turns(self, *, chat_id: str, last_n: int = 12) -> list[dict[str, Any]]:
        recent = self.replay_turns(
            chat_id=chat_id,
            ascending=False,
            limit=max(1, int(last_n)),
        )
        recent.reverse()
        return recent

    def _load_shards(self, manifest: dict[str, Any]) -> list[_ShardInfo]:
        result: list[_ShardInfo] = []
        for item in manifest.get("shards", []):
            if not isinstance(item, dict):
                continue
            result.append(
                _ShardInfo(
                    index=int(item.get("index", 1) or 1),
                    date=str(item.get("date", "") or _utc_date()),
                    file=str(item.get("file", "")).strip(),
                    turn_count=int(item.get("turn_count", 0) or 0),
                    size_bytes=int(item.get("size_bytes", 0) or 0),
                    first_turn_id=item.get("first_turn_id"),
                    last_turn_id=item.get("last_turn_id"),
                    latest_summary=str(item.get("latest_summary", "") or ""),
                )
            )
        return result

    def _pick_active_shard(self, chat_dir: Path, manifest: dict[str, Any]) -> _ShardInfo:
        shards = self._load_shards(manifest)
        today = _utc_date()
        active_file = str(manifest.get("active_shard_file", "") or "").strip()
        active = next((s for s in shards if s.file == active_file and s.file), None)
        if active is None:
            index = 1
            if shards:
                index = max(s.index for s in shards) + 1
            file_name = f"turns-{today}-{index:02d}.ndjson"
            active = _ShardInfo(
                index=index,
                date=today,
                file=file_name,
                turn_count=0,
                size_bytes=0,
                first_turn_id=None,
                last_turn_id=None,
                latest_summary="",
            )
            shards.append(active)
            manifest["active_shard_file"] = file_name
            manifest["shards"] = [s.to_dict() for s in shards]
            self._save_manifest(chat_dir, manifest)
            return active

        shard_path = chat_dir / active.file
        size_bytes = shard_path.stat().st_size if shard_path.exists() else 0
        needs_roll = (
            size_bytes >= self.max_shard_bytes
            or active.turn_count >= self.max_turns_per_shard
        )
        if not needs_roll:
            active.size_bytes = size_bytes
            return active

        index = max((s.index for s in shards), default=0) + 1
        new_file = f"turns-{today}-{index:02d}.ndjson"
        new_shard = _ShardInfo(
            index=index,
            date=today,
            file=new_file,
            turn_count=0,
            size_bytes=0,
            first_turn_id=None,
            last_turn_id=None,
            latest_summary="",
        )
        shards.append(new_shard)
        manifest["active_shard_file"] = new_file
        manifest["shards"] = [s.to_dict() for s in shards]
        self._save_manifest(chat_dir, manifest)
        return new_shard

    def append_turn(
        self,
        *,
        chat_id: str,
        user_id: str | None,
        run_id: str,
        task: str,
        result: dict[str, Any],
        logs: list[dict[str, Any]] | None = None,
        memory_decision: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self.repair_manifest(chat_id=chat_id)
        chat_dir = self._chat_dir(chat_id)
        manifest = self._load_manifest(chat_dir, chat_id=chat_id)
        turn_id = int(manifest.get("next_turn_id", 1) or 1)
        active = self._pick_active_shard(chat_dir, manifest)
        now = _utc_now_iso()

        response = str(result.get("response", "") or "")
        terminal_errors = self._terminal_errors(result)
        error_count = len(terminal_errors)
        control_event_count = self._control_event_count(result)
        metadata = result.get("metadata", {}) if isinstance(result.get("metadata"), dict) else {}
        execution_state = metadata.get("execution_state", {}) if isinstance(metadata.get("execution_state"), dict) else {}
        recent_tool_results = metadata.get("recent_tool_results", [])
        ok_tool_count = 0
        fail_tool_count = 0
        if isinstance(recent_tool_results, list):
            for item in recent_tool_results:
                if not isinstance(item, dict):
                    continue
                if bool(item.get("ok", False)):
                    ok_tool_count += 1
                else:
                    fail_tool_count += 1

        retrieval_summary = _summarize_for_retrieval(
            f"Task: {task}\nResponse: {response}\nErrors: {error_count}\n"
            f"Control events: {control_event_count}\n"
            f"Execution: {json.dumps(execution_state, ensure_ascii=False, default=str)}",
            max_chars=320,
        )
        execution_evidence = self._collect_execution_evidence(result)
        memory_candidates = self._build_memory_candidates(
            chat_id=chat_id,
            user_id=user_id,
            run_id=run_id,
            turn_id=turn_id,
            now=now,
            task=task,
            response=response,
            result=result,
            retrieval_summary=retrieval_summary,
        )
        memory_candidates, contract_validation_errors = self._apply_memory_traits(
            memory_candidates,
            evidence=execution_evidence,
            memory_decision=memory_decision,
        )
        committed_memories, write_policy = self._commit_candidates(
            memory_candidates,
            evidence=execution_evidence,
        )
        memory_profile = self._build_memory_profile(
            committed_memories,
            write_policy=write_policy,
            evidence=execution_evidence,
        )
        payload = {
            "memory_id": f"{_safe_chat_id(chat_id)}-turn-{turn_id}",
            "chat_id": _safe_chat_id(chat_id),
            "user_id": _safe_user_id(user_id),
            "run_id": str(run_id),
            "turn_id": turn_id,
            "timestamp": now,
            "task_preview": _summarize_for_retrieval(task, max_chars=220),
            "response_preview": _summarize_for_retrieval(response, max_chars=220),
            "retrieval_summary": retrieval_summary,
            "content_hash": "sha256:" + sha256(
                (str(run_id) + "\n" + task + "\n" + response).encode("utf-8", errors="ignore")
            ).hexdigest(),
            "memory_profile": memory_profile,
            "execution": {
                "ok_tool_count": ok_tool_count,
                "fail_tool_count": fail_tool_count,
                "error_count": error_count,
                "control_event_count": control_event_count,
                "has_response": bool(response.strip()),
                "quality_gate_decision": str(
                    (result.get("quality_gate", {}) or {}).get("decision", "")
                ),
                "goal_side_effect_done": bool(
                    execution_state.get("side_effect_done", False)
                ),
                "goal_aligned_side_effect": bool(
                    execution_state.get("goal_aligned_side_effect", False)
                ),
                "evidence": execution_evidence,
            },
            "write_policy": write_policy,
            "memory_candidates": memory_candidates,
            "committed_memories": committed_memories,
            "memory_decision": memory_decision if isinstance(memory_decision, dict) else {},
            "memory_contract_validation_errors": contract_validation_errors,
            "refs": {
                "selected_agents": list(result.get("selected_agents", []))
                if isinstance(result.get("selected_agents"), list)
                else [],
                "trace_count": len(logs or []),
                "review_count": len(result.get("review_reports", []))
                if isinstance(result.get("review_reports"), list)
                else 0,
            },
        }
        global_state = self._load_global_constraints(chat_dir)
        global_items = [
            x
            for x in global_state.get("items", [])
            if isinstance(x, dict)
            and str(x.get("status", "active") or "active").strip().lower() == "active"
            and str(x.get("content", "") or "").strip()
        ]
        payload["global_constraints"] = global_items

        shard_path = chat_dir / active.file
        line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"
        with shard_path.open("a", encoding="utf-8") as f:
            f.write(line)

        active.turn_count += 1
        active.size_bytes = shard_path.stat().st_size
        active.first_turn_id = active.first_turn_id or turn_id
        active.last_turn_id = turn_id
        active.latest_summary = retrieval_summary

        shards = self._load_shards(manifest)
        replaced = False
        for idx, shard in enumerate(shards):
            if shard.file == active.file:
                shards[idx] = active
                replaced = True
                break
        if not replaced:
            shards.append(active)

        manifest["next_turn_id"] = turn_id + 1
        manifest["latest_summary"] = retrieval_summary
        manifest["shards"] = [s.to_dict() for s in shards]
        self._save_manifest(chat_dir, manifest)

        summary_payload = {
            "chat_id": _safe_chat_id(chat_id),
            "last_turn_id": turn_id,
            "updated_at": now,
            "latest_summary": retrieval_summary,
            "latest_run_id": str(run_id),
        }
        _write_json(self._summary_path(chat_dir), summary_payload)

        return {
            "chat_id": _safe_chat_id(chat_id),
            "turn_id": turn_id,
            "memory_id": payload["memory_id"],
            "shard_file": active.file,
            "shard_size_bytes": active.size_bytes,
            "retrieval_summary": retrieval_summary,
            "committed_memory_count": len(committed_memories),
            "committed_memory_types": [
                str(item.get("memory_type", ""))
                for item in committed_memories
                if isinstance(item, dict)
            ],
            "committed_memories": committed_memories,
            "write_policy": write_policy,
            "memory_contract_validation_errors": contract_validation_errors,
            "memory_decision_used": isinstance(memory_decision, dict),
            "memory_decision_source": str(
                (memory_decision or {}).get("decision_source", "system_default_v1")
            )
            if isinstance(memory_decision, dict)
            else "none",
            "global_constraints": global_items,
            "global_constraints_count": len(global_items),
        }

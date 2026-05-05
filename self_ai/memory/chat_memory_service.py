# coding=utf-8
"""Service layer for chat replay and bounded recent-turn injection."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

from ..text_utils import shorten_text
from .chat_memory_compactor import compact_chat as _compact_chat
from .chat_memory_query import retrieve_memory_for_turn as _retrieve_memory_for_turn
from .chat_memory_store import ChatMemoryStore


class ChatMemoryService:
    """Read-focused service API for frontend/debug and prompt injection."""

    def __init__(self, store: ChatMemoryStore) -> None:
        self.store = store

    def get_manifest(self, *, chat_id: str) -> dict[str, Any]:
        self.store.repair_manifest(chat_id=chat_id)
        return self.store.read_manifest(chat_id=chat_id)

    def replay_turns(
        self,
        *,
        chat_id: str,
        start_turn_id: int | None = None,
        end_turn_id: int | None = None,
        limit: int = 200,
        ascending: bool = True,
    ) -> list[dict[str, Any]]:
        return self.store.replay_turns(
            chat_id=chat_id,
            start_turn_id=start_turn_id,
            end_turn_id=end_turn_id,
            limit=limit,
            ascending=ascending,
        )

    def list_compaction_candidates(
        self,
        *,
        chat_id: str,
        min_shards: int = 3,
        max_candidates: int = 8,
    ) -> list[dict[str, Any]]:
        return self.store.list_compaction_candidates(
            chat_id=chat_id,
            min_shards=min_shards,
            max_candidates=max_candidates,
        )

    def read_summary(
        self,
        *,
        chat_id: str,
        version: str | None = None,
    ) -> dict[str, Any]:
        return self.store.read_summary(chat_id=chat_id, version=version)

    def load_global_constraints(self, *, chat_id: str) -> dict[str, Any]:
        return self.store.read_global_constraints(chat_id=chat_id)

    @staticmethod
    def _estimate_chars(item: dict[str, Any]) -> int:
        return len(json.dumps(item, ensure_ascii=False, default=str))

    @staticmethod
    def _parse_iso_utc(value: str) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _time_decay(age_hours: float, decay_mode: str, pinned: bool) -> float:
        if pinned:
            return 1.0
        mode = str(decay_mode or "normal").strip().lower()
        if mode == "none":
            return 1.0
        half_life_hours = 72.0
        if mode == "fast":
            half_life_hours = 24.0
        elif mode == "slow":
            half_life_hours = 168.0
        exponent = max(0.0, age_hours) / half_life_hours
        return max(0.05, math.pow(0.5, exponent))

    def _score_turn(self, rec: dict[str, Any], *, now: datetime) -> tuple[float, dict[str, Any]]:
        profile = rec.get("memory_profile", {}) if isinstance(rec.get("memory_profile"), dict) else {}
        importance = float(profile.get("importance", 0.5) or 0.5)
        confidence = float(profile.get("confidence", 0.6) or 0.6)
        pinned = bool(profile.get("pinned", False))
        decay_mode = str(profile.get("decay_mode", "normal") or "normal")
        execution = rec.get("execution", {}) if isinstance(rec.get("execution"), dict) else {}
        evidence = execution.get("evidence", {}) if isinstance(execution.get("evidence"), dict) else {}
        ts = self._parse_iso_utc(str(rec.get("timestamp", "")))
        age_hours = 0.0
        if ts is not None:
            delta = now - ts
            age_hours = max(0.0, delta.total_seconds() / 3600.0)
        decay = self._time_decay(age_hours, decay_mode, pinned)
        importance_boost = 0.7 + 0.6 * max(0.0, min(importance, 1.0))
        confidence_boost = 0.75 + 0.5 * max(0.0, min(confidence, 1.0))
        requires_state_change = bool(evidence.get("requires_state_change", False))
        has_successful_execution = bool(evidence.get("has_successful_execution", False))
        has_goal_aligned_state_change = bool(
            evidence.get("has_goal_aligned_state_change", False)
        )
        reliability_boost = 1.0
        if requires_state_change and not has_goal_aligned_state_change:
            reliability_boost = 0.55
        elif has_goal_aligned_state_change:
            reliability_boost = 1.25
        elif has_successful_execution:
            reliability_boost = 1.1
        # Base semantic score is 1.0 in Phase 2.2-A because retrieval_summary embedding is not enabled yet.
        score = 1.0 * decay * importance_boost * confidence_boost * reliability_boost
        details = {
            "age_hours": round(age_hours, 3),
            "decay": round(decay, 4),
            "importance": round(max(0.0, min(importance, 1.0)), 4),
            "confidence": round(max(0.0, min(confidence, 1.0)), 4),
            "importance_boost": round(importance_boost, 4),
            "confidence_boost": round(confidence_boost, 4),
            "reliability_boost": round(reliability_boost, 4),
            "decay_mode": decay_mode,
            "pinned": pinned,
            "final_score": round(score, 6),
        }
        return score, details

    def load_recent_turns_for_injection(
        self,
        *,
        chat_id: str,
        last_n: int,
        max_chars_total: int,
        max_item_chars: int,
    ) -> dict[str, Any]:
        records = self.store.load_recent_turns(
            chat_id=chat_id,
            last_n=max(1, int(last_n)) * 6,
        )
        now = datetime.now(timezone.utc)
        scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for rec in records:
            if not isinstance(rec, dict):
                continue
            score, details = self._score_turn(rec, now=now)
            scored.append((score, rec, details))
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[: max(1, int(last_n))]

        total_budget = max(200, int(max_chars_total))
        item_budget = max(80, int(max_item_chars))
        kept: list[dict[str, Any]] = []
        used = 0
        dropped = 0

        for _score, rec, details in top:
            compact = {
                "turn_id": int(rec.get("turn_id", 0) or 0),
                "run_id": str(rec.get("run_id", "")),
                "timestamp": str(rec.get("timestamp", "")),
                "task_preview": shorten_text(str(rec.get("task_preview", "")), max_chars=item_budget),
                "response_preview": shorten_text(str(rec.get("response_preview", "")), max_chars=item_budget),
                "retrieval_summary": shorten_text(str(rec.get("retrieval_summary", "")), max_chars=item_budget),
                "execution": rec.get("execution", {}) if isinstance(rec.get("execution"), dict) else {},
                "memory_profile": rec.get("memory_profile", {})
                if isinstance(rec.get("memory_profile"), dict)
                else {},
                "score": details,
            }
            size = self._estimate_chars(compact)
            if kept and used + size > total_budget:
                dropped += 1
                continue
            if not kept and size > total_budget:
                # Keep one heavily-trimmed record at minimum.
                compact["task_preview"] = shorten_text(compact["task_preview"], max_chars=120)
                compact["response_preview"] = shorten_text(compact["response_preview"], max_chars=120)
                compact["retrieval_summary"] = shorten_text(compact["retrieval_summary"], max_chars=120)
                compact["execution"] = {
                    "ok_tool_count": int(
                        (compact.get("execution", {}) or {}).get("ok_tool_count", 0) or 0
                    ),
                    "fail_tool_count": int(
                        (compact.get("execution", {}) or {}).get("fail_tool_count", 0) or 0
                    ),
                    "error_count": int(
                        (compact.get("execution", {}) or {}).get("error_count", 0) or 0
                    ),
                }
                size = self._estimate_chars(compact)
                if size > total_budget:
                    compact["timestamp"] = ""
                    compact["execution"] = {}
                    compact["task_preview"] = shorten_text(compact["task_preview"], max_chars=80)
                    compact["response_preview"] = shorten_text(compact["response_preview"], max_chars=80)
                    compact["retrieval_summary"] = shorten_text(compact["retrieval_summary"], max_chars=80)
                    size = self._estimate_chars(compact)
                if size > total_budget:
                    # Hard floor: emit minimal shell and clamp used to budget.
                    compact = {
                        "turn_id": compact["turn_id"],
                        "run_id": compact["run_id"],
                        "retrieval_summary": shorten_text(
                            str(rec.get("retrieval_summary", "")),
                            max_chars=max(24, total_budget // 6),
                        ),
                        "memory_profile": rec.get("memory_profile", {})
                        if isinstance(rec.get("memory_profile"), dict)
                        else {},
                        "score": details,
                    }
                    size = min(self._estimate_chars(compact), total_budget)
            kept.append(compact)
            used += size

        if len(records) > len(kept):
            dropped += max(0, len(top) - len(kept))

        return {
            "chat_id": chat_id,
            "requested_turns": len(top),
            "kept_turns": len(kept),
            "dropped_turns": dropped,
            "budget": {
                "max_chars_total": total_budget,
                "max_item_chars": item_budget,
                "used_chars": used,
            },
            "turns": kept,
        }

    async def retrieve_memory_for_turn(
        self,
        *,
        chat_id: str,
        query: str,
        last_n: int,
        max_chars_total: int,
        max_item_chars: int,
        kernel: Any | None = None,
        session_id: str = "default",
        encode_func: Any | None = None,
        l2_collection: str = "cf_chat_memory",
        l2_collections: list[str] | None = None,
        l2_limit: int = 6,
        l3_limit: int = 2,
        global_constraints_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Fuse chat memory reads from L1 + L2 + L3 under strict chat isolation."""
        l1_payload = self.load_recent_turns_for_injection(
            chat_id=chat_id,
            last_n=last_n,
            max_chars_total=max_chars_total,
            max_item_chars=max_item_chars,
        )
        return await _retrieve_memory_for_turn(
            chat_id=chat_id,
            query=query,
            l1_payload=l1_payload,
            max_chars_total=max_chars_total,
            max_item_chars=max_item_chars,
            kernel=kernel,
            session_id=session_id,
            encode_func=encode_func,
            l2_collection=l2_collection,
            l2_collections=l2_collections,
            l2_limit=l2_limit,
            l3_limit=l3_limit,
            global_constraints_payload=global_constraints_payload,
        )

    def compact_chat_memory(
        self,
        *,
        chat_id: str,
        min_shards: int = 3,
        summary_max_chars: int = 1200,
        max_docs: int = 200,
    ) -> dict[str, Any]:
        return _compact_chat(
            store=self.store,
            chat_id=chat_id,
            min_shards=min_shards,
            summary_max_chars=summary_max_chars,
            max_docs=max_docs,
        )

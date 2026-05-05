# coding=utf-8
"""Phase-3 chat memory fusion read interface (session-isolated)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from ..text_utils import shorten_text


def _estimate_chars(item: dict[str, Any]) -> int:
    return len(str(item))


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso_ts(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _time_decay(*, created_at: str, decay_mode: str, pinned: bool) -> float:
    if pinned:
        return 1.0
    ts = _parse_iso_ts(created_at)
    if ts is None:
        return 0.95
    age_hours = max(0.0, (_safe_now() - ts).total_seconds() / 3600.0)
    mode = str(decay_mode or "normal").strip().lower()
    half_life = {
        "fast": 24.0,
        "normal": 24.0 * 5.0,
        "slow": 24.0 * 14.0,
        "none": 24.0 * 365.0 * 10.0,
    }.get(mode, 24.0 * 5.0)
    # Exponential half-life decay.
    return max(0.2, min(1.0, 0.5 ** (age_hours / max(1.0, half_life))))


def _final_l2_score(
    *,
    semantic_score: float,
    created_at: str,
    decay_mode: str,
    pinned: bool,
    importance: float,
    confidence: float,
) -> float:
    decay = _time_decay(created_at=created_at, decay_mode=decay_mode, pinned=pinned)
    importance_boost = 0.7 + 0.6 * max(0.0, min(importance, 1.0))
    confidence_boost = 0.75 + 0.5 * max(0.0, min(confidence, 1.0))
    return float(semantic_score * decay * importance_boost * confidence_boost)


def _extract_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("empty_json")
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(raw[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("invalid_json")


def _normalize_l2_item(
    raw: dict[str, Any],
    *,
    max_item_chars: int,
    collection_name: str,
) -> dict[str, Any]:
    metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}
    refs = {
        "chunk_id": str(raw.get("chunk_id", "") or ""),
        "source": str(raw.get("source", "") or ""),
        "path": str(raw.get("path", "") or ""),
        "symbol": str(raw.get("symbol", "") or ""),
        "collection": str(collection_name or ""),
    }
    semantic_score = round(_safe_float(raw.get("score", 0.0), 0.0), 6)
    created_at = str(raw.get("created_at", "") or "")
    decay_mode = str(metadata.get("decay_mode", "normal") or "normal")
    pinned = bool(metadata.get("pinned", False))
    importance = _safe_float(metadata.get("importance", 0.5), 0.5)
    confidence = _safe_float(metadata.get("confidence", 0.7), 0.7)
    final_score = _final_l2_score(
        semantic_score=semantic_score,
        created_at=created_at,
        decay_mode=decay_mode,
        pinned=pinned,
        importance=importance,
        confidence=confidence,
    )
    return {
        "memory_id": str(raw.get("chunk_id", "") or ""),
        "turn_id": _safe_int(metadata.get("turn_id"), 0),
        "run_id": str(metadata.get("run_id", "") or ""),
        "timestamp": created_at,
        "memory_type": str(metadata.get("memory_type", "semantic_memory") or "semantic_memory"),
        "memory_class": str(metadata.get("memory_class", "fact") or "fact"),
        "source_layer": "L2",
        "source_type": str(raw.get("source_type", "qdrant") or "qdrant"),
        "preview": shorten_text(str(raw.get("content", "") or ""), max_chars=max_item_chars),
        "score": {
            "semantic_score": semantic_score,
            "final_score": round(final_score, 6),
            "decay_mode": decay_mode,
            "pinned": pinned,
            "importance": importance,
            "confidence": confidence,
        },
        "refs": refs,
    }


def _normalize_l3_item(raw: dict[str, Any], *, max_item_chars: int) -> dict[str, Any]:
    metadata = raw.get("metadata", {}) if isinstance(raw.get("metadata"), dict) else {}
    nodes = raw.get("nodes", []) if isinstance(raw.get("nodes"), list) else []
    edges = raw.get("edges", []) if isinstance(raw.get("edges"), list) else []
    run_id = str(metadata.get("run_id", "") or "")
    preview = shorten_text(
        f"nodes={len(nodes)}, edges={len(edges)}, run_id={run_id}",
        max_chars=max_item_chars,
    )
    return {
        "memory_id": str(run_id or metadata.get("path_id", "") or "graph_path"),
        "turn_id": 0,
        "run_id": run_id,
        "timestamp": "",
        "memory_type": "graph_lineage",
        "memory_class": "decision",
        "source_layer": "L3",
        "source_type": "neo4j",
        "preview": preview,
        "score": {
            "semantic_score": 0.0,
            "final_score": 0.0,
            "decay_mode": "none",
            "pinned": False,
            "importance": 0.5,
            "confidence": 0.8,
        },
        "refs": {
            "run_id": run_id,
            "node_count": len(nodes),
            "edge_count": len(edges),
        },
    }


def _clip_items(
    items: list[dict[str, Any]],
    *,
    max_chars_total: int,
    max_item_chars: int,
) -> tuple[list[dict[str, Any]], int, int]:
    budget = max(200, int(max_chars_total))
    kept: list[dict[str, Any]] = []
    used = 0
    dropped = 0
    for item in items:
        preview = shorten_text(str(item.get("preview", "") or ""), max_chars=max_item_chars)
        clipped = dict(item)
        clipped["preview"] = preview
        size = _estimate_chars(clipped)
        if kept and used + size > budget:
            dropped += 1
            continue
        if not kept and size > budget:
            clipped["preview"] = shorten_text(preview, max_chars=max(80, budget // 4))
            size = min(_estimate_chars(clipped), budget)
        kept.append(clipped)
        used += size
    return kept, used, dropped


def _source_groups(items: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        key = str(item.get("source_layer", "unknown") or "unknown")
        out[key] = out.get(key, 0) + 1
    return out


async def retrieve_memory_for_turn(
    *,
    chat_id: str,
    query: str,
    l1_payload: dict[str, Any],
    max_chars_total: int,
    max_item_chars: int,
    kernel: Any | None,
    session_id: str,
    encode_func: Any | None,
    l2_collection: str = "cf_chat_memory",
    l2_collections: list[str] | None = None,
    l2_limit: int = 6,
    l3_limit: int = 2,
    global_constraints_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse L1/L2/L3 memory reads for one chat turn (chat-isolated)."""
    stats: dict[str, Any] = {
        "chat_id": chat_id,
        "l1_used": 0,
        "l2_used": 0,
        "l3_used": 0,
        "layer_health": {
            "l1_ok": True,
            "l2_ok": False,
            "l3_ok": False,
            "l1_degraded": False,
            "l2_degraded": False,
            "l3_degraded": False,
            "l2_error_type": "",
            "l3_error_type": "",
        },
        "degraded": False,
        "degraded_reasons": [],
        "l2_error": None,
        "l3_error": None,
        "fusion_failed": False,
        "l2_query_variants": 1,
        "l2_expand_error": None,
    }
    raw_items: list[dict[str, Any]] = []

    l1_turns = l1_payload.get("turns", []) if isinstance(l1_payload.get("turns"), list) else []
    global_constraints_items: list[dict[str, Any]] = []
    if isinstance(global_constraints_payload, dict):
        raw_global_items = global_constraints_payload.get("items", [])
        if isinstance(raw_global_items, list):
            global_constraints_items = [x for x in raw_global_items if isinstance(x, dict)]
    for turn in l1_turns:
        if not isinstance(turn, dict):
            continue
        raw_items.append(
            {
                "memory_id": str(turn.get("memory_id", "") or f"turn:{turn.get('turn_id', 0)}"),
                "turn_id": _safe_int(turn.get("turn_id"), 0),
                "run_id": str(turn.get("run_id", "") or ""),
                "timestamp": str(turn.get("timestamp", "") or ""),
                "memory_type": str(
                    (
                        turn.get("memory_profile", {})
                        if isinstance(turn.get("memory_profile"), dict)
                        else {}
                    ).get("primary_class", "turn_summary")
                    or "turn_summary"
                ),
                "memory_class": str(
                    (
                        turn.get("memory_profile", {})
                        if isinstance(turn.get("memory_profile"), dict)
                        else {}
                    ).get("primary_class", "ephemeral")
                    or "ephemeral"
                ),
                "source_layer": "L1",
                "source_type": "turn_log",
                "preview": shorten_text(
                    "\n".join(
                        x
                        for x in (
                            str(turn.get("task_preview", "") or "").strip(),
                            str(turn.get("response_preview", "") or "").strip(),
                            str(turn.get("retrieval_summary", "") or "").strip(),
                        )
                        if x
                    ),
                    max_chars=max_item_chars,
                ),
                "score": turn.get("score", {}) if isinstance(turn.get("score"), dict) else {},
                "refs": {
                    "turn_id": _safe_int(turn.get("turn_id"), 0),
                    "run_id": str(turn.get("run_id", "") or ""),
                },
            }
        )
    stats["l1_used"] = len(raw_items)
    stats["global_constraints_used"] = len(global_constraints_items)

    run_ids_from_l1 = [
        str(item.get("run_id", "") or "")
        for item in raw_items
        if str(item.get("run_id", "") or "")
    ]
    unique_run_ids: list[str] = []
    for run_id in run_ids_from_l1:
        if run_id not in unique_run_ids:
            unique_run_ids.append(run_id)

    if kernel is not None and callable(encode_func):
        stats["l2_attempted"] = True
        query_variants: list[str] = [str(query or "").strip()]
        query_variants = [x for x in query_variants if x]
        try:
            recent_turn_hints = []
            for turn in l1_turns[-3:]:
                if isinstance(turn, dict):
                    hint = str(turn.get("retrieval_summary", "") or turn.get("task_preview", "") or "").strip()
                    if hint:
                        recent_turn_hints.append(shorten_text(hint, max_chars=140))
            expand_prompt = (
                "Return JSON only with key `queries` as 2-3 concise search queries for session memory retrieval.\n"
                "Keep each query short and specific. No markdown.\n"
                f"User query: {str(query or '').strip()}\n"
                f"Recent hints: {json.dumps(recent_turn_hints, ensure_ascii=False)}\n"
            )
            expand_ret = await kernel.run_once(
                tool_name="model.generate",
                arguments={
                    "prompt": expand_prompt,
                    "stage": "chat_memory_query_expand",
                    "hints": {
                        "goal": "memory_query_expand_json",
                        "prefer_structured_json": True,
                        "force_tier": "fast",
                        "enable_thinking": False,
                        "max_tokens": 512,
                    },
                },
                session_id=session_id,
                metadata={"source": "chat_memory_l2_query_expand"},
            )
            if bool(expand_ret.get("ok", False)):
                expand_data = expand_ret.get("data", {}) if isinstance(expand_ret.get("data"), dict) else {}
                expand_raw = str(expand_data.get("response", "") or "")
                parsed = _extract_json_object(expand_raw)
                raw_queries = parsed.get("queries", [])
                if isinstance(raw_queries, str):
                    raw_queries = [raw_queries]
                if isinstance(raw_queries, list):
                    for q in raw_queries:
                        text = str(q or "").strip()
                        if text and text not in query_variants:
                            query_variants.append(text)
            else:
                err = expand_ret.get("error", {}) if isinstance(expand_ret.get("error"), dict) else {}
                stats["l2_expand_error"] = {
                    "type": str(err.get("type", "ToolCallFailed") or "ToolCallFailed"),
                    "message": str(err.get("message", ""))[:220],
                }
        except Exception as exc:
            stats["l2_expand_error"] = {"type": type(exc).__name__, "message": str(exc)[:220]}

        query_variants = query_variants[:3]
        stats["l2_query_variants"] = len(query_variants)
        dense_vectors: list[tuple[str, Any]] = []
        for q in query_variants:
            try:
                dense_vectors.append((q, encode_func(q)))
            except Exception as exc:
                stats["l2_error"] = {"type": type(exc).__name__, "message": str(exc)[:220]}
        if dense_vectors:
            collection_candidates = l2_collections if isinstance(l2_collections, list) else None
            if not collection_candidates:
                collection_candidates = [l2_collection]
            normalized_collections: list[str] = []
            for name in collection_candidates:
                text = str(name or "").strip()
                if text and text not in normalized_collections:
                    normalized_collections.append(text)
            l2_errors: list[dict[str, Any]] = []
            for collection_name in normalized_collections:
                for variant_idx, (_query_text, dense_vector) in enumerate(dense_vectors, start=1):
                    try:
                        l2_ret = await kernel.run_once(
                            tool_name="storage.semantic.search",
                            arguments={
                                "collection_name": collection_name,
                                "dense_vector": dense_vector,
                                "limit": max(1, int(l2_limit)),
                                "min_score": -1.0,
                            },
                            session_id=session_id,
                            metadata={
                                "source": "chat_memory_fusion_l2",
                                "collection": collection_name,
                                "query_variant": variant_idx,
                            },
                        )
                        if isinstance(l2_ret, dict) and not bool(l2_ret.get("ok", False)):
                            err = l2_ret.get("error", {}) if isinstance(l2_ret.get("error"), dict) else {}
                            l2_errors.append(
                                {
                                    "collection": collection_name,
                                    "variant": variant_idx,
                                    "type": str(err.get("type", "ToolCallFailed") or "ToolCallFailed"),
                                    "message": str(err.get("message", ""))[:220],
                                }
                            )
                            continue
                        l2_data = l2_ret.get("data", {}) if isinstance(l2_ret, dict) else {}
                        l2_items = l2_data.get("evidence", []) if isinstance(l2_data, dict) else []
                        if isinstance(l2_items, list):
                            for item in l2_items:
                                if isinstance(item, dict):
                                    normalized_item = _normalize_l2_item(
                                        item,
                                        max_item_chars=max_item_chars,
                                        collection_name=collection_name,
                                    )
                                    refs = normalized_item.get("refs", {})
                                    if not isinstance(refs, dict):
                                        refs = {}
                                    refs["query_variant"] = variant_idx
                                    normalized_item["refs"] = refs
                                    raw_items.append(normalized_item)
                    except Exception as exc:
                        l2_errors.append(
                            {
                                "collection": collection_name,
                                "variant": variant_idx,
                                "type": type(exc).__name__,
                                "message": str(exc)[:220],
                            }
                        )
            if l2_errors:
                stats["l2_error"] = l2_errors[0]
                stats["l2_errors"] = l2_errors
            stats["l2_used"] = len([x for x in raw_items if x.get("source_layer") == "L2"])
    else:
        stats["l2_attempted"] = False

    if kernel is not None:
        stats["l3_attempted"] = bool(unique_run_ids)
        try:
            for run_id in unique_run_ids[: max(1, int(l3_limit))]:
                l3_ret = await kernel.run_once(
                    tool_name="storage.graph.find_task_lineage",
                    arguments={"run_id": run_id, "limit": 8},
                    session_id=session_id,
                    metadata={"source": "chat_memory_fusion_l3"},
                )
                l3_data = l3_ret.get("data", {}) if isinstance(l3_ret, dict) else {}
                l3_paths = l3_data.get("paths", []) if isinstance(l3_data, dict) else []
                if isinstance(l3_paths, list):
                    for path in l3_paths:
                        if isinstance(path, dict):
                            path = dict(path)
                            path["metadata"] = {
                                **(path.get("metadata", {}) if isinstance(path.get("metadata"), dict) else {}),
                                "run_id": run_id,
                            }
                            raw_items.append(_normalize_l3_item(path, max_item_chars=max_item_chars))
            stats["l3_used"] = len([x for x in raw_items if x.get("source_layer") == "L3"])
        except Exception as exc:
            stats["l3_error"] = {"type": type(exc).__name__, "message": str(exc)[:220]}
    else:
        stats["l3_attempted"] = False

    # Cross-collection de-duplication for L2/L3 merged items (keep highest final_score).
    best_by_key: dict[str, dict[str, Any]] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        key = "|".join(
            [
                str(item.get("source_layer", "") or ""),
                str(item.get("memory_id", "") or ""),
                str(item.get("turn_id", "") or ""),
                str((item.get("refs", {}) if isinstance(item.get("refs"), dict) else {}).get("collection", "") or ""),
            ]
        )
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = item
            continue
        old_score = _safe_float(
            (existing.get("score", {}) if isinstance(existing.get("score"), dict) else {}).get("final_score"),
            0.0,
        )
        new_score = _safe_float(
            (item.get("score", {}) if isinstance(item.get("score"), dict) else {}).get("final_score"),
            0.0,
        )
        if new_score > old_score:
            best_by_key[key] = item
    deduped = list(best_by_key.values())

    deduped.sort(
        key=lambda x: _safe_float((x.get("score", {}) if isinstance(x.get("score"), dict) else {}).get("final_score"), 0.0),
        reverse=True,
    )
    items, used_chars, dropped = _clip_items(
        deduped,
        max_chars_total=max_chars_total,
        max_item_chars=max_item_chars,
    )
    stats["l2_used"] = len([x for x in items if x.get("source_layer") == "L2"])
    stats["l3_used"] = len([x for x in items if x.get("source_layer") == "L3"])
    l2_error = stats.get("l2_error")
    l3_error = stats.get("l3_error")
    l2_expand_error = stats.get("l2_expand_error")
    degraded_reasons: list[str] = []
    if l2_expand_error:
        degraded_reasons.append("l2_query_expand_failed")
    if l2_error:
        degraded_reasons.append("l2_read_failed")
    if l3_error:
        degraded_reasons.append("l3_read_failed")
    layer_health = {
        "l1_ok": True,
        "l2_ok": bool(stats.get("l2_attempted", False)) and not bool(l2_error),
        "l3_ok": (not bool(stats.get("l3_attempted", False))) or not bool(l3_error),
        "l1_degraded": False,
        "l2_degraded": bool(l2_error or l2_expand_error),
        "l3_degraded": bool(l3_error),
        "l2_error_type": str((l2_error or {}).get("type", "") if isinstance(l2_error, dict) else ""),
        "l3_error_type": str((l3_error or {}).get("type", "") if isinstance(l3_error, dict) else ""),
    }
    stats["layer_health"] = layer_health
    stats["degraded_reasons"] = degraded_reasons
    stats["degraded"] = bool(degraded_reasons)
    stats["fusion_failed"] = bool(
        (stats["l1_used"] + stats["l2_used"] + stats["l3_used"]) == 0
        and (l2_error or l3_error)
    )
    return {
        "chat_id": chat_id,
        "requested_turns": int(l1_payload.get("requested_turns", 0) or 0),
        "kept_turns": len(items),
        "dropped_turns": dropped,
        "budget": {
            "max_chars_total": max(200, int(max_chars_total)),
            "max_item_chars": max(80, int(max_item_chars)),
            "used_chars": used_chars,
        },
        "turns": items,
        "source_type_groups": _source_groups(items),
        "global_constraints": global_constraints_items,
        "stats": stats,
    }

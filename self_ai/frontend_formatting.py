# coding=utf-8
"""Safe frontend formatting helpers for Agent Cockpit."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

SENSITIVE_KEYS = (
    "api_key",
    "token",
    "secret",
    "password",
    "credential",
    "authorization",
    "cookie",
    "env",
)

TRACE_EVENTS = {
    "engine.run.start",
    "engine.pipeline.selected",
    "engine.stage.start",
    "engine.stage.end",
    "engine.stage.error",
    "agent_runtime.start",
    "agent_runtime.end",
    "agent_runtime.error",
    "revision_loop.start",
    "revision_loop.iteration.start",
    "revision_loop.iteration.end",
    "revision_loop.end",
    "tool.call.start",
    "tool.call.end",
    "tool.call.error",
    "engine.stage.tool.start",
    "engine.stage.tool.end",
    "engine.run.end",
    "engine.run.error",
    "mainloop.final_answer.blocked",
    "mainloop.final_answer.terminal_block",
    "prompt.mainloop.stats",
}


def safe_get(data: Any, path: str | list[str], default: Any = None) -> Any:
    """Safely get nested value from dict/list structures."""
    if data is None:
        return default
    keys = path.split(".") if isinstance(path, str) else list(path)
    current = data
    for key in keys:
        if isinstance(current, dict):
            if key not in current:
                return default
            current = current[key]
            continue
        if isinstance(current, list):
            try:
                index = int(key)
            except Exception:
                return default
            if index < 0 or index >= len(current):
                return default
            current = current[index]
            continue
        return default
    return current if current is not None else default


def truncate_text(text: Any, max_chars: int = 1000) -> str:
    """Return truncated text preview."""
    value = "" if text is None else str(text)
    limit = max(40, int(max_chars))
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def redact_sensitive(value: Any) -> Any:
    """Redact sensitive text values."""
    if not isinstance(value, str):
        return value
    lowered = value.lower()
    hard_markers = ("api_key", "password", "authorization", "cookie", "secret", "credential")
    if any(marker in lowered for marker in hard_markers):
        return "[REDACTED]"
    if "token=" in lowered or lowered.startswith("token ") or lowered.startswith("token:"):
        return "[REDACTED]"
    if len(value) > 12 and ("sk-" in lowered or "ms-" in lowered):
        return "[REDACTED]"
    return value


def _is_sensitive_key(key: str) -> bool:
    key_lower = key.lower()
    strict_tokens = {"api_key", "secret", "password", "credential", "authorization", "cookie", "env"}
    if key_lower in strict_tokens:
        return True
    if any(key_lower.endswith("_" + token) for token in strict_tokens):
        return True

    # token is special: redact "token" and "*_token", but allow "token_budget".
    if key_lower == "token" or key_lower.endswith("_token"):
        return True
    return False


def sanitize_payload(payload: Any) -> Any:
    """Recursively sanitize payload by redacting sensitive fields."""
    if isinstance(payload, dict):
        sanitized: dict[str, Any] = {}
        for key, value in payload.items():
            if _is_sensitive_key(str(key)):
                sanitized[str(key)] = "[REDACTED]"
            else:
                sanitized[str(key)] = sanitize_payload(value)
        return sanitized
    if isinstance(payload, list):
        return [sanitize_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return [sanitize_payload(item) for item in payload]
    return redact_sensitive(payload)


def _parse_trace_line(line: Any) -> dict[str, Any] | None:
    if isinstance(line, dict):
        return sanitize_payload(line)
    if not isinstance(line, str):
        return None
    raw = line.strip()
    if not raw:
        return None
    if raw.startswith("[self-ai]"):
        raw = raw[len("[self-ai]") :].strip()
    try:
        parsed = json.loads(raw)
        return sanitize_payload(parsed) if isinstance(parsed, dict) else None
    except Exception:
        return None


def _collect_trace_payloads(logs: Any) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in as_list(logs):
        parsed = _parse_trace_line(item)
        if isinstance(parsed, dict):
            payloads.append(parsed)
    return payloads


def _extract_pipeline_name(result: dict[str, Any], logs: list[dict[str, Any]]) -> str:
    metadata = as_dict(result.get("metadata"))
    pipeline = metadata.get("pipeline")
    if isinstance(pipeline, list) and pipeline:
        return " -> ".join(str(x) for x in pipeline)
    for payload in logs:
        if payload.get("event") == "engine.pipeline.selected":
            fields = as_dict(payload.get("fields"))
            name = fields.get("pipeline_name")
            if name:
                return str(name)
    return str(safe_get(result, "workflow_decision.execution_mode", ""))


def _elapsed_ms_from_timestamps(created_at: Any, updated_at: Any) -> int:
    if not created_at or not updated_at:
        return 0
    try:
        c = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        u = datetime.fromisoformat(str(updated_at).replace("Z", "+00:00"))
        return max(0, int((u - c).total_seconds() * 1000))
    except Exception:
        return 0


def format_run_overview(result: Any, logs: Any = None) -> dict[str, Any]:
    data = as_dict(sanitize_payload(result))
    trace_payloads = _collect_trace_payloads(logs)
    metadata = as_dict(data.get("metadata"))
    gate = as_dict(data.get("quality_gate"))

    selected_agents = [str(x) for x in as_list(data.get("selected_agents"))]
    agent_outputs = as_dict(data.get("agent_outputs"))
    elapsed_ms = int(metadata.get("elapsed_ms", 0) or 0)
    if elapsed_ms <= 0:
        elapsed_ms = _elapsed_ms_from_timestamps(data.get("created_at"), data.get("updated_at"))

    return {
        "run_id": str(data.get("run_id", "")),
        "session_id": str(data.get("session_id", "default")),
        "created_at": str(data.get("created_at", "")),
        "updated_at": str(data.get("updated_at", "")),
        "execution_mode": str(safe_get(data, "workflow_decision.execution_mode", "")),
        "pipeline_name": _extract_pipeline_name(data, trace_payloads),
        "selected_agents": selected_agents,
        "agent_output_count": len(agent_outputs),
        "revision_count": int(metadata.get("revision_count", 0) or 0),
        "max_revision_iterations": int(metadata.get("max_revision_iterations", 1) or 1),
        "revision_performed": bool(metadata.get("revision_performed", False)),
        "revision_target_agent": metadata.get("revision_target_agent"),
        "quality_gate.decision": str(gate.get("decision", "")),
        "quality_gate.passed": bool(gate.get("passed", False)),
        "quality_gate.requires_revision": bool(gate.get("requires_revision", False)),
        "error_count": len(as_list(data.get("errors"))),
        "model": str(data.get("model", "")),
        "elapsed_ms": elapsed_ms,
    }


def format_engine_timeline(result: Any, logs: Any) -> list[dict[str, Any]]:
    _ = result
    timeline: list[dict[str, Any]] = []
    for payload in _collect_trace_payloads(logs):
        event = str(payload.get("event", ""))
        if event not in TRACE_EVENTS:
            continue
        timeline.append(
            {
                "event": event,
                "ts": payload.get("ts"),
                "stage": payload.get("stage"),
                "status": payload.get("status"),
                "run_id": payload.get("run_id"),
                "session_id": payload.get("session_id"),
                "fields": as_dict(payload.get("fields")),
            }
        )
    return timeline


def format_tool_calls(result: Any, logs: Any) -> list[dict[str, Any]]:
    _ = result
    tool_states: dict[str, dict[str, Any]] = {}
    ordered_ids: list[str] = []
    active_without_call_id: dict[str, str] = {}
    seq = 0

    for payload in _collect_trace_payloads(logs):
        event = str(payload.get("event", ""))
        if not event.startswith("tool.call.") and not event.startswith("engine.stage.tool."):
            continue

        fields = as_dict(payload.get("fields"))
        tool_name = str(payload.get("tool_name") or fields.get("tool") or "")
        explicit_call_id = payload.get("call_id") or fields.get("call_id")
        if explicit_call_id:
            call_id = str(explicit_call_id)
        elif event.endswith(".start"):
            call_id = f"{tool_name or 'tool'}-{seq}"
            seq += 1
            active_without_call_id[tool_name or call_id] = call_id
        else:
            call_id = active_without_call_id.get(tool_name) or f"{tool_name or 'tool'}-{seq}"
            if call_id.endswith(f"-{seq}"):
                seq += 1
        if call_id not in tool_states:
            tool_states[call_id] = {
                "call_id": call_id,
                "tool_name": tool_name,
                "ok": None,
                "latency_ms": 0,
                "error_type": "",
                "input_preview": "",
                "output_preview": "",
                "metadata": {},
            }
            ordered_ids.append(call_id)

        item = tool_states[call_id]
        item["tool_name"] = tool_name or str(item["tool_name"])
        if event in {"tool.call.start", "engine.stage.tool.start"}:
            item["input_preview"] = truncate_text(fields.get("arguments_preview", ""), 220)
        elif event in {"tool.call.end", "engine.stage.tool.end"}:
            item["ok"] = bool(fields.get("ok", True))
            item["latency_ms"] = int(fields.get("latency_ms", 0) or 0)
            item["output_preview"] = truncate_text(fields.get("response_preview", ""), 220)
            active_without_call_id.pop(tool_name, None)
        elif event in {"tool.call.error"}:
            item["ok"] = False
            item["latency_ms"] = int(fields.get("latency_ms", 0) or 0)
            item["error_type"] = str(fields.get("error_type", "ToolError"))
            active_without_call_id.pop(tool_name, None)

        item["metadata"] = sanitize_payload(
            {
                "stage": payload.get("stage"),
                "status": payload.get("status"),
                "fields": fields,
            }
        )

    return [tool_states[key] for key in ordered_ids]


def format_runtime_signals(result: Any, logs: Any = None) -> dict[str, Any]:
    """Extract the high-signal runtime state the frontend should surface."""
    data = as_dict(sanitize_payload(result))
    metadata = as_dict(data.get("metadata"))
    execution_state = as_dict(metadata.get("execution_state"))
    chat_stats = as_dict(metadata.get("chat_recent_turns_stats"))
    layer_health = as_dict(chat_stats.get("layer_health"))
    chat_memory = as_dict(metadata.get("chat_memory"))
    write_policy = as_dict(chat_memory.get("write_policy"))
    l2l3_write = as_dict(metadata.get("chat_memory_l2l3_write"))
    sidecar = as_dict(metadata.get("chat_memory_sidecar"))
    trace_payloads = _collect_trace_payloads(logs)
    model_calls = [
        item for item in trace_payloads
        if item.get("event") == "engine.stage.tool.end"
        and as_dict(item.get("fields")).get("tool") == "model.generate"
    ]
    tool_calls = [item for item in trace_payloads if item.get("event") == "engine.stage.tool.end"]
    prompt_stats = as_dict(metadata.get("prompt_stats"))
    if not prompt_stats:
        for item in reversed(trace_payloads):
            if item.get("event") == "prompt.mainloop.stats":
                prompt_stats = as_dict(item.get("fields"))
                break

    return {
        "progress": str(execution_state.get("progress", "")),
        "goal_progress": str(execution_state.get("goal_progress", "")),
        "successful_calls": int(execution_state.get("successful_calls", 0) or 0),
        "failed_calls": int(execution_state.get("failed_calls", 0) or 0),
        "control_event_count": int(execution_state.get("control_event_count", 0) or 0),
        "terminal_error_count": int(execution_state.get("terminal_error_count", 0) or 0),
        "completion_gate_blocked_count": int(execution_state.get("completion_gate_blocked_count", 0) or 0),
        "fusion_failed": bool(chat_stats.get("fusion_failed", False)),
        "degraded": bool(chat_stats.get("degraded", False)),
        "degraded_reasons": as_list(chat_stats.get("degraded_reasons")),
        "layer_health": layer_health,
        "recent_turns_kept": int(chat_stats.get("kept_turns", 0) or 0),
        "recent_turns_requested": int(chat_stats.get("requested_turns", 0) or 0),
        "memory_committed_count": int(chat_memory.get("committed_memory_count", 0) or 0),
        "issue_write_allowed": bool(write_policy.get("issue_write_allowed", False)),
        "issue_suppressed_due_to_terminal_success": bool(
            write_policy.get("issue_suppressed_due_to_terminal_success", False)
        ),
        "sidecar_attempted": bool(sidecar.get("attempted", False)),
        "sidecar_succeeded": bool(sidecar.get("succeeded", False)),
        "sidecar_fallback_used": bool(sidecar.get("fallback_used", False)),
        "l2l3_failed": bool(metadata.get("chat_memory_l2l3_failed", False)),
        "qdrant_ok": int(l2l3_write.get("qdrant_ok", 0) or 0),
        "neo4j_ok": int(l2l3_write.get("neo4j_ok", 0) or 0),
        "model_call_count": len(model_calls),
        "tool_call_count": len(tool_calls),
        "prompt_chars_total": int(prompt_stats.get("chars_total", 0) or 0),
        "prompt_dynamic_ratio": prompt_stats.get("dynamic_ratio", 0),
    }


def format_agent_turns(result: Any) -> dict[str, Any]:
    data = as_dict(sanitize_payload(result))
    selected_agents = [str(x) for x in as_list(data.get("selected_agents"))]
    outputs = as_dict(data.get("agent_outputs"))
    turns: list[dict[str, Any]] = []
    for agent_name, raw in outputs.items():
        item = as_dict(raw)
        warnings = as_list(item.get("warnings"))
        used_context = as_list(item.get("used_context"))
        turns.append(
            {
                "agent_name": str(item.get("agent_name", agent_name)),
                "role": str(item.get("role", "")),
                "success": bool(item.get("success", item.get("ok", bool(item.get("output"))))),
                "summary": truncate_text(item.get("summary", ""), 240),
                "content_preview": truncate_text(item.get("output", item.get("content", "")), 320),
                "confidence": item.get("confidence"),
                "warnings_count": len(warnings),
                "used_context_count": len(used_context),
                "error": truncate_text(item.get("error", ""), 180),
                "metadata": sanitize_payload(as_dict(item.get("metadata"))),
            }
        )

    synthesizer = as_dict(outputs.get("synthesizer"))
    return {
        "selected_agents": selected_agents,
        "agent_output_count": len(outputs),
        "turns": turns,
        "synthesizer_output": {
            "summary": truncate_text(synthesizer.get("summary", ""), 300),
            "content_preview": truncate_text(synthesizer.get("output", ""), 400),
        },
    }


def format_revision_summary(result: Any, logs: Any = None) -> dict[str, Any]:
    data = as_dict(sanitize_payload(result))
    metadata = as_dict(data.get("metadata"))
    gate = as_dict(data.get("quality_gate"))
    reviews = as_list(data.get("review_reports"))
    latest_review = as_dict(reviews[-1]) if reviews and isinstance(reviews[-1], dict) else {}

    revision_instruction = (
        str(gate.get("revision_instruction", "") or "")
        or str(latest_review.get("revision_instruction", "") or "")
    )
    revision_events = [
        item
        for item in format_engine_timeline(data, logs)
        if str(item.get("event", "")).startswith("revision_loop.")
    ]

    return {
        "revision_count": int(metadata.get("revision_count", 0) or 0),
        "max_revision_iterations": int(metadata.get("max_revision_iterations", 1) or 1),
        "revision_performed": bool(metadata.get("revision_performed", False)),
        "revision_target_agent": metadata.get("revision_target_agent"),
        "revision_instruction_preview": truncate_text(revision_instruction, 260),
        "revision_events": revision_events,
    }


def format_review_reports(result: Any) -> dict[str, Any]:
    data = as_dict(sanitize_payload(result))
    reports = [as_dict(item) for item in as_list(data.get("review_reports")) if isinstance(item, dict)]
    rows: list[dict[str, Any]] = []
    for report in reports:
        rows.append(
            {
                "pass_review": bool(report.get("pass_review", False)),
                "severity": str(report.get("severity", "")),
                "requires_revision": bool(report.get("requires_revision", False)),
                "major_issues": [truncate_text(x, 140) for x in as_list(report.get("major_issues"))[:5]],
                "minor_issues": [truncate_text(x, 140) for x in as_list(report.get("minor_issues"))[:5]],
                "missing_requirements": [
                    truncate_text(x, 140) for x in as_list(report.get("missing_requirements"))[:5]
                ],
                "revision_target_agent": report.get("revision_target_agent"),
                "revision_instruction_preview": truncate_text(report.get("revision_instruction", ""), 220),
                "rationale_preview": truncate_text(report.get("rationale", ""), 220),
                "metadata": sanitize_payload(as_dict(report.get("metadata"))),
            }
        )

    return {
        "count": len(rows),
        "reports": rows,
    }


def format_quality_gate(result: Any) -> dict[str, Any]:
    data = as_dict(sanitize_payload(result))
    gate = as_dict(data.get("quality_gate"))
    return {
        "decision": str(gate.get("decision", "")),
        "passed": bool(gate.get("passed", False)),
        "severity": str(gate.get("severity", "")),
        "requires_revision": bool(gate.get("requires_revision", False)),
        "revision_target_agent": gate.get("revision_target_agent"),
        "current_iteration": gate.get("current_iteration"),
        "max_revision_iterations": gate.get("max_revision_iterations"),
        "blocking_issues": [truncate_text(x, 160) for x in as_list(gate.get("blocking_issues"))[:8]],
        "warnings": [truncate_text(x, 160) for x in as_list(gate.get("warnings"))[:8]],
        "rationale": truncate_text(gate.get("rationale", ""), 240),
        "metadata": sanitize_payload(as_dict(gate.get("metadata"))),
    }


def _source_type_counts(evidence_items: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in evidence_items:
        source_type = str(item.get("source_type", "unknown"))
        counts[source_type] = counts.get(source_type, 0) + 1
    return counts


def format_memory_context(result: Any) -> dict[str, Any]:
    data = as_dict(sanitize_payload(result))
    context_pack = as_dict(data.get("context_pack"))

    evidence_items = [as_dict(item) for item in as_list(context_pack.get("evidence_items")) if isinstance(item, dict)]
    graph_paths = [as_dict(item) for item in as_list(context_pack.get("graph_paths")) if isinstance(item, dict)]
    source_type_groups = sanitize_payload(as_dict(context_pack.get("source_type_groups")))
    graph_summary = as_dict(context_pack.get("graph_summary"))

    relation_counts = as_dict(graph_summary.get("relation_type_counts"))
    if not relation_counts and graph_paths:
        relation_counts = {}
        for item in graph_paths:
            relation = str(item.get("relation", item.get("edge_type", "unknown")))
            relation_counts[relation] = int(relation_counts.get(relation, 0)) + 1

    evidence_preview = [
        {
            "source_type": item.get("source_type", "unknown"),
            "score": item.get("score"),
            "content_preview": truncate_text(item.get("content", ""), 140),
        }
        for item in evidence_items[:10]
    ]

    return {
        "evidence_count": len(evidence_items),
        "source_type_groups": source_type_groups,
        "source_type_counts": _source_type_counts(evidence_items),
        "context_preview": truncate_text(context_pack.get("context_preview", ""), 500),
        "evidence_preview": evidence_preview,
        "graph_path_count": len(graph_paths),
        "graph_relation_type_counts": relation_counts,
        "graph_context_preview": truncate_text(context_pack.get("graph_context_preview", ""), 400),
        "token_budget": int(context_pack.get("token_budget", 0) or 0),
    }


def format_final_response(result: Any) -> dict[str, Any]:
    data = as_dict(sanitize_payload(result))
    return {
        "response": truncate_text(data.get("response", ""), 4000),
        "model": str(data.get("model", "")),
    }


def format_errors(result: Any, error: Any = None, logs: Any = None) -> dict[str, Any]:
    data = as_dict(sanitize_payload(result))
    result_errors = [as_dict(item) for item in as_list(data.get("errors")) if isinstance(item, dict)]
    runtime_error = as_dict(sanitize_payload(error))
    timeline = format_engine_timeline(data, logs)
    stage_errors = [item for item in timeline if item.get("event") in {"engine.stage.error", "engine.run.error"}]
    return {
        "error_count": len(result_errors),
        "errors": result_errors,
        "runtime_error": runtime_error,
        "timeline_errors": stage_errors,
    }

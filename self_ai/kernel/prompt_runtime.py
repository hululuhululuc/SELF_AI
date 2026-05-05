# coding=utf-8
"""Layered prompt builder with static-cache and dynamic-budget controls."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..config import settings
from ..memory.memory_contract import build_memory_contract_brief
from ..text_utils import shorten_text

SYSTEM_PROMPT_DYNAMIC_BOUNDARY = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__"


class PromptRuntime:
    """Build compact stage-specific prompts without external side effects."""

    def __init__(
        self,
        *,
        max_chars: int = 6000,
        cache_enabled: bool = True,
        cache_max_entries: int = 64,
        prompt_version: str = "v2-layered",
        dynamic_budget_ratio: float = 0.35,
        recent_errors: int = 3,
        recent_tool_results: int = 4,
        selected_agents_max: int = 5,
        tool_catalog_max: int = 30,
    ) -> None:
        self.max_chars = max(1000, int(max_chars))
        self.cache_enabled = bool(cache_enabled)
        self.cache_max_entries = max(1, int(cache_max_entries))
        self.prompt_version = str(prompt_version)
        self.dynamic_budget_ratio = max(0.1, min(float(dynamic_budget_ratio), 0.8))
        self.recent_errors = max(1, int(recent_errors))
        self.recent_tool_results = max(1, int(recent_tool_results))
        self.selected_agents_max = max(1, int(selected_agents_max))
        self.tool_catalog_max = max(1, int(tool_catalog_max))

        self._static_cache: dict[str, str] = {}
        self._static_cache_order: list[str] = []
        self._last_build_stats: dict[str, Any] = {}

    def _safe(self, value: Any, default: str = "") -> str:
        if value is None:
            return default
        if hasattr(value, "value"):
            return str(value.value)
        return str(value)

    def _as_dict(self, value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else {}

    def _as_list(self, value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    def _limit(self, text: str, *, max_chars: int | None = None) -> str:
        return shorten_text(text, max_chars=max_chars or self.max_chars)

    def _safe_json(self, value: Any, *, max_chars: int = 1600) -> str:
        text = json.dumps(value, ensure_ascii=False, default=str)
        return text if len(text) <= max_chars else text[:max_chars] + "...(truncated)"

    def _dyn_json_budget(self, *, ratio: float, floor: int, ceiling: int) -> int:
        estimate = int(self.max_chars * ratio)
        return max(floor, min(estimate, ceiling))

    def _cache_get(self, key: str) -> str | None:
        if not self.cache_enabled:
            return None
        value = self._static_cache.get(key)
        if value is None:
            return None
        if key in self._static_cache_order:
            self._static_cache_order.remove(key)
        self._static_cache_order.append(key)
        return value

    def _cache_put(self, key: str, value: str) -> None:
        if not self.cache_enabled:
            return
        self._static_cache[key] = value
        if key in self._static_cache_order:
            self._static_cache_order.remove(key)
        self._static_cache_order.append(key)
        while len(self._static_cache_order) > self.cache_max_entries:
            evict = self._static_cache_order.pop(0)
            self._static_cache.pop(evict, None)

    def consume_last_build_stats(self) -> dict[str, Any]:
        stats = dict(self._last_build_stats)
        self._last_build_stats = {}
        return stats

    def _tool_catalog_fingerprint(self, tools: list[dict[str, Any]]) -> str:
        reduced: list[dict[str, Any]] = []
        for tool in tools[: self.tool_catalog_max]:
            if not isinstance(tool, dict):
                continue
            schema = self._as_dict(tool.get("input_schema"))
            reduced.append(
                {
                    "name": self._safe(tool.get("name")),
                    "permission": self._safe(tool.get("permission")),
                    "required": self._as_list(schema.get("required"))[:12],
                    "keys": sorted(list(self._as_dict(schema.get("properties")).keys()))[:30],
                }
            )
        digest = hashlib.sha1(
            json.dumps(reduced, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return digest[:12]

    def _build_mainloop_core_system(self, *, mode: str = "decide") -> str:
        mode = str(mode or "decide").strip().lower()
        if mode == "action":
            output_schema = (
                "{\n"
                '  "type": "tool_calls",\n'
                '  "calls": [\n'
                '    {"tool_name":"...", "arguments": {...}, "reason":"..."}\n'
                "  ],\n"
                '  "execution_mode": "optional short label",\n'
                '  "workflow_decision": {"optional":"light summary"},\n'
                '  "plan": {"goal":"...", "next_step":"..."},\n'
                '  "goal_contract": {"intent_kind":"state_change|read_only|analysis","target_paths":["optional"]}\n'
                "}\n\n"
            )
            decision_rule = (
                "- This is ACTION_STEP. You MUST return tool_calls.\n"
                "- Never return final_answer in ACTION_STEP.\n"
                "- If read/state-change proof is missing, choose tool calls that can produce proof.\n"
            )
        elif mode == "answer":
            output_schema = (
                "{\n"
                '  "type": "final_answer",\n'
                '  "response": "final answer",\n'
                '  "execution_mode": "optional short label",\n'
                '  "workflow_decision": {"optional":"light summary"},\n'
                '  "completion": {\n'
                '    "outcome": "completed|blocked|failed",\n'
                '    "needs_side_effect_proof": true|false,\n'
                '    "needs_read_proof": true|false,\n'
                '    "is_complete": true|false,\n'
                '    "evidence_note": "brief proof note",\n'
                '    "evidence_refs": ["recent evidence_ref ids"],\n'
                '    "remaining_blockers": ["brief blocker descriptions"]\n'
                "  }\n"
                "}\n\n"
            )
            decision_rule = (
                "- This is ANSWER_STEP. You MUST return final_answer.\n"
                "- Base answer strictly on current tool evidence and execution flags.\n"
                "- If required proof is still missing, return NOT_COMPLETED with exact blocker.\n"
            )
        else:
            output_schema = (
                "{\n"
                '  "type": "final_answer" | "tool_calls",\n'
                '  "response": "required when type=final_answer",\n'
                '  "calls": [\n'
                '    {"tool_name":"...", "arguments": {...}, "reason":"..."}\n'
                "  ],\n"
                '  "execution_mode": "optional short label",\n'
                '  "workflow_decision": {\n'
                '    "use_quick_answer": true|false,\n'
                '    "requires_context": true|false,\n'
                '    "selected_agents": ["optional"]\n'
                "  },\n"
                '  "plan": {\n'
                '    "goal": "what this turn is trying to accomplish",\n'
                '    "success_criteria": "how to know it is done",\n'
                '    "next_step": "single next step summary"\n'
                "  },\n"
                '  "goal_contract": {\n'
                '    "intent_kind": "state_change" | "read_only" | "analysis",\n'
                '    "target_paths": ["optional path list"]\n'
                "  },\n"
                '  "completion": {\n'
                '    "outcome": "completed|blocked|failed",\n'
                '    "needs_side_effect_proof": true|false,\n'
                '    "needs_read_proof": true|false,\n'
                '    "is_complete": true|false,\n'
                '    "evidence_note": "brief proof note",\n'
                '    "evidence_refs": ["recent evidence_ref ids"],\n'
                '    "remaining_blockers": ["brief blocker descriptions"]\n'
                "  },\n"
                '  "constraint_maintenance": {\n'
                '    "enabled": true|false,\n'
                '    "reason": "short reason when current turn may add/update/delete durable session constraints"\n'
                "  }\n"
                "}\n\n"
            )
            decision_rule = "- Decide whether to return final_answer or tool_calls.\n"
        return (
            "# Role\n"
            "You are the execution brain of Self AI.\n"
            + decision_rule
            + "\n"
            "# Output (JSON only)\n"
            + output_schema
            + "\n"
            "# Constraints\n"
            "- Return JSON only, no markdown, no code fences.\n"
            "- Use the same language as the user's task unless explicitly asked to switch language.\n"
            "- Max 3 tool calls in one turn.\n"
            "- Prefer minimal calls and fast convergence.\n"
            "- Decide naturally: return final_answer directly when the task can be completed by answering, explaining, planning, confirming, or updating conversational/session preferences.\n"
            "- Return tool_calls first when the task requires external action or verification, such as creating/editing/deleting files, reading workspace state, running tests, querying storage, or inspecting project files.\n"
            "- Once you call tools, your final_answer must be grounded in recent_tool_results.\n"
            "- For current active session preferences/constraints, Session Global Constraints is authoritative over older retrieved chat memories.\n"
            "- If Session Global Constraints is (none), do not infer an active session preference from older memory unless the user explicitly asks about historical memory.\n"
            "- If file path is uncertain, call workspace.file.list first.\n"
            "- Before calling workspace.file.read/list, check existing workspace_snapshot and recent tool results.\n"
            "- If snapshot/version has not changed for the same target, do NOT repeat workspace.file.read/list.\n"
            "- For workspace.file.read/list calls, include a concrete target path and reason.\n"
            "- For conceptual architecture/research/system-summary tasks without explicit file targets, prefer storage.semantic.search and avoid broad workspace scanning.\n"
            "- Do NOT call internal plumbing tools: storage.runtime.* and storage.semantic.upsert_*.\n"
            "- Memory persistence is handled by sidecar/runtime; focus on task tools and verifiable evidence.\n"
            "- Do not create files solely to persist conversational memory such as preferences, style rules, facts, or future instructions.\n"
            "- Only use workspace write/edit/create when the user explicitly asks for a workspace/file artifact or mutation.\n"
            "- For storage.semantic.search, collection_name must be one of: "
            "cf_chat_memory, cf_review_memory, cf_task_memory, cf_doc_chunks, cf_code_chunks, cf_web_chunks.\n"
            "- Never use placeholder collection names like default/preferences.\n"
            "- Never invent tool names.\n"
            "- Use execution_state + recent_tool_results.execution_flag as truth for progress.\n"
            "- If completion_gate_feedback.missing is non-empty, satisfy missing evidence first.\n"
            "- Set completion.outcome=completed only when recent_tool_results prove the requested work succeeded.\n"
            "- Set completion.outcome=blocked or failed when recent_tool_results prove the requested work cannot currently be completed; cite exact failed evidence and blockers.\n"
            "- For final_answer with read/state-change requirements, provide completion.evidence_refs.\n"
            "- Do not set completion proof requirements for pure conversation, explanation, planning, or preference-maintenance turns.\n"
            "- If completion is not yet verified, do NOT claim completion; return tool_calls.\n"
            "- Never claim file/state changes unless tool receipts confirm state_change_committed=true.\n"
            "- Never claim read-based conclusions unless tool receipts confirm verified_read=true.\n"
            "- Use goal_contract.target_paths as hard targets for side-effect tools.\n"
            "- If writing/editing/creating, tool arguments must match goal_contract target path(s).\n"
            "- If goal_contract.intent_kind is state_change and no successful mutation yet, choose a mutation tool first.\n"
            "- If completion requires external action, do not emit final_answer before proof.\n"
            "\n"
            "# Session Global Constraint Policy\n"
            "- A numbered Session Global Constraints list may appear in dynamic context.\n"
            "- Treat that list as standing user preferences for this chat session.\n"
            "- Apply relevant items when planning, choosing tool arguments, deciding response style, and forming the final answer.\n"
            "- For current active session preferences/constraints, this list is authoritative over older retrieved chat memories.\n"
            "- If the list is (none), do not infer an active session preference from older memory unless the user explicitly asks about historical memory.\n"
            "- Current user instructions override stored session constraints when they conflict.\n"
            "- System and tool safety rules override both current user instructions and stored session constraints.\n"
            "- If you intentionally do not apply a relevant session constraint, state the reason briefly in workflow_decision.rationale.\n"
            "- Do not create, edit, or delete memory files yourself; session constraint maintenance is handled separately.\n"
            "\n"
            "# Session Constraint Maintenance Trigger\n"
            "- Set constraint_maintenance.enabled=true only when the current turn appears to add, revise, delete, or clarify a durable session-level preference or constraint.\n"
            "- Durable constraints include standing user preferences, response style rules, project conventions, recurring output format rules, and long-lived boundaries.\n"
            "- Set constraint_maintenance.enabled=false for ordinary one-off task content.\n"
        )

    def _build_tool_behavior_section(self, tools: list[dict[str, Any]]) -> str:
        lines = [
            "# Tool Behavior Contract",
            "- Use EXACT argument keys from each input_schema.",
            "- If a tool call fails repeatedly, choose another tool or finalize answer.",
            "- Prefer read tools before write tools unless task explicitly requires writing.",
        ]
        for tool in tools[: self.tool_catalog_max]:
            if not isinstance(tool, dict):
                continue
            name = self._safe(tool.get("name"))
            desc = shorten_text(self._safe(tool.get("description")), max_chars=140)
            perm = self._safe(tool.get("permission"))
            schema = self._as_dict(tool.get("input_schema"))
            required = [self._safe(x) for x in self._as_list(schema.get("required"))[:8]]
            props = list(self._as_dict(schema.get("properties")).keys())[:12]
            lines.append(f"- {name} [{perm}] :: {desc}")
            if required:
                lines.append(f"  required_args: {', '.join(required)}")
            if props:
                lines.append(f"  args_keys: {', '.join(props)}")
        return "\n".join(lines)

    def _compact_errors(self, errors: list[Any]) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        for item in errors[-self.recent_errors :]:
            if not isinstance(item, dict):
                continue
            metadata = self._as_dict(item.get("metadata"))
            if self._safe(metadata.get("signal_class")).strip().lower() == "control_flow":
                continue
            if self._safe(item.get("type")) == "CompletionGateBlocked":
                continue
            compacted.append(
                {
                    "type": self._safe(item.get("type")),
                    "stage": self._safe(item.get("stage")),
                    "message": shorten_text(self._safe(item.get("message")), max_chars=140),
                }
            )
        return compacted

    def _compact_control_events(self, events: list[Any]) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        for item in events[-3:]:
            if not isinstance(item, dict):
                continue
            compacted.append(
                {
                    "type": self._safe(item.get("type")),
                    "stage": self._safe(item.get("stage")),
                    "reason": shorten_text(self._safe(item.get("reason")), max_chars=120),
                }
            )
        return compacted

    def _compact_tool_results(self, rows: list[Any]) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        for row in rows[-self.recent_tool_results :]:
            if not isinstance(row, dict):
                continue
            compacted.append(
                {
                    "evidence_ref": self._safe(row.get("evidence_ref")),
                    "tool_name": self._safe(row.get("tool_name")),
                    "ok": bool(row.get("ok", False)),
                    "error_type": self._safe(self._as_dict(row.get("error")).get("type")),
                    "data_preview": shorten_text(self._safe(row.get("data_preview")), max_chars=180),
                }
            )
        return compacted

    def _compact_recent_turns(self, rows: list[Any], *, max_turns: int = 5) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        for row in rows[-max_turns:]:
            if not isinstance(row, dict):
                continue
            compacted.append(
                {
                    "turn_id": int(row.get("turn_id", 0) or 0),
                    "run_id": self._safe(row.get("run_id")),
                    "task_preview": shorten_text(self._safe(row.get("task_preview")), max_chars=300),
                    "response_preview": shorten_text(self._safe(row.get("response_preview")), max_chars=300),
                    "retrieval_summary": shorten_text(
                        self._safe(row.get("retrieval_summary")),
                        max_chars=600,
                    ),
                    "execution": self._as_dict(row.get("execution")),
                }
            )
        return compacted

    def _build_mainloop_dynamic_context(self, state: dict[str, Any], context: dict[str, Any]) -> str:
        task = self._safe(state.get("normalized_task") or state.get("task"))
        run_id = self._safe(state.get("run_id"))
        session_id = self._safe(state.get("session_id"), "default")
        turn_index = int(context.get("turn_index", 1) or 1)
        selected_agents = [self._safe(x) for x in self._as_list(state.get("selected_agents"))[: self.selected_agents_max]]
        errors = self._compact_errors(self._as_list(state.get("errors")))
        metadata = self._as_dict(state.get("metadata"))
        control_events = self._compact_control_events(self._as_list(metadata.get("control_events")))
        recent_results = self._compact_tool_results(self._as_list(context.get("recent_tool_results")))
        execution_state = self._as_dict(context.get("execution_state"))
        last_model_plan = self._as_dict(context.get("last_model_plan"))
        goal_contract = self._as_dict(context.get("goal_contract"))
        completion_gate_feedback = self._as_dict(context.get("completion_gate_feedback"))
        context_compaction = self._as_dict(context.get("context_compaction"))
        workspace_snapshot = self._as_dict(context.get("workspace_snapshot"))
        workspace_snapshot_version = int(context.get("workspace_snapshot_version", 0) or 0)
        last_parse_error = shorten_text(self._safe(context.get("last_parse_error")), max_chars=180)
        recent_turn_limit = max(5, int(getattr(settings, "chat_memory_recent_turns", 16) or 16))
        recent_turns = self._compact_recent_turns(
            self._as_list(metadata.get("chat_recent_turns")),
            max_turns=recent_turn_limit,
        )
        global_constraints = self._as_list(metadata.get("chat_global_constraints"))
        constraint_lines: list[str] = []
        for idx, row in enumerate(global_constraints[:24], start=1):
            if not isinstance(row, dict):
                continue
            content = self._safe(row.get("content") or row.get("preview"))
            if not content:
                continue
            constraint_lines.append(f"{idx}. {shorten_text(content, max_chars=240)}")
        session_global_constraints = "\n".join(constraint_lines) if constraint_lines else "(none)"
        recent_turns_stats = self._as_dict(metadata.get("chat_recent_turns_stats"))
        recent_turns_budget_data = self._as_dict(metadata.get("chat_recent_turns_budget"))
        memory_capability = {
            "scope": "session_isolated_only",
            "layers": ["L1_turn_log", "L2_qdrant", "L3_neo4j"],
            "read_collections": [
                str(getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory") or "cf_chat_memory"),
                "cf_review_memory",
                "cf_task_memory",
                "cf_doc_chunks",
                "cf_code_chunks",
                "cf_web_chunks",
            ],
            "write_router_hint": {
                "preference_constraint_decision_fact": str(
                    getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory") or "cf_chat_memory"
                ),
                "issue_failure": "cf_review_memory",
            },
            "sensitive_data_rule": "do_not_emit_raw_memory_content",
            "memory_contract": build_memory_contract_brief(
                chat_collection=str(
                    getattr(settings, "chat_memory_l2_collection_default", "cf_chat_memory")
                    or "cf_chat_memory"
                ),
                max_chars=480,
            ),
        }
        execution_state_budget = self._dyn_json_budget(ratio=0.10, floor=1200, ceiling=30000)
        goal_contract_budget = self._dyn_json_budget(ratio=0.06, floor=900, ceiling=16000)
        compaction_budget = self._dyn_json_budget(ratio=0.03, floor=400, ceiling=12000)
        snapshot_budget = self._dyn_json_budget(ratio=0.12, floor=1000, ceiling=50000)
        recent_turns_budget_chars = self._dyn_json_budget(ratio=0.22, floor=1200, ceiling=80000)
        errors_budget = self._dyn_json_budget(ratio=0.08, floor=900, ceiling=30000)
        control_events_budget = self._dyn_json_budget(ratio=0.04, floor=300, ceiling=10000)
        tool_results_budget = self._dyn_json_budget(ratio=0.20, floor=1200, ceiling=70000)
        model_plan_budget = self._dyn_json_budget(ratio=0.08, floor=900, ceiling=30000)
        stats_budget = self._dyn_json_budget(ratio=0.04, floor=300, ceiling=12000)
        lines = [
            SYSTEM_PROMPT_DYNAMIC_BOUNDARY,
            "# Dynamic Context",
            f"Run ID: {run_id}",
            f"Session ID: {session_id}",
            f"Turn: {turn_index}",
            f"Task: {task}",
            f"Selected agents: {', '.join(selected_agents) if selected_agents else '(none)'}",
            f"Execution state: {self._safe_json(execution_state, max_chars=execution_state_budget)}",
            f"Goal contract: {self._safe_json(goal_contract, max_chars=goal_contract_budget)}",
            f"Completion gate feedback: {self._safe_json(completion_gate_feedback, max_chars=goal_contract_budget)}",
            f"Workspace snapshot version: {workspace_snapshot_version}",
            f"Workspace snapshot: {self._safe_json(workspace_snapshot, max_chars=snapshot_budget)}",
            f"Context compaction: {self._safe_json(context_compaction, max_chars=compaction_budget)}",
            f"Recent chat turns stats: {self._safe_json(recent_turns_stats, max_chars=stats_budget)}",
            f"Recent chat turns budget: {self._safe_json(recent_turns_budget_data, max_chars=stats_budget)}",
            "Session Global Constraints:\n" + session_global_constraints,
            f"Recent chat turns: {self._safe_json(recent_turns, max_chars=recent_turns_budget_chars)}",
            f"Memory capability brief: {self._safe_json(memory_capability, max_chars=4000)}",
            f"Last model plan: {self._safe_json(last_model_plan, max_chars=model_plan_budget)}",
            f"Known errors: {self._safe_json(errors, max_chars=errors_budget)}",
            f"Control events: {self._safe_json(control_events, max_chars=control_events_budget)}",
            f"Recent tool results: {self._safe_json(recent_results, max_chars=tool_results_budget)}",
        ]
        if last_parse_error:
            lines.append(f"Last parse error: {last_parse_error}")
        return "\n".join(lines)

    def _build_mainloop_prompt(self, state: dict[str, Any], context: dict[str, Any], *, mode: str = "decide") -> str:
        tools = [tool for tool in self._as_list(context.get("tools")) if isinstance(tool, dict)]
        fp = self._tool_catalog_fingerprint(tools)
        static_key = f"mainloop:{mode}:{self.prompt_version}:{fp}"
        static_hit = False
        static_part = self._cache_get(static_key)
        if static_part is None:
            core_part = self._build_mainloop_core_system(mode=mode)
            tool_part = self._build_tool_behavior_section(tools)
            static_part = "\n\n".join(
                [
                    core_part,
                    tool_part,
                ]
            )
            self._cache_put(static_key, static_part)
        else:
            static_hit = True

        dynamic_part = self._build_mainloop_dynamic_context(state, context)
        max_dynamic_chars = max(400, int(self.max_chars * self.dynamic_budget_ratio))
        dynamic_part = self._limit(dynamic_part, max_chars=max_dynamic_chars)
        separator = "\n\n"
        reserved_for_dynamic = min(len(dynamic_part), max_dynamic_chars)
        static_budget = max(300, self.max_chars - reserved_for_dynamic - len(separator))
        if "Tool Behavior Contract" in static_part and len(static_part) > static_budget:
            core_part, tool_part = static_part.split("\n\n# Tool Behavior Contract", 1)
            tool_part = "# Tool Behavior Contract" + tool_part
            tool_budget = min(len(tool_part), max(300, static_budget // 3))
            core_budget = max(100, static_budget - tool_budget - len(separator))
            static_part = (
                self._limit(core_part, max_chars=core_budget)
                + separator
                + self._limit(tool_part, max_chars=tool_budget)
            )
        else:
            static_part = self._limit(static_part, max_chars=static_budget)
        prompt = static_part + separator + dynamic_part
        if len(prompt) > self.max_chars:
            # Keep dynamic context intact; trim static prefix first.
            overflow = len(prompt) - self.max_chars
            static_part = self._limit(static_part, max_chars=max(100, len(static_part) - overflow))
            prompt = static_part + separator + dynamic_part
        prompt = self._limit(prompt, max_chars=self.max_chars)

        self._last_build_stats = {
            "stage": f"mainloop_{mode}",
            "prompt_version": self.prompt_version,
            "static_cache_hit": static_hit,
            "chars_total": len(prompt),
            "chars_static": min(len(static_part), len(prompt)),
            "chars_dynamic": min(len(dynamic_part), len(prompt)),
            "dynamic_ratio": round(min(len(dynamic_part), len(prompt)) / max(len(prompt), 1), 4),
            "recent_errors_used": self.recent_errors,
            "recent_tool_results_used": self.recent_tool_results,
            "tool_catalog_used": min(len(tools), self.tool_catalog_max),
            "tool_catalog_fingerprint": fp,
        }
        return prompt

    def build_prompt(
        self,
        state: dict[str, Any],
        stage: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        context = context or {}
        task = self._safe(state.get("normalized_task") or state.get("task"))
        decision = self._as_dict(state.get("workflow_decision"))
        profile = self._as_dict(state.get("task_profile"))
        plan = self._as_dict(state.get("plan"))
        context_pack = self._as_dict(state.get("context_pack"))
        agent_outputs = self._as_dict(state.get("agent_outputs"))
        review_reports = self._as_list(state.get("review_reports"))
        gate = self._as_dict(state.get("quality_gate"))

        if stage == "mainloop_decide":
            return self._build_mainloop_prompt(state, context, mode="decide")
        if stage == "mainloop_action":
            return self._build_mainloop_prompt(state, context, mode="action")
        if stage == "mainloop_answer":
            return self._build_mainloop_prompt(state, context, mode="answer")

        if stage == "quick_answer":
            prompt = (
                "Answer the user task directly and concisely.\n"
                "Do not use markdown tables.\n"
                f"Task:\n{task}"
            )
            return self._limit(prompt, max_chars=1200)

        if stage == "analyze_task":
            prompt = (
                "Classify the user task. Return JSON only with keys: "
                "intent, domain, requires_code, requires_repo_context, requires_research, "
                "requires_planning, requires_review, expected_artifact, complexity, risk_level, confidence, rationale.\n"
                "No prose.\n"
                f"Task:\n{task}"
            )
            return self._limit(prompt, max_chars=1800)

        if stage == "plan":
            prompt = (
                "Return JSON only with keys: task_summary, execution_mode, rationale, steps.\n"
                "Each step needs: step_id, goal, agent_name, depends_on, expected_output, done_criteria.\n"
                f"Task: {task}\n"
                f"Intent: {self._safe(profile.get('intent'))}\n"
                f"Complexity: {self._safe(profile.get('complexity'))}\n"
                f"Risk: {self._safe(profile.get('risk_level'))}\n"
                f"Execution mode: {self._safe(decision.get('execution_mode'))}"
            )
            return self._limit(prompt, max_chars=2200)

        if stage == "research":
            prompt = (
                "Summarize what to retrieve for this task in one concise line.\n"
                f"Task: {task}\n"
                f"Intent: {self._safe(profile.get('intent'))}"
            )
            return self._limit(prompt, max_chars=1200)

        if stage == "role_agents":
            prompt = (
                "Role-agent execution context summary:\n"
                f"Task: {task}\n"
                f"Execution mode: {self._safe(decision.get('execution_mode'))}\n"
                f"Plan steps: {len(self._as_list(plan.get('steps')))}\n"
                f"Evidence count: {len(self._as_list(context_pack.get('evidence_items')))}"
            )
            return self._limit(prompt, max_chars=1600)

        if stage == "review":
            prompt = (
                "Review context summary:\n"
                f"Task: {task}\n"
                f"Agent outputs: {len(agent_outputs)}\n"
                f"Plan steps: {len(self._as_list(plan.get('steps')))}"
            )
            return self._limit(prompt, max_chars=1500)

        if stage == "quality_gate_context":
            latest = review_reports[-1] if review_reports and isinstance(review_reports[-1], dict) else {}
            prompt = (
                "Quality gate context:\n"
                f"pass_review={latest.get('pass_review')} severity={latest.get('severity')} "
                f"major={len(self._as_list(latest.get('major_issues')))} "
                f"missing={len(self._as_list(latest.get('missing_requirements')))}"
            )
            return self._limit(prompt, max_chars=1000)

        if stage == "final_response":
            synth = self._as_dict(agent_outputs.get("synthesizer"))
            synth_output = self._safe(synth.get("output"))
            evidence_count = len(self._as_list(context_pack.get("evidence_items")))
            prompt = (
                "Provide the final response using available summaries.\n"
                f"Task: {task}\n"
                f"Execution mode: {self._safe(decision.get('execution_mode'))}\n"
                f"Evidence count: {evidence_count}\n"
                f"Plan step count: {len(self._as_list(plan.get('steps')))}\n"
                f"Quality gate: {self._safe(gate.get('decision'))}\n"
                f"Synthesizer candidate: {shorten_text(synth_output, max_chars=800)}"
            )
            return self._limit(prompt, max_chars=self.max_chars)

        return self._limit(f"Task:\n{task}", max_chars=1200)
